from captum.attr import LayerIntegratedGradients
from ts.torch_handler.base_handler import BaseHandler
from transformers import GPT2TokenizerFast
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
import transformers
import torch
from abc import ABC
import json
import logging
import os
import ast
from typing import Dict

# import sys
# import pathlib
# curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
# sys.path.append(os.path.join(curr_dir, "pandora.zip"))
# from pandora.packaging.tokenizer import SentenceTokenizer

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        # Loading the shared object of compiled Faster Transformer Library if Faster Transformer is set
        if self.setup_config["FasterTransformer"]:
            faster_transformer_complied_path = os.path.join(
                model_dir, "libpyt_fastertransformer.so"
            )
            torch.classes.load_library(faster_transformer_complied_path)

        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(
                model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"] == "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir
                )
            elif self.setup_config["mode"] == "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_dir)
            else:
                logger.warning("Missing the operation mode.")
            # HF GPT2 models options can be gpt2, gpt2-medium, gpt2-large, gpt2-xl
            # this basically palce different model blocks on different devices,
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L962
            if (
                self.setup_config["model_parallel"]
                and "gpt2" in self.setup_config["model_name"]
            ):
                self.model.parallelize()
            else:
                self.model.to(self.device)

        else:
            logger.warning("Missing the checkpoint or state_dict.")

        if "gpt2" in self.setup_config["model_name"]:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", pad_token="<|endoftext|>"
            )

        elif any(
            fname
            for fname in os.listdir(model_dir)
            if fname.startswith("vocab.") and os.path.isfile(fname)
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        # self.tokenizer = SentenceTokenizer.from_pretrained(
        #     model_dir, do_lower_case=self.setup_config["do_lower_case"])

        self.model.eval()
        logger.info(
            "Transformer model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.id2label = json.load(f)
        else:
            logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    def add_input_text_to_batch(
            self,
            input_ids_batch,
            attention_mask_batch,
            input_text):
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        if self.setup_config["captum_explanation"]:
            input_text_target = ast.literal_eval(input_text)
            input_text = input_text_target["text"]
        logger.info(f"input_text is {input_text}")

        max_length = self.setup_config["max_length"]
        logger.info("Received text: '%s'", input_text)
        # preprocessing text for sequence_classification, token_classification or text_generation

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=int(max_length),
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        # making a batch out of the recieved requests
        # attention masks are passed for cases where input tokens are padded.
        if input_ids.shape is not None:
            if input_ids_batch is None:
                input_ids_batch = input_ids
                attention_mask_batch = attention_mask
            else:
                input_ids_batch = torch.cat(
                    (input_ids_batch, input_ids), 0)
                attention_mask_batch = torch.cat(
                    (attention_mask_batch, attention_mask), 0
                )
        return (input_ids_batch, attention_mask_batch)

    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        indexes = []
        num_texts = 0
        for idx, data in enumerate(requests):
            if "data" in data and "body" in data:
                raise ValueError(
                    "Expect one of field in [data, body] to be set, but got both")
            elif "data" in data:
                # Get request expects every reqeusts has one input string
                input_text = data.get("data")
                input_ids_batch, attention_mask_batch = self.add_input_text_to_batch(
                    input_ids_batch,
                    attention_mask_batch,
                    input_text)
                num_texts += 1
                indexes.append(num_texts)
            elif "body" in data:
                # Post request expects a list of text from the same column and predict the most likely class for the batch.
                # [
                #   {"text": "sentence 1"},
                #   {"text": "sentence 2"},
                # ]
                json_objs = data.get("body")
                for json_obj in json_objs:
                    input_ids_batch, attention_mask_batch = self.add_input_text_to_batch(
                        input_ids_batch,
                        attention_mask_batch,
                        json_obj["text"])
                    num_texts += 1
                indexes.append(num_texts)
            else:
                # empty request, continue
                continue
        return (input_ids_batch, attention_mask_batch, indexes)

    def inference(self, input_batch):
        return run_inference(
            self.setup_config["mode"], self.model, self.id2label, input_batch, self.tokenizer)

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(self, input_batch, text, target):
        """This function initialize and calls the layer integrated gradient to get word importance
        of the input text if captum explanation has been selected through setup_config
        Args:
            input_batch (int): Batches of tokens IDs of text
            text (str): The Text specified in the input request
            target (int): The Target can be set to any acceptable label under the user's discretion.
        Returns:
            (list): Returns a list of importances and words.
        """

        if self.setup_config["captum_explanation"]:
            embedding_layer = getattr(
                self.model, self.setup_config["embedding_name"])
            embeddings = embedding_layer.embeddings
            self.lig = LayerIntegratedGradients(
                captum_sequence_forward, embeddings)
        else:
            logger.warning(
                "Captum Explanation is not chosen and will not be available")

        if isinstance(text, (bytes, bytearray)):
            text = text.decode("utf-8")
        text_target = ast.literal_eval(text)

        if not self.setup_config["mode"] == "question_answering":
            text = text_target["text"]
        self.target = text_target["target"]

        input_ids, ref_input_ids, attention_mask = construct_input_ref(
            text, self.tokenizer, self.device, self.setup_config["mode"]
        )
        all_tokens = get_word_token(input_ids, self.tokenizer)
        response = {}
        response["words"] = all_tokens
        if (
            self.setup_config["mode"] == "sequence_classification"
            or self.setup_config["mode"] == "token_classification"
        ):

            attributions, delta = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )

            attributions_sum = summarize_attributions(attributions)
            response["importances"] = attributions_sum.tolist()
            response["delta"] = delta[0].tolist()

        elif self.setup_config["mode"] == "question_answering":
            attributions_start, delta_start = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 0, self.model),
                return_convergence_delta=True,
            )
            attributions_end, delta_end = self.lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                target=self.target,
                additional_forward_args=(attention_mask, 1, self.model),
                return_convergence_delta=True,
            )
            attributions_sum_start = summarize_attributions(attributions_start)
            attributions_sum_end = summarize_attributions(attributions_end)
            response["importances_answer_start"] = attributions_sum_start.tolist()
            response["importances_answer_end"] = attributions_sum_end.tolist()
            response["delta_start"] = delta_start[0].tolist()
            response["delta_end"] = delta_end[0].tolist()
        return [response]


def run_inference(mode: str, model, id2label: Dict[str, str], input_batch, tokenizer):
    """Predict the class (or classes) of the received text using the
    serialized transformers checkpoint.
    Args:
        input_batch (list): List of Text Tensors from the pre-process function is passed here
    Returns:
        list : It returns a list of the predicted value for the input text
    """
    input_ids_batch, attention_mask_batch, indexes = input_batch
    inferences = []
    # Handling inference for sequence_classification.
    if mode == "sequence_classification":
        predictions = model(input_ids_batch, attention_mask_batch)
        logger.info(
            f"This the output size from the Seq classification model {predictions[0].size()}")
        logger.info(
            f"This the output from the Seq classification model {predictions}")

        num_rows, num_cols = predictions[0].shape
        for i in range(num_rows):
            out = predictions[0][i].unsqueeze(0)
            y_hat = out.argmax(1).item()
            y_softmax = torch.softmax(out, 1)[0]
            predicted_idx = str(y_hat)
            named_softmax = {}
            for idx, prob in enumerate(y_softmax.tolist()):
                name = id2label[str(idx)]
                named_softmax[name] = prob
            inferences.append({
                "class": id2label[predicted_idx],
                "probability": named_softmax[id2label[predicted_idx]],
                "softmax": named_softmax,
            })
    # Handling inference for token_classification.
    elif mode == "token_classification":
        raise NotImplementedError

    # merge inferences output for column level prediction requests.
    start = 0
    merged_inferences = []
    for index in indexes:
        sub_inferences = inferences[start:index+1]
        if len(sub_inferences) == 1:
            merged_inferences.append(sub_inferences[0])
        else:
            names = [inf["class"] for inf in sub_inferences]
            merged_output = merge_inferences(names)
            merged_inferences.append(merged_output)
        start = index
    return merged_inferences


def merge_inferences(inferences):
    # TODO: replace naive implementation, simply take the max count.
    max_count_name = max(set(inferences), key=inferences.count)
    return {
        "class": max_count_name,
        "probability": 1.0 * inferences.count(max_count_name) / len(inferences),
        "softmax": [],
    }


def construct_input_ref(text, tokenizer, device, mode):
    """For a given text, this function creates token id, reference id and
    attention mask based on encode which is faster for captum insights
    Args:
        text (str): The text specified in the input request
        tokenizer (AutoTokenizer Class Object): To word tokenize the input text
        device (cpu or gpu): Type of the Environment the server runs on.
    Returns:
        input_id(Tensor): It attributes to the tensor of the input tokenized words
        ref_input_ids(Tensor): Ref Input IDs are used as baseline for the attributions
        attention mask() :  The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them.
    """
    if mode == "question_answering":
        question_context = ast.literal_eval(text)
        question = question_context["question"]
        context = question_context["context"]
        text_ids = tokenizer.encode(
            question, context, add_special_tokens=False)

    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    logger.info("text_ids %s", text_ids)
    logger.info("[tokenizer.cls_token_id] %s", [tokenizer.cls_token_id])
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    logger.info("input_ids %s", input_ids)

    input_ids = torch.tensor([input_ids], device=device)
    # construct reference token ids
    ref_input_ids = (
        [tokenizer.cls_token_id]
        + [tokenizer.pad_token_id] * len(text_ids)
        + [tokenizer.sep_token_id]
    )
    ref_input_ids = torch.tensor([ref_input_ids], device=device)
    # construct attention mask
    attention_mask = torch.ones_like(input_ids)
    return input_ids, ref_input_ids, attention_mask


def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    """This function is used to get the predictions from the model and this function
    can be used independent of the type of the BERT Task.
    Args:
        inputs (list): Input for Predictions
        attention_mask (list, optional): The attention mask is a binary tensor indicating the position
         of the padded indices so that the model does not attend to them, it defaults to None.
        position (int, optional): Position depends on the BERT Task.
        model ([type], optional): Name of the model, it defaults to None.
    Returns:
        list: Prediction Outcome
    """
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred


def summarize_attributions(attributions):
    """Summarises the attribution across multiple runs
    Args:
        attributions ([list): attributions from the Layer Integrated Gradients
    Returns:
        list : Returns the attributions after normalizing them.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_word_token(input_ids, tokenizer):
    """constructs word tokens from token id using the BERT's
    Auto Tokenizer
    Args:
        input_ids (list): Input IDs from construct_input_ref method
        tokenizer (class): The Auto Tokenizer Pre-Trained model object
    Returns:
        (list): Returns the word tokens
    """
    indices = input_ids[0].detach().tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices)
    # Remove unicode space character from BPE Tokeniser
    tokens = [token.replace("Ġ", "") for token in tokens]
    return tokens
