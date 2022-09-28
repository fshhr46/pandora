
from ts.torch_handler.base_handler import BaseHandler

import transformers
import torch
from abc import ABC
import json
import logging
import os

# load pandora python dependencies
from . import feature
from . import inference
from tokenizer import SentenceTokenizer
from model import BertForSentence

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

        # self.device = torch.device(
        #     "cuda:" + str(properties.get("gpu_id"))
        #     if torch.cuda.is_available() and properties.get("gpu_id") is not None
        #     else "cpu"
        # )
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
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
            self.model = BertForSentence.from_pretrained(
                model_dir)
            self.model.to(self.device)
        else:
            logger.warning("Missing the checkpoint or state_dict.")

        # Load tokenizer
        self.tokenizer = SentenceTokenizer.from_pretrained(
            model_dir, do_lower_case=self.setup_config["do_lower_case"])

        self.model.eval()
        logger.info(
            "Transformer model from path %s loaded successfully", model_dir)

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        assert os.path.isfile(mapping_file_path)
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                index2name = json.load(f)
                self.label2id = {label: int(i)
                                 for i, label in index2name.items()}
                self.id2label = {int(i): label
                                 for i, label in index2name.items()}
        else:
            logger.warning("Missing the index_to_name.json file.")
        self.initialized = True

    def add_input_text_to_batch(
            self,
            input_ids_batch,
            attention_mask_batch,
            segment_ids_batch,
            max_seq_length: int,
            request_data):

        feat = feature.extract_feature_from_request(
            request_data,
            self.label2id,
            max_seq_length,
            self.tokenizer,
        )

        input_ids = feat.input_ids[None, :].to(self.device)
        attention_mask = feat.input_mask[None, :].to(self.device)
        segment_ids = feat.segment_ids[None, :].to(self.device)

        # making a batch out of the recieved requests
        # attention masks are passed for cases where input tokens are padded.
        if input_ids.shape is not None:
            # must be 2D tensor (batch_size, seq_length)
            assert len(input_ids.shape) == 2
            if input_ids_batch is None:
                input_ids_batch = input_ids
                attention_mask_batch = attention_mask
                segment_ids_batch = segment_ids
            else:
                input_ids_batch = torch.cat(
                    (input_ids_batch, input_ids), 0)
                attention_mask_batch = torch.cat(
                    (attention_mask_batch, attention_mask), 0
                )
                segment_ids_batch = torch.cat(
                    (segment_ids_batch, segment_ids), 0
                )
        return (input_ids_batch, attention_mask_batch, segment_ids_batch)

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
        segment_ids_batch = None
        indexes = []
        num_texts = 0
        for idx, request_data in enumerate(requests):
            if "data" in request_data and "body" in request_data:
                raise ValueError(
                    "Expect one of field in [data, body] to be set, but got both")
            elif "data" in request_data:
                # Get request expects every reqeusts has one input string
                input_ids_batch, attention_mask_batch, segment_ids_batch = self.add_input_text_to_batch(
                    input_ids_batch,
                    attention_mask_batch,
                    segment_ids_batch,
                    max_seq_length=self.setup_config["max_length"],
                    request_data=request_data)
                num_texts += 1
                indexes.append(num_texts)
            elif "body" in request_data:
                # Post request expects a list of text from the same column and predict the most likely class for the batch.
                # [
                #   {"data": "sentence 1", "column_name": "col_1"},
                #   {"data": "sentence 2", "column_name": "col_2"},
                # ]
                json_objs = request_data.get("body")
                for json_obj in json_objs:
                    input_ids_batch, attention_mask_batch, segment_ids_batch = self.add_input_text_to_batch(
                        input_ids_batch,
                        attention_mask_batch,
                        segment_ids_batch,
                        max_length=self.setup_config["max_length"],
                        request_seq_data=json_obj)
                    num_texts += 1
                indexes.append(num_texts)
            else:
                # empty request, continue
                continue
        # Batch definition
        # input_ids_batch, attention_mask_batch, segment_ids_batch, indexes
        return (input_ids_batch, attention_mask_batch, segment_ids_batch, indexes)

    def inference(self, input_batch):
        with torch.no_grad():
            return inference.run_inference(
                self.setup_config["mode"], self.model, self.id2label, input_batch)

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(self, input_batch, request_data, target):
        with torch.no_grad():
            return inference.run_get_insights(
                # configs
                mode=self.setup_config["mode"],
                embedding_name=self.setup_config["embedding_name"],
                captum_explanation=self.setup_config["captum_explanation"],
                max_seq_length=self.setup_config["max_length"],
                # model related
                model=self.model,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                # device
                device=self.device,
                # input related
                input_batch=input_batch,
                request_data=request_data,
                target=target)
