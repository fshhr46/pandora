
from ts.torch_handler.base_handler import BaseHandler

import transformers
import torch
from abc import ABC
import json
import logging
import os

# load pandora python dependencies
import feature
import inference
from constants import CHARBERT_CHAR_VOCAB
from tokenizer import SentenceTokenizer

from model import (
    BertForSentence,
    BertBaseModelType,
)
from char_bert_model import CharBertForSequenceClassification
from transformers import BertTokenizer
from classifier import ClassifierType

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

        # Load bert base model name and model type
        self.bert_base_model_name = self.setup_config["bert_base_model_name"]
        self.bert_model_type = self.setup_config["bert_model_type"]
        self.include_char_data = self.bert_model_type == BertBaseModelType.char_bert

        # TODO: Write a function that gets config/tokenizer/model
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        if self.setup_config["save_mode"] == "pretrained":
            classifier_type = self.setup_config["classifier_type"]
            classifier_cls = ClassifierType.get_classifier_cls(
                classifier_type=classifier_type,
            )
            if self.bert_model_type == BertBaseModelType.char_bert:
                # Note that charbert has 2 times the hidden size
                # Load model
                self.model = CharBertForSequenceClassification.from_pretrained(
                    model_dir, classifier_cls)
                # Load tokenizer
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_dir, do_lower_case=self.setup_config["do_lower_case"])
            else:
                # Load model
                self.model = BertForSentence.from_pretrained(
                    model_dir, classifier_cls)
                # Load tokenizer
                self.tokenizer = SentenceTokenizer.from_pretrained(
                    model_dir, do_lower_case=self.setup_config["do_lower_case"])
        else:
            logger.warning("Missing the checkpoint or state_dict.")
        self.model.to(self.device)

        # set model training type
        self.training_type = self.setup_config["training_type"]
        self.meta_data_types = self.setup_config["meta_data_types"]

        # get insights setup
        # TODO: remove this hard coded number
        self.n_steps = 1

        # char bert setup
        if self.bert_model_type == BertBaseModelType.char_bert:
            self.char2ids_dict = feature.load_char_vocab(CHARBERT_CHAR_VOCAB)
        else:
            self.char2ids_dict = None

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
            char_input_ids_batch,
            start_ids_batch,
            end_ids_batch,
            max_seq_length: int,
            request_data):

        feat = feature.extract_feature_from_request(
            request_data,
            self.training_type,
            self.meta_data_types,
            self.label2id,
            max_seq_length,
            self.tokenizer,
            char2ids_dict=self.char2ids_dict,
        )

        input_ids = feat.input_ids[None, :].to(self.device)
        attention_mask = feat.input_mask[None, :].to(self.device)
        segment_ids = feat.segment_ids[None, :].to(self.device)

        # add char-bert tensors
        if self.include_char_data:
            char_input_ids = feat.char_input_ids[None, :].to(self.device)
            start_ids = feat.start_ids[None, :].to(self.device)
            end_ids = feat.end_ids[None, :].to(self.device)
        else:
            char_input_ids, start_ids, end_ids = None, None, None

        # making a batch out of the recieved requests
        # attention masks are passed for cases where input tokens are padded.
        if input_ids.shape is not None:
            # must be 2D tensor (batch_size, seq_length)
            assert len(input_ids.shape) == 2
            if input_ids_batch is None:
                input_ids_batch = input_ids
                attention_mask_batch = attention_mask
                segment_ids_batch = segment_ids

                # add char-bert tensors
                if self.include_char_data:
                    char_input_ids_batch = char_input_ids
                    start_ids_batch = start_ids
                    end_ids_batch = end_ids
            else:
                input_ids_batch = torch.cat(
                    (input_ids_batch, input_ids), 0)
                attention_mask_batch = torch.cat(
                    (attention_mask_batch, attention_mask), 0
                )
                segment_ids_batch = torch.cat(
                    (segment_ids_batch, segment_ids), 0
                )
                # add char-bert tensors
                if self.include_char_data:
                    char_input_ids_batch = torch.cat(
                        (char_input_ids_batch, char_input_ids), 0
                    )
                    start_ids_batch = torch.cat(
                        (start_ids_batch, start_ids), 0
                    )
                    end_ids_batch = torch.cat(
                        (end_ids_batch, end_ids), 0
                    )
        return (input_ids_batch, attention_mask_batch, segment_ids_batch,
                char_input_ids_batch, start_ids_batch, end_ids_batch)

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
        char_input_ids_batch = None
        start_ids_batch = None
        end_ids_batch = None
        indexes = []
        num_texts = 0
        for idx, request_data in enumerate(requests):
            logger.info(f"request_data is {request_data}")
            if "data" in request_data and "body" in request_data:
                raise ValueError(
                    "Expect one of field in [data, body] to be set, but got both")
            elif "body" in request_data:
                # Post request expects a list of text from the same column and predict the most likely class for the batch.
                # [
                #   {"data": "sentence 1", "column_name": "col_1"},
                #   {"data": "sentence 2", "column_name": "col_2"},
                # ]
                json_objs = request_data.get("body")
                for json_obj in json_objs:
                    input_ids_batch, attention_mask_batch, segment_ids_batch, \
                        char_input_ids_batch, start_ids_batch, end_ids_batch = self.add_input_text_to_batch(
                            input_ids_batch,
                            attention_mask_batch,
                            segment_ids_batch,
                            char_input_ids_batch,
                            start_ids_batch,
                            end_ids_batch,
                            max_length=self.setup_config["max_length"],
                            request_seq_data=json_obj)
                num_texts += 1
                indexes.append(num_texts)
            # Handle single request
            # {
            #   "data": "sentence 1",
            #   "column_name": "col_1",
            #   "column_comment": "com_1",
            #   "column_description": "des_1"
            # }
            else:
                # Get request expects every reqeusts has one input string
                input_ids_batch, attention_mask_batch, segment_ids_batch, \
                    char_input_ids_batch, start_ids_batch, end_ids_batch = self.add_input_text_to_batch(
                        input_ids_batch,
                        attention_mask_batch,
                        segment_ids_batch,
                        char_input_ids_batch,
                        start_ids_batch,
                        end_ids_batch,
                        max_seq_length=self.setup_config["max_length"],
                        request_data=request_data)
                num_texts += 1
                indexes.append(num_texts)
        # Batch definition
        # input_ids_batch, attention_mask_batch, segment_ids_batch, indexes
        if self.include_char_data:
            return (input_ids_batch, attention_mask_batch, segment_ids_batch,
                    char_input_ids_batch, start_ids_batch, end_ids_batch,
                    indexes)
        else:
            return (input_ids_batch, attention_mask_batch, segment_ids_batch, indexes)

    def inference(self, input_batch_with_index):
        with torch.no_grad():
            # with index removed
            indexes = input_batch_with_index[-1]
            input_batch = input_batch_with_index[:-1]
            inputs = feature.build_inputs_from_batch(
                batch=input_batch, include_labels=False, include_char_data=self.include_char_data)
            logits_list, sigmoids_list = inference.run_inference(
                inputs,
                self.setup_config["mode"],
                self.model)
            formated_outputs = inference.format_outputs(
                logits_list=logits_list, sigmoids_list=sigmoids_list, id2label=self.id2label)
            # if indexes is passed, merge results (column level inferencing)
            if indexes:
                formated_outputs = inference.merge_outputs(
                    formated_outputs=formated_outputs, indexes=indexes)
            return formated_outputs

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def get_insights(self, input_batch, request_data, target_id):
        target_id = int(target_id.decode("utf-8"))
        assert target_id in self.id2label, f"target_id not found in {self.id2label}"
        target = self.id2label[target_id]
        logger.info(f"getting insights for target {target}")
        with torch.no_grad():
            return inference.run_get_insights(
                # configs
                mode=self.setup_config["mode"],
                embedding_name=self.setup_config["embedding_name"],
                # model related
                model=self.model,
                tokenizer=self.tokenizer,
                # device
                device=self.device,
                # input related
                input_batch=input_batch,
                target=target_id,
                n_steps=self.n_steps,
                include_char_data=self.include_char_data)
