from typing import List

from pandora.packaging.feature import (
    TrainingType,
    MetadataType,
)
from pandora.packaging.model import BertBaseModelType
from pandora.tools.common import logger


def get_num_epochs(
        training_type: TrainingType,
        meta_data_types: List[str],
        bert_model_type: BertBaseModelType):
    # if num_epochs is not passed, set num_epochs by training type
    if training_type == TrainingType.column_data:
        num_epochs = 4
    elif training_type == TrainingType.mixed_data:
        num_epochs = 2
    elif training_type == TrainingType.meta_data:
        if bert_model_type == BertBaseModelType.char_bert and \
            len(meta_data_types) == 1 and \
                meta_data_types[0] == MetadataType.column_name:
            num_epochs = 15
        else:
            num_epochs = 30
    else:
        raise ValueError
    logger.info(f"num_epochs: {num_epochs}")
    return num_epochs


def get_batch_size(
        training_type: TrainingType,
        meta_data_types: List[str],
        bert_model_type: BertBaseModelType):
    # if num_epochs is not passed, set num_epochs by training type
    if training_type == TrainingType.column_data:
        batch_size = 24
    elif training_type == TrainingType.mixed_data:
        batch_size = 24
    elif training_type == TrainingType.meta_data:
        if bert_model_type == BertBaseModelType.char_bert and \
            len(meta_data_types) == 1 and \
                meta_data_types[0] == MetadataType.column_name:
            batch_size = 6
        else:
            batch_size = 24
    else:
        raise ValueError
    logger.info(f"batch_size: {batch_size}")
    return batch_size


def get_mode_type_and_name(
        training_type: TrainingType,
        meta_data_types: List[str]):

    # Set base model name
    if training_type == TrainingType.meta_data and \
            len(meta_data_types) == 1 and \
            meta_data_types[0] == MetadataType.column_name:
        bert_base_model_name = "char-bert"
    else:
        bert_base_model_name = "bert-base-chinese"

    # Set base model type
    if bert_base_model_name == "char-bert":
        bert_model_type = BertBaseModelType.char_bert
    else:
        bert_model_type = BertBaseModelType.bert

    return bert_model_type, bert_base_model_name
