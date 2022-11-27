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
