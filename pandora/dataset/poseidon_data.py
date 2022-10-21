from enum import Enum
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import json

import pandora.dataset.dataset_utils as dataset_utils
from pandora.packaging.feature import TrainingType, ModelType
from pandora.tools.common import logger


class DatasetJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, TagValidationResultType):
            return o.__dict__
        if isinstance(o, PartitionDistribution):
            return o.__dict__
        if isinstance(o, PartitionResult):
            return o.__dict__
        return DatasetJSONEncoder(self, o)


class TagValidationResultType(str, Enum):
    tag_not_found = "tag_not_found"
    not_enough_data = "not_enough_data"
    valid = "valid"


@dataclass
class PartitionDistribution(object):
    # distribution
    all: int
    train: int
    dev: int
    test: int

    # percentiles
    train_p: float
    dev_p: float
    test_p: float


@dataclass
class PartitionResult(object):
    tag_id: str
    tag_name: str
    result: TagValidationResultType
    partition_distribution: PartitionDistribution

    def __init__(self,
                 tag_id: str,
                 tag_name: str,
                 result: TagValidationResultType,
                 partition_distribution: Dict = None) -> None:
        self.tag_id = tag_id
        self.tag_name = tag_name
        self.result = result
        self.partition_distribution = partition_distribution


def load_poseidon_dataset_file(
        dataset_path: str):

    with open(dataset_path, "r", encoding='utf-8') as dataset_f:
        dataset = json.load(dataset_f)
    config = dataset["dataset_config"]

    # get training type
    training_type = TrainingType(dataset["data_type"])
    logger.info(f"training_type is {training_type}")

    # get model_type type
    model_type = ModelType(dataset["model_type"])
    logger.info(f"model_type is {model_type}")

    # get metadata types
    meta_data_types = dataset["metadata_type"]
    logger.info(f"metadata_types are {meta_data_types}")

    return dataset, config, training_type, model_type, meta_data_types


def load_poseidon_dataset(
        dataset_path: str):

    dataset, config, training_type, _, _ = load_poseidon_dataset_file(
        dataset_path)

    # check tag_ids in dataset definition
    tag_ids = config["tag_ids"]
    if len(set(tag_ids)) != len(tag_ids):
        raise ValueError(
            f"tag_ids in dataset config have duplicated ids. ids: {tag_ids}")

    # JSON limitation: convert tag_id to from string to int
    tags_by_id = {int(tag_id_str): tag
                  for tag_id_str, tag in dataset["tags"].items()}

    valid_tag_ids = []
    invalid_tag_ids = []
    for config_tag_id in tag_ids:
        # ensure that tag_id found in dataset_config.tag_ids are also found in dataset.tags
        if config_tag_id in tags_by_id:
            valid_tag_ids.append(config_tag_id)
        else:
            invalid_tag_ids.append(config_tag_id)

    # group data by tag_ids and column_ids
    col_data = dataset["column_data"]
    data_by_tag_ids = {}
    for col_id, column_data in col_data.items():

        col_tag_ids = column_data["tag_ids"]

        for col_tag_id in col_tag_ids:
            # check if tag_id is valid
            if col_tag_id not in valid_tag_ids:
                continue

            # add tag data if not presented
            if col_tag_id not in data_by_tag_ids:
                data_by_tag_ids[col_tag_id] = {}
            tag_data = data_by_tag_ids[col_tag_id]

            # add column data if not presented
            if col_id not in tag_data:
                tag_data[col_id] = []
            col_text = tag_data[col_id]

            tag_name = tags_by_id[col_tag_id]["name"]
            label_ids = [create_unique_label_name(
                tag_name=tag_name, tag_id=col_tag_id)]

            data_entries = create_data_entries_by_training_type(
                training_type=training_type,
                label_ids=label_ids,
                tag_name=tag_name,
                column_data=column_data,
            )
            col_text.extend(data_entries)
    return valid_tag_ids, invalid_tag_ids, tags_by_id, data_by_tag_ids, dataset


def partition_poseidon_dataset(
        dataset_path: str,
        output_dir: str,
        min_samples: int,
        data_ratios: Dict,
        seed: int):

    # load dataset file
    valid_tag_ids, invalid_tag_ids, tags_by_id, data_by_tag_ids, dataset = load_poseidon_dataset(
        dataset_path)

    # get training type
    training_type = TrainingType(dataset["data_type"])
    logger.info(f"training_type is {training_type}")

    # added up partitions for all tags and columns
    partitions_train = []
    partitions_dev = []
    partitions_test = []

    # outputs
    valid_tags = []
    invalid_tags = [PartitionResult(invalid_tag_id, "", TagValidationResultType.tag_not_found)
                    for invalid_tag_id in invalid_tag_ids]
    # check tags
    final_labels = []
    for valid_tag_id in valid_tag_ids:
        tag_name = tags_by_id[valid_tag_id]["name"]
        # Chec if dataset has any column ID belongs to the current tag.
        if valid_tag_id not in data_by_tag_ids:
            invalid_tags.append(
                PartitionResult(
                    valid_tag_id, tag_name, TagValidationResultType.not_enough_data))
        else:
            # get column data, validate and write data by tag
            tag_data = data_by_tag_ids[valid_tag_id]
            valid, data_partitions, distribution = _validate_and_partition_tag_data(
                tag_data, training_type, min_samples, data_ratios=data_ratios, seed=seed)
            tag_dir = os.path.join(output_dir, str(valid_tag_id))
            os.makedirs(tag_dir, exist_ok=True)
            dataset_utils.write_partitions(data_partitions, tag_dir)

            # create result objects.
            if valid:
                # build overall partition
                partitions_train.extend(data_partitions["train"])
                partitions_dev.extend(data_partitions["dev"])
                partitions_test.extend(data_partitions["test"])

                # create valid result
                result = PartitionResult(
                    valid_tag_id,
                    tag_name,
                    TagValidationResultType.valid,
                    distribution)
                valid_tags.append(result)

                # add label
                final_labels.append(
                    create_unique_label_name(tag_name=tag_name, tag_id=valid_tag_id))
            else:
                result = PartitionResult(
                    valid_tag_id,
                    tag_name,
                    TagValidationResultType.not_enough_data,
                    distribution)
                invalid_tags.append(result)

        # write merged partitions
        data_partitions_all = {
            "train": partitions_train,
            "dev": partitions_dev,
            "test": partitions_test,
        }
        dataset_utils.write_partitions(
            data_partitions_all, output_dir=output_dir)
        distribution = get_partition_distribution(
            data_partitions=data_partitions_all)
        # dump labels
        dataset_utils.write_labels(output_dir=output_dir, labels=final_labels)
    summary = {
        # TODO: Use dataclass
        "distribution": distribution,
        "valid_tags": valid_tags,
        "invalid_tags": invalid_tags,
    }
    return summary


def create_data_entries_by_training_type(
        training_type: TrainingType,
        label_ids: List[str],
        tag_name: str,
        column_data: dict):

    meta_data = column_data["metadata"]

    if training_type == TrainingType.column_data or \
            training_type == TrainingType.mixed_data:
        # Add data entry
        # TODO: Hard Coded list
        data_entries = []
        for data_entry in column_data["recognition_data"]:
            tag_name = tag_name,
            data_entries.append(dataset_utils.DataEntry(
                text=data_entry["content"],
                label=label_ids,
                meta_data=meta_data,
            ))
        return data_entries
    elif training_type == TrainingType.meta_data:
        return [dataset_utils.DataEntry(
            text="",
            label=label_ids,
            meta_data=meta_data,
        )]
    else:
        raise ValueError


def create_unique_label_name(tag_name, tag_id):
    # return f"{tag_name}_{tag_id}"
    return f"{tag_name}"


def _validate_and_partition_tag_data(tag_data, training_type, min_samples, data_ratios: List, seed: int) -> Tuple[bool, Dict, PartitionDistribution]:
    # TODO: provide multiple partition/selection functions
    all_samples = []
    for col_id, col_data in tag_data.items():
        all_samples.extend(col_data)

    # TODO: get a better strategy for meta_data training
    if training_type == TrainingType.meta_data:
        data_partitions = {
            "train": all_samples,
            "dev": all_samples,
            "test": all_samples
        }
        return True, data_partitions, get_partition_distribution(data_partitions)

    valid = len(all_samples) >= min_samples
    data_partitions = dataset_utils.split_dataset(
        all_samples=all_samples, data_ratios=data_ratios, seed=seed)
    return valid, data_partitions, get_partition_distribution(data_partitions=data_partitions)


def get_partition_distribution(data_partitions):
    num_entries = sum([len(partition)
                      for partition in data_partitions.values()])
    num_train = len(data_partitions["train"])
    num_dev = len(data_partitions["dev"])
    num_test = len(data_partitions["test"])
    return PartitionDistribution(
        all=num_entries,
        train=len(data_partitions["train"]),
        dev=len(data_partitions["dev"]),
        test=len(data_partitions["test"]),
        train_p=1.0 * num_train / num_entries if num_train > 0 else 0,
        dev_p=1.0 * num_dev / num_entries if num_dev > 0 else 0,
        test_p=1.0 * num_test / num_entries if num_test > 0 else 0,
    )
