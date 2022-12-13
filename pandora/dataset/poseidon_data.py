from enum import Enum
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import json

import pandora.dataset.dataset_utils as dataset_utils
from pandora.packaging.feature import TrainingType, ModelTaskType
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
    model_type = ModelTaskType(dataset["model_type"])
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
        num_folds: int,
        seed: int):

    # load dataset file
    valid_tag_ids, invalid_tag_ids, tags_by_id, data_by_tag_ids, dataset = load_poseidon_dataset(
        dataset_path)

    # save data partition args
    data_partition_args = {
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "min_samples": min_samples,
        "data_ratios": data_ratios,
        "num_folds": dataset_path,
        "seed": seed
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_utils.get_data_partition_args_file_path(output_dir), "w") as f:
        json.dump(data_partition_args, f)

    # added up partitions for all tags and columns
    partitions_train, partitions_dev, partitions_test = [], [], []
    if num_folds == 0:
        k_partitions_train, k_partitions_dev, k_partitions_test = None, None, None
    else:
        k_partitions_train, k_partitions_dev, k_partitions_test = {}, {}, {}
        for k in range(num_folds):
            k_partitions_train[k], k_partitions_dev[k], k_partitions_test[k] = \
                [], [], []
        # TODO: Fix hacky hard coded index 0
        # partitions_train, partitions_dev, partitions_test = \
        #     k_partitions_train[0], k_partitions_dev[0], k_partitions_test[0]
    data_partitions_all = {
        "train": partitions_train,
        "dev": partitions_dev,
        "test": partitions_test,
    }
    final_labels = []

    # outputs
    valid_tags = []
    invalid_tags = [PartitionResult(invalid_tag_id, "", TagValidationResultType.tag_not_found)
                    for invalid_tag_id in invalid_tag_ids]
    # check tags
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

            # When num_folds == 0, it means it is not doing cross-validation.
            valid, data_partitions, distribution = _validate_and_partition_tag_data_column_data(
                tag_data, min_samples=min_samples, data_ratios=data_ratios, seed=seed)
            # get column data, validate and write data by tag
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
                result = PartitionResult(
                    valid_tag_id,
                    tag_name,
                    TagValidationResultType.not_enough_data,
                    distribution)
                invalid_tags.append(result)

            if num_folds > 0:
                valid, k_data_partitions, distribution = _validate_and_partition_tag_data_meta_data(
                    tag_data, num_folds=num_folds, data_ratios=data_ratios, seed=seed)

                for k, data_partitions in k_data_partitions.items():
                    tag_dir = os.path.join(
                        output_dir, f"fold_{k}", str(valid_tag_id))
                    os.makedirs(tag_dir, exist_ok=True)
                    dataset_utils.write_partitions(data_partitions, tag_dir)

                    # create result objects.
                    if valid:
                        # build overall partition
                        k_partitions_train[k].extend(data_partitions["train"])
                        k_partitions_dev[k].extend(data_partitions["dev"])
                        k_partitions_test[k].extend(data_partitions["test"])

                        # create valid result
                        result = PartitionResult(
                            valid_tag_id,
                            tag_name,
                            TagValidationResultType.valid,
                            distribution)
                        valid_tags.append(result)

                    else:
                        result = PartitionResult(
                            valid_tag_id,
                            tag_name,
                            TagValidationResultType.not_enough_data,
                            distribution)
                        invalid_tags.append(result)

                # write merged partitions
                for k in range(num_folds):
                    kth_data_partitions_all = {
                        "train": k_partitions_train[k],
                        "dev": k_partitions_dev[k],
                        "test": k_partitions_test[k],
                    }
                    kth_fold_output_dir = os.path.join(
                        output_dir, f"fold_{k}")
                    dataset_utils.write_partitions(
                        kth_data_partitions_all, output_dir=kth_fold_output_dir)
                    # dump labels
                    dataset_utils.write_labels(
                        output_dir=kth_fold_output_dir, labels=final_labels)

    # write merged partitions
    dataset_utils.write_partitions(
        data_partitions_all, output_dir=output_dir)
    dataset_utils.write_labels(
        output_dir=output_dir, labels=final_labels)

    distribution = get_partition_distribution(
        data_partitions=data_partitions_all)
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


def _get_tag_data_samples(tag_data):
    # TODO: provide multiple partition/selection functions
    tag_samples = []
    for col_id, col_data in tag_data.items():
        tag_samples.extend(col_data)
    return tag_samples


def _validate_and_partition_tag_data_column_data(tag_data, min_samples, data_ratios: List, seed: int) -> Tuple[bool, Dict, PartitionDistribution]:
    # Validation
    tag_samples = _get_tag_data_samples(tag_data)
    num_samples = len(tag_samples)
    valid = num_samples >= min_samples
    if valid:
        data_partitions = dataset_utils.split_dataset(
            all_samples=tag_samples, data_ratios=data_ratios, seed=seed)
    else:
        logger.info(
            f"num_samples({num_samples}) must NOT be smaller than min_samples {min_samples}")
        data_partitions = {
            "train": [],
            "dev": [],
            "test": [],
        }
    data_distributions = get_partition_distribution(
        data_partitions=data_partitions)
    return valid, data_partitions, data_distributions


def _validate_and_partition_tag_data_meta_data(tag_data, num_folds, data_ratios: List, seed: int) -> Tuple[bool, Dict, PartitionDistribution]:
    # Validation
    tag_samples = _get_tag_data_samples(tag_data)
    num_samples = len(tag_samples)
    valid = num_samples >= num_folds
    if valid:
        k_fold_data_partitions = dataset_utils.split_dataset_k_folds(
            all_samples=tag_samples, data_ratios=data_ratios, seed=seed, num_folds=num_folds)
    else:
        logger.info(
            f"num_samples({num_samples}) must NOT be smaller than num_folds {num_folds}")
        k_fold_data_partitions = {
            k: {
                "train": [],
                "dev": [],
                "test": [],
            } for k in range(num_folds)
        }
    data_distributions = get_partition_distribution(k_fold_data_partitions[0])
    return valid, k_fold_data_partitions, data_distributions


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
