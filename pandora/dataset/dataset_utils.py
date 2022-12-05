import random
import json
from dataclasses import dataclass
import os
from typing import Dict, List
import pandas as pd
import numpy as np

# K-Fold validation is expensive
MAX_NUM_FOLD = 3


class DataEntryEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


@dataclass
class DataEntry(object):
    def __init__(self,
                 text: str,
                 label: str,
                 meta_data: Dict = None,
                 ) -> None:
        self.text = text
        self.label = label
        self.meta_data = meta_data
    text: str
    label: List
    meta_data: Dict


def _get_num_samples(all_samples, data_ratios, seed):

    error_msg = validate_ratios(data_ratios)
    if error_msg:
        raise ValueError(error_msg)
    # TODO: optimize this function - no need to load all then write.
    # write as it reads instead.
    random.Random(seed).shuffle(all_samples)
    train_percentage, dev_percentage, _ = \
        data_ratios["train"], data_ratios["dev"], data_ratios["test"]
    num_samples = len(all_samples)
    num_train_samples = int(num_samples * train_percentage)
    num_dev_samples = int(num_samples * dev_percentage)
    num_test_samples = num_samples - num_train_samples - num_dev_samples
    return num_samples, \
        num_train_samples, \
        num_dev_samples, \
        num_test_samples


def split_dataset(all_samples, data_ratios, seed) -> Dict:
    num_samples, \
        num_train_samples, \
        num_dev_samples, \
        num_test_samples = _get_num_samples(all_samples, data_ratios, seed)

    # splitting
    train_dev_data = all_samples[:num_train_samples + num_dev_samples]
    data_partitions = {
        "train": train_dev_data[:num_train_samples],
        "dev": train_dev_data[num_train_samples:],
        "test": all_samples[num_train_samples + num_dev_samples:]
    }
    return data_partitions


def split_dataset_k_folds(all_samples, data_ratios, seed, num_folds):
    num_samples, \
        _, \
        _, \
        num_test_samples = _get_num_samples(all_samples, data_ratios, seed)
    # splitting
    train_dev_data = all_samples[:num_samples - num_test_samples]
    test_data = all_samples[-num_test_samples:]
    num_train_dev_samples = len(train_dev_data)

    # Validation
    assert num_folds <= num_samples, \
        f"num_folds({num_folds}) must be smaller than num_samples {num_samples}"
    assert num_folds <= MAX_NUM_FOLD and num_folds > 0, \
        f"num_folds({num_folds}) must be between 0 and {MAX_NUM_FOLD}"

    fold_size = num_train_dev_samples // num_folds
    num_samples_mod = num_train_dev_samples % num_folds
    start_index = 0
    fold_start_indexes = [start_index]
    for i in range(num_folds):
        start_index = start_index + fold_size + \
            (1 if i < num_samples_mod else 0)
        fold_start_indexes.append(start_index)

    # Nth fold to it's partitions
    k_folds_partitions = {}
    for i in range(num_folds):
        train_folds = []
        train_folds.extend(train_dev_data[:fold_start_indexes[i]])
        train_folds.extend(train_dev_data[fold_start_indexes[i + 1]:])
        test_fold = train_dev_data[fold_start_indexes[i]
            : fold_start_indexes[i + 1]]
        data_partitions = {
            "train": train_folds,
            "dev": test_fold,
            "test": test_data
        }
        k_folds_partitions[i] = data_partitions
    return k_folds_partitions


def calculate_label_distribution(samples):
    label_records = []
    for sample in samples:
        label = sample.get_label()
        label_records.append(label)
    df = pd.DataFrame.from_records(
        np.array(label_records, dtype=[('col_1', "U200")]))
    distribution = df.groupby("col_1")["col_1"].count() / len(df)
    return distribution.to_dict()


def write_partitions(data_partitions, output_dir):
    for partition, data in data_partitions.items():
        output_path = os.path.join(output_dir, f"{partition}.json")
        with open(output_path, "w") as fr:
            for obj in data:
                json.dump(obj, fr, ensure_ascii=False, cls=DataEntryEncoder)
                fr.write("\n")


def write_labels(output_dir, labels):
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f, ensure_ascii=False)


def validate_ratios(data_ratios) -> str:
    sum_ratios = sum(data_ratios.values())
    if round(sum_ratios, 8) != 1:
        return f"parition rations must add up to 1. sum_ratios: {sum_ratios}, ratios: {data_ratios}"
    else:
        return ""


def get_partitioned_data_folder(resource_dir):
    return os.path.join(resource_dir, "datasets")


def get_data_partition_args_file_path(output_dir):
    return os.path.join(output_dir, "data_partition_args.json")
