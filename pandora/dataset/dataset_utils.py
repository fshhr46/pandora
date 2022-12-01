import random
import json
from dataclasses import dataclass
import os
from typing import Dict, List
import pandas as pd
import numpy as np


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


def split_dataset(all_samples, data_ratios, seed) -> Dict:
    error_msg = validate_ratios(data_ratios)
    if error_msg:
        raise ValueError(error_msg)
    # TODO: optimize this function - no need to load all then write.
    # write as it reads instead.
    random.Random(seed).shuffle(all_samples)
    train_percentage, dev_percentage, test_percentage = \
        data_ratios["train"], data_ratios["dev"], data_ratios["test"]
    num_samples = len(all_samples)
    num_train_samples = int(num_samples * train_percentage)
    num_dev_samples = int(num_samples * dev_percentage)

    # splitting
    train_dev_data = all_samples[:num_train_samples + num_dev_samples]
    data_partitions = {
        "train": train_dev_data[:num_train_samples],
        "dev": train_dev_data[num_train_samples:],
        "test": all_samples[num_train_samples + num_dev_samples:]
    }
    return data_partitions


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
