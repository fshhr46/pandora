import random
import json
from dataclasses import dataclass
import os
from typing import Dict, List


class DataEntryEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


@dataclass
class DataEntry(object):
    def __init__(self,
                 text: str,
                 label: str,
                 column_name: str = None,
                 ) -> None:
        self.text = text
        self.label = label
        self.column_name = column_name
    text: str
    label: List
    column_name: str


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
