from cProfile import label
import os
from pathlib import Path
import json
from enum import Enum
from subprocess import getoutput
import random

supported_partitions = ["train", "dev", "test"]


class Dataset(str, Enum):
    column_data = "column_data"
    short_sentence = "short_sentence"
    long_sentence = "long_sentence"


def get_data_dir(resource_dir):
    return os.path.join(resource_dir, "data")


def get_output_data_folder(resource_dir):
    return os.path.join(resource_dir, "datasets", "sentence")


def get_data_folders(data_dir):
    folders = []
    for dataset in Dataset:
        folders.append(os.path.join(data_dir, dataset))
    return folders


def convert_column_data(data_partition, resource_dir, output_folder):
    assert data_partition in supported_partitions
    labels = ['address', 'book', 'company', 'game', 'government',
              'movie', 'name', 'organization', 'position', 'scene']

    dataset_dir = os.path.join(get_data_dir(resource_dir), "Cluener2020")
    truth_file = os.path.join(dataset_dir, f"{data_partition}.json")
    columns_data = {}

    with open(truth_file, 'r') as fr:
        for id, line in enumerate(fr):
            obj = json.loads(line)
            word_labels = obj["label"]
            for word_label, postions in word_labels.items():
                assert word_label in labels, f"word_label {word_label} is not in labels.\n{labels}"
                if word_label not in columns_data:
                    columns_data[word_label] = []
                columns_data[word_label].extend(list(postions.keys()))

    output_dir = os.path.join(
        get_output_data_folder(resource_dir), output_folder)
    output_file_all = os.path.join(output_dir, f"{data_partition}_raw.json")
    output_dir_partition = os.path.join(
        output_dir, data_partition)
    os.makedirs(output_dir_partition, exist_ok=True)

    with open(output_file_all, 'w') as fr_out:
        for label_name, column_data in columns_data.items():
            output_file = os.path.join(
                output_dir_partition, f"{data_partition}.{label_name}.json")
            with open(output_file, 'w') as f:
                json.dump(column_data, f, ensure_ascii=False)
            for word in column_data:
                out_line = {"text": word, "label": [label_name]}
                json.dump(out_line, fr_out, ensure_ascii=False)
                fr_out.write("\n")
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f, ensure_ascii=False)


def convert_long_sentence(data_partition, resource_dir, output_folder):
    assert data_partition in supported_partitions

    dataset_dir = os.path.join(get_data_dir(resource_dir), "iflytek_public")
    labels_file = os.path.join(dataset_dir, f"labels.json")
    labels = []
    with open(labels_file, 'r') as fr:
        for line in fr:
            labels.append(json.loads(line)["label_des"])

    truth_file = os.path.join(dataset_dir, f"{data_partition}.json")

    columns_data = {}
    output_dir = os.path.join(
        get_output_data_folder(resource_dir), output_folder)
    output_file_all = os.path.join(output_dir, f"{data_partition}_raw.json")
    output_dir_partition = os.path.join(
        output_dir, data_partition)
    os.makedirs(output_dir_partition, exist_ok=True)
    with open(truth_file, 'r') as fr:
        with open(output_file_all, 'w') as fr_out:
            for id, line in enumerate(fr):
                obj = json.loads(line)
                label = obj["label_des"]
                assert label in labels, f"label {label} is not in labels.\n{labels}"
                if label not in columns_data:
                    columns_data[label] = []
                out_line = {"text": obj["sentence"], "label": [label]}
                json.dump(out_line, fr_out, ensure_ascii=False)
                fr_out.write("\n")
                columns_data[label].append(obj["sentence"])

    for column_name, column_data in columns_data.items():
        output_file = os.path.join(
            output_dir_partition, f"{data_partition}.{column_name}.json")
        with open(output_file, 'w') as f:
            json.dump(column_data, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f, ensure_ascii=False)


def convert_short_sentence(data_partition, resource_dir, output_folder):
    assert data_partition in supported_partitions

    dataset_dir = os.path.join(get_data_dir(resource_dir), "tnews_public")
    labels_file = os.path.join(dataset_dir, f"labels.json")
    labels = []
    with open(labels_file, 'r') as fr:
        for line in fr:
            labels.append(json.loads(line)["label_desc"])

    truth_file = os.path.join(dataset_dir, f"{data_partition}.json")

    columns_data = {}
    output_dir = os.path.join(
        get_output_data_folder(resource_dir), output_folder)
    output_file_all = os.path.join(output_dir, f"{data_partition}_raw.json")
    output_dir_partition = os.path.join(
        output_dir, data_partition)
    os.makedirs(output_dir_partition, exist_ok=True)
    with open(truth_file, 'r') as fr:
        with open(output_file_all, 'w') as fr_out:
            for id, line in enumerate(fr):
                obj = json.loads(line)
                label = obj["label_desc"]
                assert label in labels, f"label {label} is not in labels.\n{labels}"
                if label not in columns_data:
                    columns_data[label] = []
                out_line = {"text": obj["sentence"], "label": [label]}
                json.dump(out_line, fr_out, ensure_ascii=False)
                fr_out.write("\n")
                columns_data[label].append(obj["sentence"])
    for column_name, column_data in columns_data.items():
        output_file = os.path.join(
            output_dir_partition, f"{data_partition}.{column_name}.json")
        with open(output_file, 'w') as f:
            json.dump(column_data, f, ensure_ascii=False)
    with open(os.path.join(output_dir, "labels.json"), 'w') as f:
        json.dump(labels, f, ensure_ascii=False)


def build_datasets(resource_dir):
    partitions = ["train", "dev"]
    for partition in partitions:
        convert_column_data(
            partition, resource_dir=resource_dir, output_folder=Dataset.column_data)
        convert_long_sentence(
            partition, resource_dir=resource_dir, output_folder=Dataset.long_sentence)
        convert_short_sentence(
            partition, resource_dir=resource_dir, output_folder=Dataset.short_sentence)


def split_test_set_from_train(resource_dir, data_ratios, seed=42):
    assert sum(data_ratios) == 1
    for dataset in Dataset:
        all_samples = []
        for partition in ["train", "dev"]:
            input_file = os.path.join(
                get_output_data_folder(resource_dir), dataset, f"{partition}_raw.json")
            with open(input_file, 'r') as fr:
                for _, line in enumerate(fr):
                    obj = json.loads(line)
                    all_samples.append(obj)
        random.Random(seed).shuffle(all_samples)

        print(f"\n\n====== splitting dataset: {dataset}\n")
        train_percentage, dev_percentage, test_percentage = data_ratios
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

        print(f"total samples: {len(all_samples)}")

        for partition, data in data_partitions.items():
            print(f"\npartition: {partition}, num_samples: {len(data)}")
            print(
                data_partitions[partition][0])
            output_path = os.path.join(
                get_output_data_folder(resource_dir), dataset, f"{partition}.json")
            print(f"writing to {output_path}")
            with open(output_path, "w") as fr:
                for obj in data:
                    json.dump(obj, fr, ensure_ascii=False)
                    fr.write("\n")


if __name__ == "__main__":
    resource_dir = os.path.join(Path.home(), "workspace", "resource")
    build_datasets(resource_dir)
    split_test_set_from_train(resource_dir, data_ratios=[0.6, 0.2, 0.2])
