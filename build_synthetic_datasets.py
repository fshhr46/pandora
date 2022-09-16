#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pathlib
from dataclasses import fields

from pandora.data.encoder import DataJSONEncoder
from pandora.dataset.configs import (
    DATA_GENERATORS,
    DATA_CLASSES,
    CLASSIFICATION_COLUMN_2_LABEL_ID,
    CLASSIFICATION_LABELS,
)

import pandora.dataset.dataset_utils as dataset_utils


def generate_column_names(output_dir):
    column_names = []
    for data_class in DATA_CLASSES:
        column_names.extend(data_class.__annotations__.keys())
    with open(os.path.join(output_dir, "column_names.json"), "w") as fr:
        json.dump(column_names, fr, ensure_ascii=False)
    print(column_names)
    return column_names


def generate_data(num_data_entry, output_dir, generators, labels):
    generate_column_names(output_dir=output_dir)
    data_file = os.path.join(output_dir, "synthetic_raw.json")
    with open(os.path.join(output_dir, "data_table.json"), "w") as table_fr:
        with open(data_file, "w") as raw_data_fr:
            for _ in range(num_data_entry):
                data_entry = {}
                for generator in generators:
                    data = generator().generate()
                    for f in fields(data):
                        val = getattr(data, f.name)
                        if f.name in data_entry:
                            raise "key conflict: {f.name} already exists in output"

                        if f.name not in CLASSIFICATION_COLUMN_2_LABEL_ID:
                            raise "{f.name} is not labeled. Labels:\n {CLASSIFICATION_COLUMN_2_LABEL_ID}"
                        data_entry[f.name] = val

                        # Write training data
                        out_line = {
                            "text": val, "label": CLASSIFICATION_COLUMN_2_LABEL_ID[f.name]}
                        json.dump(out_line, raw_data_fr, ensure_ascii=False)
                        raw_data_fr.write("\n")
                # write table data
                json.dump(data_entry, table_fr, ensure_ascii=False, sort_keys=True,
                          cls=DataJSONEncoder)
                table_fr.write("\n")
    dataset_utils.write_labels(output_dir=output_dir, labels=labels)
    return data_file


def partition_data(output_dir, data_file, data_ratios, seed):
    all_samples = []
    with open(data_file, 'r') as fr:
        for _, line in enumerate(fr):
            data_entry = dataset_utils.DataEntry(**json.loads(line))
            all_samples.append(data_entry)
    data_partitions = dataset_utils.split_dataset(
        all_samples=all_samples, data_ratios=data_ratios, seed=seed)
    dataset_utils.write_partitions(
        data_partitions, output_dir)


if __name__ == '__main__':
    # output_dir = os.path.join(
    #     pathlib.Path.home(), "workspace", "resource", "outputs", "bert-base-chinese", "synthetic_data", "datasets", "synthetic_data")
    output_dir = os.path.join(
        pathlib.Path.home(), "workspace", "resource", "datasets", "synthetic_data")

    os.makedirs(output_dir, exist_ok=True)
    num_data_entry = 100
    data_file = generate_data(num_data_entry=num_data_entry, output_dir=output_dir,
                              generators=DATA_GENERATORS, labels=CLASSIFICATION_LABELS)
    data_ratios = {"train": 0.6, "dev": 0.2, "test": 0.2}
    seed = 42
    partition_data(output_dir, data_file=data_file,
                   data_ratios=data_ratios, seed=seed)
