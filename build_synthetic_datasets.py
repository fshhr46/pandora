#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cgi import print_directory
import json
import os
import pathlib
from dataclasses import fields

from pandora.data.encoder import DataJSONEncoder

import pandora.dataset.dataset_utils as dataset_utils
from pandora.packaging.feature import TrainingType
import pandora.tools.test_utils as test_utils

# import pandora.dataset.configs as configs
import pandora.dataset.configs_demo as configs
import pandora.dataset.configs_demo_2 as configs
import pandora.dataset.configs_demo_3 as configs

from pandora.poseidon.client import (
    get_client,
    create_tags,
    tag_table_columns,
    classify_and_rate_table_columns,
)


def generate_data(
        training_type,
        dataset_name,
        database_name,
        num_data_entry,
        output_dir,
        generators,
        labels,
        classification_class_paths,
        classification_class_path_2_rating,
        column_name_2_label,
        column_name_2_class_path,
        column_name_2_comment,
        is_test_data=False,
        ingest_data=False):

    column_names_to_include = sorted(column_name_2_label.keys())
    print(f"column_names_to_include is {column_names_to_include}")
    data_file = os.path.join(output_dir, f"{dataset_name}.json")
    dataset = []

    def _add_classification_rating_data(out_line):
        class_path = column_name_2_class_path.get(
            column_name, "")
        if not class_path:
            return

        assert class_path in classification_class_paths, f"class_path {class_path} not found"
        rating_name = classification_class_path_2_rating.get(
            class_path, "")
        if class_path:
            out_line["meta_data"]["class_path"] = class_path
        if rating_name:
            out_line["meta_data"]["rating_name"] = rating_name

    with open(os.path.join(output_dir, f"{dataset_name}_data_table.json"), "w") as table_fr:
        with open(data_file, "w") as raw_data_fr:
            for _ in range(num_data_entry):
                data_entry = {}
                for generator in generators:
                    data = generator(is_test_data=is_test_data).generate()
                    for f in fields(data):
                        # some field name starts with "_" to use numeric data.
                        # For example "2022_q1_sales" will be written as field "_2022_q1_sales"
                        column_name = f.name.strip("_")
                        if column_name not in column_names_to_include:
                            continue
                        val = getattr(data, f.name)
                        if column_name in data_entry:
                            print(
                                f"key conflict: column_name {column_name} already exists in output")
                            print(data_entry)
                            raise
                        data_entry[column_name] = val

                        # Handle non-meta_data use case. Write data entries
                        if training_type != TrainingType.meta_data:

                            column_comment = column_name_2_comment[column_name]
                            column_description = column_comment
                            # Write training data
                            out_line = {
                                "text": val,
                                "label": column_name_2_label[column_name],
                                "meta_data": {
                                    "column_name": column_name,
                                    "column_comment": column_comment,
                                    "column_description": column_description,
                                }
                            }
                            _add_classification_rating_data(out_line)
                            json.dump(out_line, raw_data_fr,
                                      ensure_ascii=False)
                            raw_data_fr.write("\n")

                # Sort data_entry so it aligns with column name's order
                mysql_data_row = sorted(
                    data_entry.items(), key=lambda k_v: k_v[0])
                # write table data
                json.dump(data_entry, table_fr, ensure_ascii=False, sort_keys=True,
                          cls=DataJSONEncoder)
                assert len(mysql_data_row) == len(column_names_to_include), \
                    f"{len(mysql_data_row)} != {len(column_names_to_include)}"
                dataset.append(mysql_data_row)
                table_fr.write("\n")

    # Handle meta_data case
    if training_type == TrainingType.meta_data:
        with open(data_file, "w") as raw_data_fr:
            for col_tags in column_name_2_label.items():
                column_name = col_tags[0]
                column_comment = column_name_2_comment[column_name]
                column_description = column_comment
                # Write training data
                out_line = {
                    "text": "",
                    "label": col_tags[1],
                    "meta_data": {
                        "column_name": column_name,
                        "column_comment": column_comment,
                        "column_description": column_description,
                    }
                }
                _add_classification_rating_data(out_line)
                json.dump(out_line, raw_data_fr,
                          ensure_ascii=False)
                raw_data_fr.write("\n")

    dataset_utils.write_labels(output_dir=output_dir, labels=labels)

    # Check if mysql is available
    host = "10.0.1.178"
    port = "7733"
    # and then check the response...
    if ingest_data and os.system("ping -c 1 " + host) == 0:
        print("ingesting data to mysql")
        test_utils.create_database(
            database_name=database_name,
            host=host,
            port=port,
        )

        test_utils.cleanup_table(
            table_name=dataset_name,
            database_name=database_name,
            host=host,
            port=port,
        )

        test_utils.ingest_to_mysql(
            table_name=dataset_name,
            database_name=database_name,
            host=host,
            port=port,
            column_names=column_names_to_include,
            column_name_2_comment=column_name_2_comment,
            dataset=dataset,
            data_type="VARCHAR(100)")
    return data_file


def partition_data(training_type, output_dir, data_file, data_ratios, seed):
    all_samples = []
    with open(data_file, 'r') as fr:
        for _, line in enumerate(fr):
            data_entry = dataset_utils.DataEntry(**json.loads(line))
            all_samples.append(data_entry)

    # TODO: get a better strategy for meta_data training
    if training_type == TrainingType.meta_data:
        data_partitions = {
            "train": all_samples,
            "dev": all_samples,
            "test": all_samples
        }
        return data_partitions

    data_partitions = dataset_utils.split_dataset(
        all_samples=all_samples, data_ratios=data_ratios, seed=seed)
    return data_partitions


def load_one_table(file_path):
    table_data = {}
    with open(file_path) as f:
        for line in f.readlines():
            column_obj = json.loads(line.strip())
            column_name = column_obj["meta_data"]["column_name"]
            column_obj["label"] = [
                f"{label}_pdr_demo" for label in column_obj["label"]]
            table_data[column_name] = column_obj
    return table_data


def build_dataset(
    training_type,
    database_name='pandora',
    dataset_name="demo_dataset",
    num_data_entry_train=10,
    num_data_entry_test=10,
    ingest_data=False,
):
    # output_dir = os.path.join(
    #     pathlib.Path.home(), "workspace", "resource", "outputs", "bert-base-chinese", "synthetic_data", "datasets", "synthetic_data")

    tables_data = {}
    output_dir = os.path.join(
        pathlib.Path.home(), "workspace", "resource", "datasets", dataset_name)
    print(f"dataset output_dir is {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    # Create train / eval data
    seed = 42

    table_name_train = f"{dataset_name}_train"
    data_file_train = generate_data(
        training_type,
        dataset_name=table_name_train,
        database_name=database_name,
        num_data_entry=num_data_entry_train,
        output_dir=output_dir,
        generators=configs.DATA_GENERATORS,
        labels=configs.CLASSIFICATION_LABELS,
        classification_class_paths=configs.CLASSIFICATION_CLASS_PATHS,
        classification_class_path_2_rating=configs.CLASSIFICATION_CLASS_PATH_2_RATING,
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN,
        column_name_2_class_path=configs.CLASSIFICATION_RATING_COLUMN_2_CLASSPATH__TRAIN,
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT,
        is_test_data=False,
        ingest_data=ingest_data)
    tables_data[table_name_train] = load_one_table(data_file_train)
    data_ratios_train = {"train": 0.8, "dev": 0.2, "test": 0.0}
    data_partitions_train_dev = partition_data(training_type,
                                               output_dir, data_file=data_file_train,
                                               data_ratios=data_ratios_train, seed=seed)

    # Create test data
    table_name_test_1 = f"{dataset_name}_test_1"
    data_file_test_1 = generate_data(
        training_type,
        dataset_name=table_name_test_1,
        database_name=database_name,
        num_data_entry=num_data_entry_test,
        output_dir=output_dir,
        generators=configs.DATA_GENERATORS,
        labels=configs.CLASSIFICATION_LABELS,
        classification_class_paths=[],
        classification_class_path_2_rating={},
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TEST,
        column_name_2_class_path={},
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT,
        is_test_data=False,
        ingest_data=ingest_data)
    tables_data[table_name_test_1] = load_one_table(data_file_test_1)
    data_ratios_test = {"train": 0.0, "dev": 0.0, "test": 1.0}
    data_partitions_test_1 = partition_data(training_type,
                                            output_dir, data_file=data_file_test_1,
                                            data_ratios=data_ratios_test, seed=seed)

    # Create 2nd test data
    table_name_test_2 = f"{dataset_name}_test_2"
    data_file_test_2 = generate_data(
        training_type,
        dataset_name=table_name_test_2,
        database_name=database_name,
        num_data_entry=num_data_entry_test,
        output_dir=output_dir,
        generators=configs.DATA_GENERATORS,
        labels=configs.CLASSIFICATION_LABELS,
        classification_class_paths=[],
        classification_class_path_2_rating={},
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TEST,
        column_name_2_class_path={},
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT,
        is_test_data=False,
        ingest_data=ingest_data)
    tables_data[table_name_test_2] = load_one_table(data_file_test_2)
    data_ratios_test = {"train": 0.0, "dev": 0.0, "test": 1.0}
    data_partitions_test_2 = partition_data(training_type,
                                            output_dir, data_file=data_file_test_2,
                                            data_ratios=data_ratios_test, seed=seed)

    data_partitions = {
        "train": data_partitions_train_dev["train"],
        "dev": data_partitions_train_dev["dev"],
        "test": data_partitions_test_1["test"]
    }
    # dump output
    dataset_utils.write_partitions(
        data_partitions, output_dir)
    return tables_data


if __name__ == '__main__':
    dataset_name_prefix = f"pandora_demo_1127"
    num_data_entry_train = 100
    num_data_entry_test = 10
    dataset_name = f"{dataset_name_prefix}_{num_data_entry_train}_{num_data_entry_test}"
    database_name = dataset_name_prefix

    # Controls
    run_create_tables = False
    add_tagging = True
    ingest_data = False
    host = "10.0.1.8"

    training_type = TrainingType.mixed_data
    metadata_types = []
    metadata_types = ["COLUMN_NAME"]

    poseidon_client = get_client(host=host)
    tables_data = build_dataset(
        training_type=training_type,
        database_name=database_name,
        dataset_name=dataset_name,
        num_data_entry_train=num_data_entry_train,
        num_data_entry_test=num_data_entry_test,
        ingest_data=ingest_data,
    )

    labels = [f"{label}_pdr_demo" for label in configs.CLASSIFICATION_LABELS]

    # create tags
    list_tags_resp = poseidon_client.list_tags()
    assert list_tags_resp.status_code == 200, list_tags_resp.text
    existing_tags = {tag["key"]: tag for tag in json.loads(list_tags_resp.text)[
        "tags"]}

    create_tags(
        host=host,
        labels=labels, existing_tags=existing_tags)

    # Get updated tags
    list_tags_resp = poseidon_client.list_tags()
    assert list_tags_resp.status_code == 200, list_tags_resp.text
    tag_name_to_obj = {tag["key"]: tag for tag in json.loads(list_tags_resp.text)[
        "tags"]}

    if run_create_tables:

        # create datasource
        if host == "10.0.1.8":
            collection_id = 2518
        elif host == "10.0.1.48":
            collection_id = 4
        else:
            raise
        create_resp = poseidon_client.create_datasource(
            resource_name=database_name,
            collection_id=collection_id)
        assert create_resp.status_code == 200, create_resp.text

        # get datasource ID
        response_create_obj = json.loads(create_resp.text)
        datasource_id = response_create_obj["id"]
        datasource_name = response_create_obj["name"]

    else:
        # create datasource
        if host == "10.0.1.8":
            datasource_id = 528996
        elif host == "10.0.1.48":
            datasource_id = 192
        else:
            raise
        datasource_name = database_name

    # metasync
    sync_resp = poseidon_client.do_meta_sync(
        datasource_id=datasource_id, path=database_name)
    assert sync_resp.status_code == 200, sync_resp.text

    # Create taggings
    tag_dataset = tag_table_columns(
        poseidon_client=poseidon_client,
        tables_data=tables_data,
        tag_name_to_obj=tag_name_to_obj,
        datasource_id=datasource_id,
        datasource_name=datasource_name,
        add_tagging=add_tagging,
        tables_names_to_tag=[f"{dataset_name}_train"]
    )
    traning_data_type = training_type.upper()  # "COLUMN_DATA"
    description = f"{traning_data_type} | {metadata_types} | bert-base-chinese"
    train_resp = poseidon_client.start_training(
        f"pandora_demo_{training_type}_1127",
        dataset=tag_dataset,
        model_type="TAG",
        traning_data_type=traning_data_type,
        metadata_types=metadata_types,
        description=description,
    )
    assert train_resp.status_code == 200, train_resp.text

    # 参考 JR/T 0197-2020 金融数据安全 数据安全分级指南
    template_id = 3461

    # {"ratingId": "2694", "classificationId": "9696", "subjectType": "COLUMN", "subjectId": "784843", "templateId": "3461"}
    list_classifications_resp = poseidon_client.list_classifications(
        template_id=template_id)
    assert list_classifications_resp.status_code == 200, list_classifications_resp.text
    class_path_to_obj = {class_obj["namePath"]: class_obj for class_obj in json.loads(list_classifications_resp.text)[
        "classifications"]}

    list_ratings_resp = poseidon_client.list_ratings(template_id=template_id)
    assert list_ratings_resp.status_code == 200, list_ratings_resp.text
    rating_name_to_obj = {rating_obj["name"]: rating_obj for rating_obj in json.loads(list_ratings_resp.text)[
        "ratings"]}

    class_rate_dataset = classify_and_rate_table_columns(
        poseidon_client=poseidon_client,
        template_id=template_id,
        tables_data=tables_data,
        class_path_to_obj=class_path_to_obj,
        rating_name_to_obj=rating_name_to_obj,
        datasource_id=datasource_id,
        datasource_name=datasource_name,
        add_tagging=add_tagging,
        tables_names_to_tag=[f"{dataset_name}_train"]
    )

    traning_data_type = training_type.upper()  # "COLUMN_DATA"
    description = f"{traning_data_type} | {metadata_types} | bert-base-chinese"
    train_resp = poseidon_client.start_training(
        f"pandora_demo_{training_type}_1127",
        dataset=class_rate_dataset,
        model_type="CLASSIFICATION",
        traning_data_type=traning_data_type,
        metadata_types=metadata_types,
        description=description,
    )
    assert train_resp.status_code == 200, train_resp.text
