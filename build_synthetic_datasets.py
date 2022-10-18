#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cgi import print_directory
from distutils.command.config import config
import json
import os
import pathlib
from dataclasses import fields

from pandora.data.encoder import DataJSONEncoder

import pandora.dataset.dataset_utils as dataset_utils


def generate_data(
        dataset_name,
        database_name,
        num_data_entry,
        output_dir,
        generators,
        labels,
        column_name_2_label,
        column_name_2_comment,
        is_test_data=False):

    column_names_to_include = sorted(column_name_2_label.keys())
    print(f"column_names_to_include is {column_names_to_include}")
    data_file = os.path.join(output_dir, f"{dataset_name}.json")
    dataset = []
    with open(os.path.join(output_dir, f"{dataset_name}_data_table.json"), "w") as table_fr:
        with open(data_file, "w") as raw_data_fr:
            for _ in range(num_data_entry):
                data_entry = {}
                for generator in generators:
                    data = generator(is_test_data=is_test_data).generate()
                    for f in fields(data):
                        if f.name not in column_names_to_include:
                            continue
                        val = getattr(data, f.name)
                        if f.name in data_entry:
                            print(
                                f"key conflict: {f.name} already exists in output")
                            print(data_entry)
                            raise
                        data_entry[f.name] = val

                        # Write training data
                        out_line = {
                            "text": val,
                            "label": column_name_2_label[f.name]}
                        out_line["column_name"] = f.name
                        json.dump(out_line, raw_data_fr, ensure_ascii=False)
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
    dataset_utils.write_labels(output_dir=output_dir, labels=labels)

    # Check if mysql is available
    host = "10.0.1.178"
    port = "7733"
    response = os.system("ping -c 1 " + host)
    # and then check the response...
    if response == 0:
        print("ingesting data to mysql")
        create_table(
            table_name=dataset_name,
            database_name=database_name,
            host=host,
            port=port,
        )
        ingest_to_mysql(
            table_name=dataset_name,
            database_name=database_name,
            host=host,
            port=port,
            column_names=column_names_to_include,
            column_name_2_comment=column_name_2_comment,
            dataset=dataset)
    return data_file


def partition_data(output_dir, data_file, data_ratios, seed):
    all_samples = []
    with open(data_file, 'r') as fr:
        for _, line in enumerate(fr):
            data_entry = dataset_utils.DataEntry(**json.loads(line))
            all_samples.append(data_entry)
    data_partitions = dataset_utils.split_dataset(
        all_samples=all_samples, data_ratios=data_ratios, seed=seed)
    return data_partitions


# alias vm_178_mysql_l="mysql -h 10.0.1.178 -u root -pSudodata-123 -P 7733"
# alias vm_178_mysql="mysql -h 10.0.1.178 -u root -pSudodata -P 16969"
# alias vm_140_mysql="mysql -h 10.0.1.140 -u root -pSudodata-123 -P 33039"
# alias vm_67_mysql="mysql -h 10.0.1.67 -u root -pSudodata-123 -P 33039"
# alias vm_me_mysql="mysql -h 10.0.1.48 -u root -pSudodata-123 -P 33039"
def create_table(
    table_name,
    database_name,
    host="10.0.1.178",
    port="7733",
    user="root",
    password="Sudodata-123",
):
    import mysql.connector
    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )
    cursor = connection.cursor()
    try:
        # create database
        create_db_sql = f'CREATE DATABASE {database_name}'
        cursor.execute(create_db_sql)
    except:
        print(f"database {database_name} already exists")

    cursor.execute(f"use {database_name}")
    try:
        # create table
        cursor.execute(f"drop table {table_name}")
    except:
        print(f"table {table_name} does not exists")
    connection.commit()


def ingest_to_mysql(
    table_name,
    column_names,
    column_name_2_comment,
    dataset,
    database_name,
    host="10.0.1.178",
    port="7733",
    user="root",
    password="Sudodata-123",
):
    import mysql.connector
    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )
    cursor = connection.cursor()
    try:
        # create table
        cursor.execute(f"use {database_name}")
        columns_query = ", \n".join(
            [f"{col} VARCHAR(50) COMMENT '{column_name_2_comment[col]}'" for col in column_names])
        sql = f'''CREATE TABLE {table_name} (
                {columns_query}
            )'''
        cursor.execute(sql)

        # insert data
        values_template = ", ".join(["%s"] * len(column_names))
        mySql_insert_query = f"INSERT INTO {table_name} VALUES ({values_template})"
        records_to_insert = [tuple([k_v[1] for k_v in data_entry])
                             for data_entry in dataset]
        cursor.executemany(mySql_insert_query, records_to_insert)
        connection.commit()
        print(cursor.rowcount, "Record inserted successfully into Laptop table")
    except mysql.connector.Error as error:
        print("Failed to insert record into MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def build_dataset(
    dataset_name="demo_dataset",
    num_data_entry_train=10,
    num_data_entry_test=10,
):
    # output_dir = os.path.join(
    #     pathlib.Path.home(), "workspace", "resource", "outputs", "bert-base-chinese", "synthetic_data", "datasets", "synthetic_data")
    database_name = 'pandora'
    output_dir = os.path.join(
        pathlib.Path.home(), "workspace", "resource", "datasets", dataset_name)
    print(f"dataset output_dir is {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    # Create train / eval data
    seed = 42

    # import pandora.dataset.configs as configs
    import pandora.dataset.configs_demo as configs
    import pandora.dataset.configs_demo_2 as configs

    data_file_train = generate_data(
        dataset_name=f"{dataset_name}_train",
        database_name=database_name,
        num_data_entry=num_data_entry_train, output_dir=output_dir,
        generators=configs.DATA_GENERATORS, labels=configs.CLASSIFICATION_LABELS,
        is_test_data=False,
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TRAIN,
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT)
    data_ratios_train = {"train": 0.8, "dev": 0.2, "test": 0.0}
    data_partitions_train_dev = partition_data(output_dir, data_file=data_file_train,
                                               data_ratios=data_ratios_train, seed=seed)

    # Create test data
    data_file_test = generate_data(
        dataset_name=f"{dataset_name}_test_1",
        database_name=database_name,
        num_data_entry=num_data_entry_test, output_dir=output_dir,
        generators=configs.DATA_GENERATORS, labels=configs.CLASSIFICATION_LABELS,
        is_test_data=True,
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TEST,
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT)
    data_ratios_test = {"train": 0.0, "dev": 0.0, "test": 1.0}
    data_partitions_test_1 = partition_data(output_dir, data_file=data_file_test,
                                            data_ratios=data_ratios_test, seed=seed)

    # Create 2nd test data
    data_file_test = generate_data(
        dataset_name=f"{dataset_name}_test_2",
        database_name=database_name,
        num_data_entry=num_data_entry_test, output_dir=output_dir,
        generators=configs.DATA_GENERATORS, labels=configs.CLASSIFICATION_LABELS,
        is_test_data=True,
        column_name_2_label=configs.CLASSIFICATION_COLUMN_2_LABEL_ID_TEST,
        column_name_2_comment=configs.CLASSIFICATION_COLUMN_2_COMMENT)
    data_ratios_test = {"train": 0.0, "dev": 0.0, "test": 1.0}
    data_partitions_test_2 = partition_data(output_dir, data_file=data_file_test,
                                            data_ratios=data_ratios_test, seed=seed)

    data_partitions = {
        "train": data_partitions_train_dev["train"],
        "dev": data_partitions_train_dev["dev"],
        "test": data_partitions_test_1["test"]
    }
    # dump output
    dataset_utils.write_partitions(
        data_partitions, output_dir)


if __name__ == '__main__':
    dataset_name_prefix = f"pandora_demo_meta_comment_2"
    num_data_entry_train = 100
    num_data_entry_test = 100
    dataset_name = f"{dataset_name_prefix}_{num_data_entry_train}_{num_data_entry_test}"

    build_dataset(
        dataset_name=dataset_name,
        num_data_entry_train=num_data_entry_train,
        num_data_entry_test=num_data_entry_test,
    )
