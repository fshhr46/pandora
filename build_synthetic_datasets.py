#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pathlib
from dataclasses import fields

import mysql.connector

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


def generate_data(
        dataset_name,
        num_data_entry, output_dir, generators, labels,
        ingest_data=True):
    column_names = generate_column_names(output_dir=output_dir)
    data_file = os.path.join(output_dir, "synthetic_raw.json")
    dataset = []
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
                dataset.append(data_entry)
                table_fr.write("\n")
    dataset_utils.write_labels(output_dir=output_dir, labels=labels)

    if ingest_data:
        print("ingesting data to mysql")
        create_table(
            table_name=dataset_name)
        ingest_to_mysql(
            table_name=dataset_name,
            column_names=column_names,
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
    dataset_utils.write_partitions(
        data_partitions, output_dir)


# alias vm_178_mysql_l="mysql -h 10.0.1.178 -u root -pSudodata-123 -P 7733"
# alias vm_178_mysql="mysql -h 10.0.1.178 -u root -pSudodata -P 16969"
# alias vm_140_mysql="mysql -h 10.0.1.140 -u root -pSudodata-123 -P 33039"
# alias vm_67_mysql="mysql -h 10.0.1.67 -u root -pSudodata-123 -P 33039"
# alias vm_me_mysql="mysql -h 10.0.1.48 -u root -pSudodata-123 -P 33039"
def create_table(
    table_name,
    database_name='training_test_data',
    host="10.0.1.48",
    port="33039",
    user="root",
    password="Sudodata-123",
):
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
    dataset,
    database_name='training_test_data',
    host="10.0.1.48",
    port="33039",
    user="root",
    password="Sudodata-123",
):
    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )
    cursor = connection.cursor()
    try:
        cursor.execute(f"use {database_name}")
        columns_query = ", \n".join(
            [f"{col} VARCHAR(50)" for col in column_names])
        sql = f'''CREATE TABLE {table_name} (
                {columns_query}
            )'''
        cursor.execute(sql)

        # insert data
        values_template = ", ".join(["%s"] * len(column_names))
        mySql_insert_query = f"INSERT INTO {table_name} VALUES ({values_template})"

        records_to_insert = [tuple(list(data_entry.values()))
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


if __name__ == '__main__':
    # output_dir = os.path.join(
    #     pathlib.Path.home(), "workspace", "resource", "outputs", "bert-base-chinese", "synthetic_data", "datasets", "synthetic_data")
    dataset_name = "synthetic_data"
    output_dir = os.path.join(
        pathlib.Path.home(), "workspace", "resource", "datasets", dataset_name)

    os.makedirs(output_dir, exist_ok=True)
    num_data_entry = 100
    data_file = generate_data(
        dataset_name=dataset_name,
        num_data_entry=num_data_entry, output_dir=output_dir,
        generators=DATA_GENERATORS, labels=CLASSIFICATION_LABELS,
        ingest_data=True)
    data_ratios = {"train": 0.6, "dev": 0.2, "test": 0.2}
    seed = 42
    partition_data(output_dir, data_file=data_file,
                   data_ratios=data_ratios, seed=seed)
