import json
import os
import math
from pathlib import Path
import pandas as pd
import pandora.tools.test_utils as test_utils
import logging
from pandora.poseidon.client import get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

poseidon_client = get_client()


def create_tags(
        labels,
        host="10.0.1.48",
        port="33039",
        user="root",
        password="Sudodata-123"):

    import mysql.connector
    connection = mysql.connector.connect(
        host=host,
        port=port,
        user=user,
        password=password,
    )
    cursor = connection.cursor()
    cursor.execute(f"use governance")
    records_to_insert = []
    for label in labels:
        record = [None, 1665944320972756876,
                  1665944320972756876, None, label, True, label, 1]
        records_to_insert.append(record)

    values_template = ", ".join(["%s"] * 8)
    mySql_insert_query = f"INSERT INTO tag VALUES({values_template})"
    cursor.executemany(mySql_insert_query, records_to_insert)
    connection.commit()


def normalize_column_name(column_name):
    column_name = column_name.lower()
    if column_name in ['key', 'group']:
        return f"`{column_name}`"
    column_name = column_name.replace(" ", "_")
    # column_name = column_name.replace("-", "_")
    # column_name = column_name.replace(":", "_")
    # column_name = column_name.replace(",", "_")
    # column_name = column_name.replace("(", "_")
    # column_name = column_name.replace(")", "_")
    # column_name = column_name.replace("{", "_")
    # column_name = column_name.replace("}", "_")
    # column_name = column_name.replace("/", "_")
    # column_name = column_name.replace("?", "_")
    # column_name = column_name.replace("&", "_")
    # column_name = column_name.replace("%", "p")
    return column_name


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_one_table(table_name, table_data_df):
    logger.info(f"loading data for table name {table_name}")
    # logger.info(f"table schema is {table_data_df.columns}")
    table_data_rows = table_data_df.to_dict(orient='records')
    table_data = {}
    table_labels = set()
    for row in table_data_rows:
        column_comment = row[table_data_df.columns[0]]
        column_name = row[table_data_df.columns[1]]
        data_type = row[table_data_df.columns[2]]
        is_main_key = row[table_data_df.columns[3]]
        classification = tag = row[table_data_df.columns[4]]
        rating = row[table_data_df.columns[5]]

        label = tag
        column_obj = {
            "text": "",
            "label": [label],
            "meta_data": {
                "table_name": table_name,
                "column_name": column_name,
                "column_comment": column_comment,
                "data_type": data_type,
                "is_main_key": is_main_key,
            }
        }
        if not label or type(label) == float and math.isnan(label):
            logger.info(f"invalid column object {column_obj}")
            continue
        table_labels.add(label)
        table_data[column_name] = column_obj
    return table_data, table_labels


def load_data(file_path):
    tables_data_df = pd.read_excel(
        file_path, sheet_name=None)

    tables_data = {}
    all_labels = set()
    for table_name, table_data_df in tables_data_df.items():
        table_data, table_labels = load_one_table(
            table_name, table_data_df=table_data_df)
        tables_data[table_name] = table_data
        all_labels.update(table_labels)

    # break large table into smaller ones
    tables_data_batch = {}
    for table_name, table_data in tables_data.items():
        column_batches = divide_chunks(list(table_data.items()), 100)
        batch_num = 0

        for column_batche in column_batches:

            table_name_with_batch = f"{table_name}_{batch_num}"
            table_data_with_batch = {}
            tables_data_batch[table_name_with_batch] = table_data_with_batch
            for column_name, column_obj in column_batche:
                normalized_column_name = normalize_column_name(column_name)
                column_obj["meta_data"]["column_name"] = normalized_column_name
                column_obj["meta_data"]["table_name"] = table_name_with_batch
                table_data_with_batch[normalized_column_name] = column_obj
            batch_num += 1
    return tables_data_batch, list(all_labels)


def tag_table_columns(
        tables_data,
        tag_name_to_obj,
        datasource_id,
        datasource_name,
        add_tagging=False):

    dataset_tags = {}
    dataset_tags_data = {}

    # list databases - should have only 1
    list_dbs_resp = poseidon_client.list_databases(datasource_id=datasource_id)
    assert list_dbs_resp.status_code == 200, list_dbs_resp.text
    list_dbs_resp_obj = json.loads(list_dbs_resp.text)
    assert len(list_dbs_resp_obj["assets"]) == 1
    db_id = list_dbs_resp_obj["assets"][0]["id"]
    db_name = list_dbs_resp_obj["assets"][0]["name"]

    # list tables in the DB, should have 78
    list_tables_resp = poseidon_client.list_tables(
        datasource_id=datasource_id, db_name=db_name)
    assert list_tables_resp.status_code == 200, list_tables_resp.text
    list_tables_resp_obj = json.loads(list_tables_resp.text)
    assert len(list_tables_resp_obj["assets"]) == 15, len(
        list_tables_resp_obj["assets"])

    for asset in list_tables_resp_obj["assets"]:
        table_asset_id = asset["id"]

        table_name = asset["name"]
        logger.info(f"tagging columns in table {table_name}")
        table_data = tables_data[table_name]

        # build column comment 2 column id
        list_columns_resp = poseidon_client.list_columns(
            asset_id=table_asset_id)
        assert list_columns_resp.status_code == 200, list_columns_resp.text
        list_columns_resp_obj = json.loads(list_columns_resp.text)

        table_taggings = []
        for column in list_columns_resp_obj["columns"]:
            column_comment = column["comment"]
            column_id = column["id"]
            column_name = column["name"]
            # column_obj = table_data[column_comment]

            if column_name not in table_data:
                raise ValueError(
                    f"column {column_name} not in table_data {table_data}")
            column_obj = table_data[column_name]
            column_path = f"{datasource_name}/{db_name}/{table_name}/{column_name}"

            col_labels = column_obj["label"]
            logger.info(
                f"===== tagging column_name {column_name}(column_comment: {column_comment}) with tags/labels {col_labels}")
            # TODO: only support single label for now.
            if len(col_labels) != 1:
                raise ValueError(
                    f"every column {column_name} should only have one label")

            for label in col_labels:

                tag_data = tag_name_to_obj[label]
                tag_id = tag_data["id"]

                # Create tags
                if tag_id not in dataset_tags:
                    dataset_tags[tag_id] = label
                    dataset_tags_data[tag_id] = {
                        "data": []
                    }
                dataset_tags_data[tag_id]["data"].append(
                    {"id": column_id, "name": column_name, "path": column_path}
                )

                tag = {"key": label, "value": "true",
                       "subjectId": column_id, "subjectType": "COLUMN"}
                table_taggings.append(tag)

        if add_tagging:
            poseidon_client.add_column_tags(tags=table_taggings)

    tags_list = []
    for tag_id, tag_key in dataset_tags.items():
        tags_list.append({"id": tag_id, "key": tag_key})
    dataset = {
        "tags": tags_list,
        "tagsData": dataset_tags_data
    }
    return dataset


def create_tables(
        host,
        port,
        database_name,
        tables_data):

    # create database
    test_utils.create_database(
        database_name=database_name,
        host=host,
        port=port,
    )

    for table_name, table_data in tables_data.items():
        # table_name = table["meta_data"]["table_name"]

        logger.info(f"creating table {table_name}")
        column_names = []
        column_name_2_comment = {}

        for column_name, column_obj in table_data.items():
            column_comment = column_obj["meta_data"]["column_comment"]

            column_names.append(column_name)
            column_name_2_comment[column_name] = column_comment

        logger.info(f"===== creating table {table_name}")
        logger.info(f"===== column_names\n {column_names}")
        test_utils.cleanup_table(
            table_name=table_name,
            database_name=database_name,
            host=host,
            port=port,
        )
        logger.warning(f"deleted table {table_name}")

        test_utils.ingest_to_mysql(
            table_name=table_name,
            database_name=database_name,
            host=host,
            port=port,
            column_names=column_names,
            column_name_2_comment=column_name_2_comment,
            dataset=[])


if __name__ == '__main__':
    run_create_tables = False
    tables_data, labels = load_data(
        "./test_data/data_benchmark_cn_mobile.xlsx")

    # create tags
    # create_tags(labels=labels)
    list_tags_resp = poseidon_client.list_tags()
    assert list_tags_resp.status_code == 200, list_tags_resp.text
    tag_name_to_obj = {tag["key"]: tag for tag in json.loads(list_tags_resp.text)[
        "tags"]}

    # Create tables in mysql
    database_name = "test_real_data_cn_mobile"
    if run_create_tables:
        logger.info("ingesting data to mysql")
        host = "10.0.1.178"
        port = "7733"
        create_tables(
            host=host,
            port=port,
            database_name=database_name,
            tables_data=tables_data
        )

        # create datasource
        collection_id = 4
        create_resp = poseidon_client.create_datasource(
            resource_name=database_name,
            collection_id=collection_id)
        assert create_resp.status_code == 200, create_resp.text

        # get datasource ID
        response_create_obj = json.loads(create_resp.text)
        datasource_id = response_create_obj["id"]
        datasource_name = response_create_obj["name"]

        # metasync
        sync_resp = poseidon_client.do_meta_sync(
            datasource_id=datasource_id, path=database_name)
        assert sync_resp.status_code == 200, sync_resp.text
    else:
        raise
        datasource_id = 174
        datasource_name = database_name

    dataset = tag_table_columns(
        tables_data=tables_data,
        tag_name_to_obj=tag_name_to_obj,
        datasource_id=datasource_id,
        datasource_name=datasource_name,
        add_tagging=True,
    )
    poseidon_client.start_training("test_real_data", dataset=dataset)

    # delete datasource
    # delete_resp = poseidon_client.delete_datasource(
    #     datasource_id=datasource_id)
    # assert delete_resp.status_code == 200, delete_resp.text
