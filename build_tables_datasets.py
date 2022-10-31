import json
import os
import pandora.tools.test_utils as test_utils
import logging
from pandora.poseidon.client import get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

poseidon_client = get_client()


def load_data(file_path):
    tables_data = {}
    labels = set()
    unique_data = set()
    duplicated = {}
    column_name_mapping = {}
    with open(file_path) as f:
        for line in f.readlines():
            obj = json.loads(line.strip())
            table_name = obj["meta_data"]["table_name"]
            # use column_comment as column_name_cn
            column_comment = obj["meta_data"]["column_comment"]

            # add label
            label = obj["label"][0]
            labels.add(label)

            combined = f"{table_name}_{column_comment}_{label}"
            if combined in unique_data:
                duplicated[combined] = duplicated.get(combined, 1) + 1
                logger.warning(f"skipping duplicated data entry: {combined}")
                continue
            unique_data.add(combined)

            if table_name not in tables_data:
                tables_data[table_name] = {}
            table_data = tables_data[table_name]

            # TODO: fix this
            # index = len(table_data)
            # column_name = f"{table_name}_col_{index}"
            column_name = column_comment

            # Check duplicate column name
            if column_name in table_data:
                existing_obj = table_data[column_name]
                existing_labels = existing_obj["label"]
                if label in existing_labels:
                    raise ValueError(f"duplicated data entry {obj}")
                logger.info(
                    f"adding additional label {label} to existing ones: {existing_labels}")
                existing_labels.append(label)
            else:
                obj["meta_data"]["column_name"] = column_name
                table_data[column_name] = obj
    return tables_data, list(labels), duplicated


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

        logger.info(f"===== creating table {table_name}")
        index = 0
        for column_name, column_obj in table_data.items():

            # TODO: fix this, can't use chinese chars in column name
            # column_name = column_obj["meta_data"]["column_name"]
            # Update column name in object
            column_name = f"{table_name}_col_{index}"
            index += 1
            column_obj["meta_data"]["column_name"] = column_name
            column_comment = column_obj["meta_data"]["column_comment"]

            column_names.append(column_name)
            column_name_2_comment[column_name] = column_comment

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


def make_request(url: str, post: bool = False, data=None, headers=None):
    import requests
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data


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

    # list tables in the DB, should have 34
    list_tables_resp = poseidon_client.list_tables(
        datasource_id=datasource_id, db_name=db_name)
    assert list_tables_resp.status_code == 200, list_tables_resp.text
    list_tables_resp_obj = json.loads(list_tables_resp.text)
    assert len(list_tables_resp_obj["assets"]) == 34, len(
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
            column_obj = table_data[column_comment]
            column_path = f"{datasource_name}/{db_name}/{table_name}/{column_name}"

            col_labels = column_obj["label"]
            logger.info(
                f"===== tagging column_name {column_name}(column_comment: {column_comment}) with tags/labels {col_labels}")
            if len(col_labels) <= 0:
                raise ValueError(f"column {column_name} has no labels")

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


if __name__ == '__main__':
    create_tables = False
    tables_data, labels, duplicated = load_data(
        "test_data/data.json")
    # logger.info(json.dumps(tables, ensure_ascii=True, indent=4))

    # create tags
    # create_tags(labels=labels)
    list_tags_resp = poseidon_client.list_tags()
    assert list_tags_resp.status_code == 200, list_tags_resp.text
    tag_name_to_obj = {tag["key"]: tag for tag in json.loads(list_tags_resp.text)[
        "tags"]}

    # Create tables in mysql
    database_name = "test_real_data"
    if create_tables:
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
        collection_id = 2
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
        datasource_id = 140
        datasource_name = database_name

    dataset = tag_table_columns(
        tables_data=tables_data,
        tag_name_to_obj=tag_name_to_obj,
        datasource_id=datasource_id,
        datasource_name=datasource_name,
        add_tagging=False,
    )
    poseidon_client.start_training("test_real_data", dataset=dataset)

    # delete datasource
    # delete_resp = poseidon_client.delete_datasource(
    #     datasource_id=datasource_id)
    # assert delete_resp.status_code == 200, delete_resp.text
