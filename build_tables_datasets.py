import json
import os
import pandora.tools.test_utils as test_utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                logger.info(
                    f"conflict: {obj}, existing: {existing_obj}")
                existing_labels = existing_obj["label"]
                if label in existing_labels:
                    raise ValueError(f"duplicated data entry {obj}")
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


def get_label_2_tagid(
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
    labels = [f"\'{label}\'" for label in labels]
    query_template = f"select id, `key` from tag where `key` in ({', '.join(labels)})"
    cursor.execute(query_template)
    label_2_tagid_info_list = cursor.fetchall()

    label_2_tagid = {}
    for label_2_tagid_info in label_2_tagid_info_list:
        label_2_tagid[label_2_tagid_info[1]] = label_2_tagid_info[0]
    return label_2_tagid


def get_column_id_2_data(
        column_comments,
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
    col_id_col_name_col_comments = []
    column_comments = [f"\'{comment}\'" for comment in column_comments]
    for comment in column_comments:
        logger.info("==")
        logger.info(comment)
        # query_template = f"select id, name, comment from `column` where comment in ({', '.join(column_comments)})"
        query_template = f"select id, name, comment from `column` where comment=({comment})"
    # logger.info(query_template)
        cursor.execute(query_template)
        col_id_col_name_col_comment = cursor.fetchall()
        logger.info(col_id_col_name_col_comment)
        # assert len(col_id_col_name_col_comment) == 1
        # import time
        # time.sleep(1)
        col_id_col_name_col_comments.extend(col_id_col_name_col_comment)
    return col_id_col_name_col_comments


def make_request(url: str, post: bool = False, data=None, headers=None):
    import requests
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data


def tag_columns(
        tables_data,
        label_2_tagid,
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

    # TODO: Fix this hard coded datasource_id
    datasource_id = 108
    get_tables_query = f"select id, name from asset where type=1 and datasource_id={datasource_id}"
    cursor.execute(get_tables_query)
    table_info_list = cursor.fetchall()
    table_name_2_asset_id = {}
    for table_info in table_info_list:
        table_name_2_asset_id[table_info[1]] = table_info[0]
    connection.commit()

    for table_name, table_data in tables_data.items():

        cursor = connection.cursor()
        cursor.execute(f"use governance")
        records_to_insert = []

        # build column comment 2 column id
        table_asset_id = table_name_2_asset_id[table_name]
        column_info_query = f"select id, name, comment from `column` where asset_id={table_asset_id};"
        cursor.execute(column_info_query)

        column_info_list = cursor.fetchall()
        logger.info(
            f"fetching column_comment_2_column_info for table {table_name}, table_asset_id: {table_asset_id}")
        logger.info(f"column_info_list is {column_info_list}")

        column_comment_2_column_info = {}
        for column_info in column_info_list:
            column_comment_2_column_info[column_info[2]] = column_info
        logger.info(
            "column_comment_2_column_info:\n {column_comment_2_column_info}")

        for column_name, column_obj in table_data.items():
            column_comment = column_obj["meta_data"]["column_comment"]
            if column_comment not in column_comment_2_column_info:
                raise ValueError(
                    f"{column_comment} not in column_info_list {column_info_list}")
            column_info = column_comment_2_column_info[column_comment]
            column_id = column_info[0]

            all_labels = column_obj["label"]
            for label in all_labels:
                tag_name = label
                tag_id = label_2_tagid[tag_name]
                record = [None, 1666957310388597317,
                          1666957310388597317, None, 3, column_id, 1, tag_id]
                logger.info(f"tagging data query {record}")
                records_to_insert.append(record)
                # TODO: Fix this - only one label is allowed
                break

        # records_to_insert = [[None, 1666957310388597317,
        #             1666957310388597317, None, 3, column_id, 1, tag_id]]
        values_template = ", ".join(["%s"] * 8)
        mySql_insert_query = f"INSERT INTO subject2subject VALUES({values_template})"
        # records_to_insert = records_to_insert[:3]
        # logger.info(records_to_insert)
        cursor.executemany(mySql_insert_query, records_to_insert)
        connection.commit()


if __name__ == '__main__':
    tables_data, labels, duplicated = load_data(
        "test_data/data.json")
    # logger.info(json.dumps(tables, ensure_ascii=True, indent=4))

    # create tags
    # create_tags(labels=labels)

    # Create tables
    # logger.info("ingesting data to mysql")
    # host = "10.0.1.178"
    # port = "7733"
    # # and then check the response...
    # dataset_name = "test_real_data"
    # create_tables(
    #     host=host,
    #     port=port,
    #     database_name=dataset_name,
    #     tables_data=tables_data
    # )

    # get tag IDs and column IDs
    label_2_tagid = get_label_2_tagid(labels)
    # logger.info(label_2_tagid)
    # col_id_col_name_col_comments = get_column_id_2_data(column_comments)
    # logger.info(col_id_col_name_col_comments)

    tag_columns(
        tables_data=tables_data,
        label_2_tagid=label_2_tagid,
    )
