import json
import os
import math
from pathlib import Path
import pandas as pd
import logging

from pandora.tools.test_utils import (
    create_tables,
    get_test_data_dir,
)

from pandora.poseidon.client import (
    get_client,
    create_tags,
    tag_table_columns,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

poseidon_client = get_client()


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
        if not label or type(label) == float and math.isnan(label):
            logger.info(
                f"invalid column label {label}. column_name: {column_name}")
            continue

        code, name = label.split(":")
        label = f"{code.strip()}:{name.strip()}"
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

    data_json_path = os.path.join(
        get_test_data_dir(), "data_benchmark_cn_mobile_data.json")
    with open(data_json_path, "w") as f_out:
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
                    json.dump(column_obj, f_out, ensure_ascii=False)
                    f_out.write("\n")
                batch_num += 1
    label_list = sorted(list(all_labels))
    labels_file = os.path.join(
        get_test_data_dir(), "data_benchmark_cn_mobile_labels.json"
    )
    json.dump(label_list, open(labels_file, "w"), indent=4, ensure_ascii=False)
    return tables_data_batch, label_list


if __name__ == '__main__':
    run_create_tables = False
    add_tagging = False
    raw_data_file = os.path.join(
        get_test_data_dir(), "data_benchmark_cn_mobile.xlsx"
    )
    tables_data, labels = load_data(raw_data_file)

    # create tags
    list_tags_resp = poseidon_client.list_tags()
    assert list_tags_resp.status_code == 200, list_tags_resp.text
    existing_tags = {tag["key"]: tag for tag in json.loads(list_tags_resp.text)[
        "tags"]}
    create_tags(labels=labels, existing_tags=existing_tags)

    # Get updated tags
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
        # raise
        datasource_id = 191
        datasource_name = database_name

    dataset = tag_table_columns(
        poseidon_client=poseidon_client,
        tables_data=tables_data,
        tag_name_to_obj=tag_name_to_obj,
        datasource_id=datasource_id,
        datasource_name=datasource_name,
        add_tagging=add_tagging,
    )

    traning_data_type = "META_DATA"
    metadata_types = ["COLUMN_NAME", "COLUMN_COMMENT"]
    description = f"{traning_data_type} | {metadata_types}"
    poseidon_client.start_training(
        "cn_mobile_meta_mixed",
        dataset=dataset,
        traning_data_type=traning_data_type,
        metadata_types=metadata_types,
        description=description,
    )
    # delete datasource
    # delete_resp = poseidon_client.delete_datasource(
    #     datasource_id=datasource_id)
    # assert delete_resp.status_code == 200, delete_resp.text
