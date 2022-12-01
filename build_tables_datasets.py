import json
import os
from pathlib import Path
import logging

import pandora

from pandora.packaging.losses import LossType
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


def is_valid_column_name(column_name):
    valid = True
    if len(column_name) > 62:
        valid = False
        logger.info(
            f"invalid column_name. column_name is too long: {column_name} ")
    return valid


def load_raw_translations():
    home = str(Path.home())
    comment_translations = os.path.join(
        get_test_data_dir(), "translations.json")
    translated_column_comments = json.load(open(comment_translations))

    return translated_column_comments


def load_data(
    file_path,
    ensure_single_label=True
):
    tables_data = {}
    all_labels = set()
    combined_data_en_count = {}
    combined_data_cn_count = {}

    tables_and_columns_en = {}
    duplicated_columns_en = {}

    tables_and_columns_cn = {}
    duplicated_columns_cn = {}

    translated_column_comments = load_raw_translations()
    data_json_path = os.path.join(
        get_test_data_dir(), "data_benchmark_gov_data.json")
    with open(data_json_path, "w") as f_out:
        with open(file_path) as f:
            for line in f.readlines():
                obj = json.loads(line.strip())
                table_name = obj["meta_data"]["table_name"]
                # use column_comment as column_name_cn
                column_comment = obj["meta_data"]["column_comment"]
                translated_col_comment = translated_column_comments[column_comment]

                # add label
                assert len(
                    obj["label"]) == 1, f"obj should have only one label {obj}"
                label = obj["label"][0]
                all_labels.add(label)

                combined_en = f"{table_name}_{translated_col_comment}_{label}"
                combined_data_en_count[combined_en] = combined_data_en_count.get(
                    combined_en, 0) + 1

                combined_cn = f"{table_name}_{column_comment}_{label}"
                combined_data_cn_count[combined_en] = combined_data_cn_count.get(
                    combined_cn, 0) + 1

                if combined_data_en_count[combined_en] > 1:
                    assert combined_data_en_count[combined_en] == combined_data_en_count[combined_en]
                    logger.warning(
                        f"skipping duplicated data entry: {combined_en}")
                    continue

                if table_name not in tables_and_columns_en:
                    tables_and_columns_en[table_name] = {}

                table_en = tables_and_columns_en[table_name]

                # TODO: fix this
                # index = len(table_data)
                # column_name = f"{table_name}_col_{index}"
                column_name = translated_col_comment

                # validate column name
                if not is_valid_column_name(column_name):
                    continue

                    # find duplicates by table and column name
                if column_name not in table_en:
                    table_en[column_name] = list()
                table_en[column_name].append(label)

                table_column_en = f'{table_name}_{column_name}'
                if len(table_en[column_name]) > 1:
                    duplicated_columns_en[table_column_en] = table_en[column_name]

                    if ensure_single_label:
                        # Column name needs to add an index to ensure single index
                        index = len(duplicated_columns_en[table_column_en])
                        column_name = f"{translated_col_comment}_{index}"

                # add objects to table_data
                if table_name not in tables_data:
                    tables_data[table_name] = {}
                table_data = tables_data[table_name]

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
                json.dump(obj, f_out, ensure_ascii=False)
                f_out.write("\n")

    # tables_data_file = os.path.join(
    #     get_test_data_dir(), "tables_data.json"
    # )
    # json.dump(tables_data, open(tables_data_file, "w"),
    #           indent=4, ensure_ascii=False)

    label_list = sorted(list(all_labels))
    label_json_path = os.path.join(
        get_test_data_dir(), "data_benchmark_gov_labels.json")
    json.dump(label_list, open(
        label_json_path, "w"), indent=4, ensure_ascii=False)

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

    # if duplicated_columns_en:
    #     duplicated_columns_en_file = os.path.join(
    #         get_test_data_dir(), "duplicated_columns_en.json"
    #     )
    #     json.dump(duplicated_columns_en, open(
    #         duplicated_columns_en_file, "w"), indent=4, ensure_ascii=False)
    #     raise ValueError(
    #         f"duplicated column translations:\n {json.dumps(duplicated_columns_en, indent=4, ensure_ascii=False)}")

    return tables_data_batch, label_list


def divide_chunks(l, n):

    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def make_request(url: str, post: bool = False, data=None, headers=None):
    import requests
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data


if __name__ == '__main__':
    run_create_tables = False
    add_tagging = False
    raw_data_file = os.path.join(
        get_test_data_dir(), "data_benchmark_gov.json"
    )
    tables_data, labels = load_data(raw_data_file)
    # logger.info(json.dumps(tables, ensure_ascii=True, indent=4))

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
    database_name = "test_real_data_translated"
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
        collection_id = 3
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
        datasource_id = 188
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
    metadata_types = ["COLUMN_COMMENT"]
    loss_type = LossType.x_ent
    description = f"{traning_data_type} | {metadata_types} | bert-base-chinese | {loss_type}"
    poseidon_client.start_training(
        "gov_meta_name",
        dataset=dataset,
        model_type="TAG",
        traning_data_type=traning_data_type,
        metadata_types=metadata_types,
        description=description,
    )
    # delete datasource
    # delete_resp = poseidon_client.delete_datasource(
    #     datasource_id=datasource_id)
    # assert delete_resp.status_code == 200, delete_resp.text
