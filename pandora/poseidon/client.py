import requests
import json
import logging

logger = logging.getLogger(__name__)


class PoseidonClient(object):
    def __init__(
            self,
            host,
            port):
        self.host = host
        self.port = port

    def _get_url(self, suffix=None):
        url = f"http://{self.host}:{self.port}"
        if suffix:
            url = f"{url}/{suffix}"
        return url

    def _get_headers(self):
        return {
            "Authorization": "sudosudo",
            "Content-Type": "application/json;charset=UTF-8"
        }

    def create_datasource(self, resource_name, collection_id=1):
        data = {
            "name": resource_name,
            "type": "MYSQL",
            "collectionId": collection_id,
            "connectionInfo": {
                "mysql": {
                    "host": "10.0.1.178",
                    "port": "7733",
                    "user": "root",
                    "password": "Sudodata-123"}
            },
            "eventScrapeInfo": {
                "eventScrapeEnabled": True
            }
        }
        return requests.post(
            self._get_url("gapi/catalog/datasources"),
            data=json.dumps(data),
            headers=self._get_headers())

    def delete_datasource(self, datasource_id):
        return requests.delete(
            self._get_url(f"gapi/catalog/datasources/{datasource_id}"),
            headers=self._get_headers())

    def do_meta_sync(self, datasource_id, path=""):
        data = {
            "datasourceId": datasource_id,
            "path": path,
            "refreshNow": True
        }
        return requests.post(
            self._get_url(f"gapi/catalog/one_off_refresh/{datasource_id}"),
            data=json.dumps(data),
            headers=self._get_headers())

    def add_column_tags(self, tags):
        data = {"addTags": tags, "deleteTags": []}
        return requests.post(
            self._get_url(f"gapi/catalog/tags/bulk_update"),
            data=json.dumps(data),
            headers=self._get_headers())

    def delete_column_tags(self, tags):
        data = {"addTags": [], "deleteTags": tags}
        return requests.post(
            self._get_url(f"gapi/catalog/tags/bulk_update"),
            data=json.dumps(data),
            headers=self._get_headers())

    # {"adds": [{"ratingId": "2694", "classificationId": "9696", "subjectType": "COLUMN", "subjectId": "784843", "templateId": "3461"}, {
    #     "ratingId": "2694", "classificationId": "9683", "subjectType": "COLUMN", "subjectId": "784844", "templateId": "3461"}], "deletes": []}
    def add_column_classification_ratings(self, classification_ratings):
        data = {"adds": classification_ratings, "deletes": []}
        return requests.post(
            self._get_url(f"gapi/catalog/classification_ratings/bulk_update"),
            data=json.dumps(data),
            headers=self._get_headers())

    def delete_column_classification_ratings(self, classification_ratings):
        data = {"adds": [], "deletes": classification_ratings}
        return requests.post(
            self._get_url(f"gapi/catalog/classification_ratings/bulk_update"),
            data=json.dumps(data),
            headers=self._get_headers())

    def list_databases(self, datasource_id):
        return requests.get(
            self._get_url(
                f"gapi/catalog/assets?datasourceId={datasource_id}&type=MIRROR_DB"),
            headers=self._get_headers())

    def list_tables(self, datasource_id, db_name):
        return requests.get(
            self._get_url(
                f"gapi/catalog/assets?datasourceId={datasource_id}&type=MIRROR_TABLE&parentDb={db_name}&paginator.notPaging=true"),
            headers=self._get_headers())

    def list_columns(self, asset_id):
        # {"tags":[{"id":"175", "key":"科研管理系统", "value":"true", "description":"科研管理系统", "owners":[{"id":"1", "account":"admin", "username":"admin"}], "usedCount":"0", "createdTs":"2022-10-16T18:18:40.972756876Z", "updatedTs":"2022-10-16T18:18:40.972756876Z"}]}
        return requests.get(
            self._get_url(
                f"gapi/catalog/columns?assetId={asset_id}&&withTags=true&withClassificationRatings=true&paginator.notPaging=true"),
            headers=self._get_headers())

    # ====== Tagging related APIs
    def list_tags(self):
        return requests.get(
            self._get_url(
                f"gapi/catalog/tags?paginator.notPaging=true&orderByUsedCount=true"),
            headers=self._get_headers())

    # {"classifications": [{"id": "9335", "templateId": "3459", "name": "用户相关数据",
    #                       "namePath": "/用户相关数据", "description": "", "children": [], "parentId":"0"}]}
    def list_classifications(self, template_id):
        return requests.get(
            self._get_url(
                f"gapi/scan/template/classification?templateId={template_id}"),
            headers=self._get_headers())

    # ratings response {"ratings":[{"id":"2690","templateId":"3459","name":"一级数据","description":"","priority":1}]}
    def list_ratings(self, template_id):
        return requests.get(
            self._get_url(
                f"gapi/scan/template/rating?templateId={template_id}"),
            headers=self._get_headers())

    # ====== Training related APIs
    def start_training(
        self,
        model_name,
        dataset,
        model_type,
        traning_data_type="META_DATA",
        metadata_types=["COLUMN_COMMENT"],
        description="",
    ):
        data = {
            "baseModelId": "1",
            "name": model_name,
            "dataSet": dataset,
            "traningDataType": traning_data_type,
            "metadataTypes": metadata_types,
            "modelType": model_type,
            "description": description,
        }
        return requests.post(
            self._get_url(f"gapi/recognition/training/job/initial"),
            data=json.dumps(data),
            headers=self._get_headers())


def create_tags(
        labels,
        existing_tags,
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
        if label in existing_tags:
            continue
        record = [None, 1665944320972756876,
                  1665944320972756876, None, label, "true", label, 1]
        records_to_insert.append(record)

    if records_to_insert:
        values_template = ", ".join(["%s"] * 8)
        mySql_insert_query = f"INSERT INTO tag VALUES({values_template})"
        cursor.executemany(mySql_insert_query, records_to_insert)
        connection.commit()


def tag_table_columns(
        poseidon_client,
        tables_data,
        tag_name_to_obj,
        datasource_id,
        datasource_name,
        add_tagging=False,
        tables_names_to_tag=[]):

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
    assert len(list_tables_resp_obj["assets"]) == len(tables_data), len(
        list_tables_resp_obj["assets"])

    for asset in list_tables_resp_obj["assets"]:
        table_asset_id = asset["id"]

        table_name = asset["name"]
        if tables_names_to_tag and table_name not in tables_names_to_tag:
            logger.info(f"skip tagging table {table_name}")
            continue
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
                       "subjectId": column_id,
                       "description": label,
                       "subjectType": "COLUMN"}
                table_taggings.append(tag)

        if add_tagging:
            add_column_tags_resp = poseidon_client.add_column_tags(
                tags=table_taggings)
            assert add_column_tags_resp.status_code == 200, add_column_tags_resp.text

    tags_list = []
    for tag_id, tag_key in dataset_tags.items():
        tags_list.append({"id": tag_id, "key": tag_key})
    dataset = {
        "tags": tags_list,
        "tagsData": dataset_tags_data
    }
    return dataset


def classify_and_rate_table_columns(
        poseidon_client,
        template_id,
        tables_data,
        class_path_to_obj,
        rating_name_to_obj,
        datasource_id,
        datasource_name,
        add_tagging=False,
        tables_names_to_tag=[]):

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
    assert len(list_tables_resp_obj["assets"]) == len(tables_data), len(
        list_tables_resp_obj["assets"])

    for asset in list_tables_resp_obj["assets"]:
        table_asset_id = asset["id"]

        table_name = asset["name"]
        if tables_names_to_tag and table_name not in tables_names_to_tag:
            logger.info(f"skip tagging table {table_name}")
            continue
        logger.info(f"tagging columns in table {table_name}")
        table_data = tables_data[table_name]

        # build column comment 2 column id
        list_columns_resp = poseidon_client.list_columns(
            asset_id=table_asset_id)
        assert list_columns_resp.status_code == 200, list_columns_resp.text
        list_columns_resp_obj = json.loads(list_columns_resp.text)

        table_class_ratings = []
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

            col_meta_data = column_obj["meta_data"]
            class_path = col_meta_data["class_path"]
            rating_name = col_meta_data["rating_name"]

            logger.info(
                f"===== tagging column_name {column_name}(column_comment: {column_comment}) with class_path {class_path}")
            class_data = class_path_to_obj[class_path]
            rating_data = rating_name_to_obj[rating_name]
            class_id = class_data["id"]
            raiting_id = rating_data["id"]

            # Create tags
            if class_id not in dataset_tags:
                dataset_tags[class_id] = class_path
                dataset_tags_data[class_id] = {
                    "data": []
                }
            dataset_tags_data[class_id]["data"].append(
                {"id": column_id, "name": column_name, "path": column_path}
            )

            # class_rating = {"key": class_path, "value": "true",
            #                 "subjectId": column_id,
            #                 "description": class_path,
            #                 "subjectType": "COLUMN"}
            class_rating = {"ratingId": raiting_id, "classificationId": class_id,
                            "subjectType": "COLUMN", "subjectId": column_id, "templateId": template_id}
            table_class_ratings.append(class_rating)

        if add_tagging:
            add_column_classification_ratings_resp = poseidon_client.add_column_classification_ratings(
                classification_ratings=table_class_ratings)
            assert add_column_classification_ratings_resp.status_code == 200, add_column_classification_ratings_resp.text

    tags_list = []
    for tag_id, tag_key in dataset_tags.items():
        tags_list.append({"id": tag_id, "key": tag_key})
    dataset = {
        "templateId": template_id,
        "tags": tags_list,
        "tagsData": dataset_tags_data
    }
    return dataset


def get_client(host="10.0.1.48", port="7758"):
    return PoseidonClient(host=host, port=port)


if __name__ == '__main__':
    poseidon_client = PoseidonClient(host="10.0.1.48", port="7758")

    dataset = {
        "tags": [{"id": "19", "key": "个人信息"}],
        "tagsData": {
            "19": {
                "data": [
                    {"id": "22202", "name": "age",
                     "path": "default/demo_1019_train/pandora_demo_1019_100_10_train/age"},
                    {"id": "22203", "name": "birthday",
                     "path": "default/demo_1019_train/pandora_demo_1019_100_10_train/birthday"},
                    {"id": "22206", "name": "gender",
                     "path": "default/demo_1019_train/pandora_demo_1019_100_10_train/gender"},
                    {"id": "22207", "name": "gender_digit",
                     "path": "default/demo_1019_train/pandora_demo_1019_100_10_train/gender_digit"},
                    {"id": "22212", "name": "name_cn", "path": "default/demo_1019_train/pandora_demo_1019_100_10_train/name_cn"}]
            }
        }
    }
    training_resp = poseidon_client.start_training(
        model_name="test_poseidon_client",
        dataset=dataset,
    )
    assert training_resp.status_code == 200, training_resp.text
    raise

    # create datasource
    create_resp = poseidon_client.create_datasource(
        "test_real_data_manual", collection_id=2)
    assert create_resp.status_code == 200, create_resp.text

    # get datasource ID
    response_create_obj = json.loads(create_resp.text)
    datasource_id = response_create_obj["id"]

    # metasync
    sync_resp = poseidon_client.do_meta_sync(
        datasource_id=datasource_id, path="test_real_data_manual")
    assert sync_resp.status_code == 200, sync_resp.text

    # list databases - should have only 1
    list_dbs_resp = poseidon_client.list_databases(datasource_id=datasource_id)
    assert list_dbs_resp.status_code == 200, list_dbs_resp.text
    list_dbs_resp_obj = json.loads(list_dbs_resp.text)
    assert len(list_dbs_resp_obj["assets"]) == 1
    db_id = list_dbs_resp_obj["assets"][0]["id"]
    db_name = list_dbs_resp_obj["assets"][0]["name"]

    # list tables in the DB
    list_tables_resp = poseidon_client.list_tables(
        datasource_id=datasource_id, db_name=db_name)
    assert list_tables_resp.status_code == 200, list_tables_resp.text
    list_tables_resp_obj = json.loads(list_tables_resp.text)
    assert len(list_tables_resp_obj["assets"]) == 34, len(
        list_tables_resp_obj["assets"])

    # list columns in a table
    asset = list_tables_resp_obj["assets"][0]
    asset_id = asset["id"]
    list_columns_resp = poseidon_client.list_columns(asset_id=asset_id)
    assert list_columns_resp.status_code == 200, list_columns_resp.text
    list_columns_resp_obj = json.loads(list_columns_resp.text)
    column = list_columns_resp_obj["columns"][0]
    print(column)
    column_id = column["id"]

    # add and delete tags
    tags = [{"key": "互联网金融管理政策文件", "value": "true",
             "subjectId": column_id, "subjectType": "COLUMN"}]
    add_tag_resp = poseidon_client.add_column_tags(tags=tags)
    assert add_tag_resp.status_code == 200, add_tag_resp.text
    delete_tag_resp = poseidon_client.delete_column_tags(tags=tags)
    assert delete_tag_resp.status_code == 200, delete_tag_resp.text

    # delete
    delete_resp = poseidon_client.delete_datasource(
        datasource_id=datasource_id)
    assert delete_resp.status_code == 200, delete_resp.text
