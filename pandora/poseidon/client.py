import requests
import json


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

    # ====== Training related APIs
    def start_training(
        self,
        model_name,
        dataset,
        traning_data_type="META_DATA",
        metadata_types=["COLUMN_COMMENT"],
        model_type="TAG",
    ):
        data = {
            "baseModelId": "1",
            "name": model_name,
            "dataSet": dataset,
            "traningDataType": traning_data_type,
            "metadataTypes": metadata_types,
            "modelType": model_type,
        }
        return requests.post(
            self._get_url(f"gapi/recognition/training/job/initial"),
            data=json.dumps(data),
            headers=self._get_headers())


def get_client():
    return PoseidonClient(host="10.0.1.48", port="7758")


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
