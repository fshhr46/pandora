import torch
import os
import json
import pathlib
from pathlib import Path

from torch.utils.data import DataLoader, SequentialSampler

import pandora
import pandora.tools.mps_utils as mps_utils
import pandora.packaging.feature as feature
import pandora.tools.runner_utils as runner_utils
import pandora.tools.training_utils as training_utils

from pandora.tools.common import logger
from pandora.packaging.feature import (
    batch_collate_fn_bert,
    batch_collate_fn_char_bert,
)

from pandora.packaging.model import BertBaseModelType


MAX_SEQ_LENGTH = 64
HANDLER_MODE = "sequence_classification"


def get_device():

    # TODO: Setup M1 chip
    if torch.cuda.is_available():
        device_name = "cuda"
    elif mps_utils.has_mps:
        device_name = "mps"
    else:
        device_name = "cpu"
    logger.info(f"device_name is {device_name}")
    device = torch.device(device_name)
    return device


def get_test_data():
    data = [
        {"guid": "test-0", "text": "对不起，我们凤县只拿实力说话",
            "label": ["news_finance"], "pred": ["news_agriculture"]},
        {"guid": "test-1", "text": "泰国有哪些必买的化妆品？",
            "label": ["news_world"], "pred": ["news_world"]},
        {"guid": "test-2", "text": "1993年美国为什么要出兵索马里？",
            "label": ["news_world"], "pred": ["news_world"]},
        {"guid": "test-3", "text": "去年冲亚冠恒大“送”三分，如今近况同样不佳的华夏会还人情吗？",
            "label": ["news_sports"], "pred": ["news_sports"]},
        {"guid": "test-4", "text": "毛骗中的邵庄",
            "label": ["news_entertainment"], "pred": ["news_travel"]},
        {"guid": "test-5", "text": "网上少儿编程的培训机构有哪些？",
            "label": ["news_edu"], "pred": ["news_edu"]},
        {"guid": "test-6", "text": "鬼谷子话术，一句话教你不被别人牵着鼻子走",
            "label": ["news_edu"], "pred": ["news_entertainment"]},
        {"guid": "test-7", "text": "我没钱只买得起盗版书，这可耻吗？",
            "label": ["news_tech"], "pred": ["news_culture"]},
        {"guid": "test-8", "text": "87版红楼梦里最幸福的两个人，因为红楼梦结缘，情定终身",
            "label": ["news_entertainment"], "pred": ["news_culture"]},
        {"guid": "test-9", "text": "颜强专栏：更衣室里的千万奖金",
            "label": ["news_sports"], "pred": ["news_sports"]},
        {"guid": "test-10", "text": "6年老员工结婚，收到老板送上的“大”红包，婚假结束他就辞职了",
            "label": ["news_story"], "pred": ["news_story"]},
        {"guid": "test-11", "text": "“敬礼娃娃”郎铮：想找全救我的解放军叔叔",
            "label": ["news_sports"], "pred": ["news_military"]},
        {"guid": "test-12", "text": "现在煤炭价格多少钱一吨?2018年煤炭行情预测会如何?上涨还是下跌？",
            "label": ["news_agriculture"], "pred": ["news_finance"]},
        {"guid": "test-13", "text": "生物战已让普京说破，我们是否应该警醒？",
            "label": ["news_military"], "pred": ["news_military"]},
        {"guid": "test-14", "text": "去日本购物有哪些APP可以参考呢？",
            "label": ["news_tech"], "pred": ["news_travel"]},
        {"guid": "test-15", "text": "海信电器澄清：未参与斯洛文尼亚家电收购事宜",
            "label": ["news_tech"], "pred": ["news_finance"]},
        {"guid": "test-16", "text": "想咨询河北大学生医保和农村医保问题，去哪里咨询？",
            "label": ["news_edu"], "pred": ["news_edu"]},
        {"guid": "test-17", "text": "农村俗语“男不得初一，女不得十五”，是什么意思？",
            "label": ["news_agriculture"], "pred": ["news_culture"]},
        {"guid": "test-18", "text": "如何评价国乒女队教练李隼：日本差中国非一星半点？",
            "label": ["news_sports"], "pred": ["news_sports"]},
        {"guid": "test-19", "text": "你吃过大菜糕吗？出了百色买不到！",
            "label": ["news_travel"], "pred": ["news_travel"]},
        {"guid": "test-20", "text": "无标题文章",
            "label": ["news_edu"], "pred": ["news_world"]},
    ]
    return [json.dumps(line, ensure_ascii=False) for line in data]


def load_model_for_test(device, datasets, model_package_dir, training_type, meta_data_types, loss_type):
    import pandora.service.job_runner as job_runner
    home = str(pathlib.Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    bert_model_type = "bert"
    bert_base_model_name = "bert-base-chinese"

    num_epochs = training_utils.get_num_epochs(
        training_type,
        meta_data_types,
        bert_model_type=bert_model_type,
    )

    batch_size = training_utils.get_batch_size(
        training_type,
        meta_data_types,
        bert_model_type=bert_model_type,
    )

    arg_list = job_runner.get_training_args(
        # model args
        bert_model_type=bert_model_type,
        bert_base_model_name=bert_base_model_name,
        training_type=training_type,
        meta_data_types=meta_data_types,
        loss_type=loss_type,
        # training args
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    arg_list.extend(
        job_runner.get_default_dirs(
            resource_dir,
            cache_dir,
            bert_base_model_name=bert_base_model_name,
            datasets=datasets,
        ))
    arg_list.extend(
        job_runner.set_actions(
            do_train=False,
            do_eval=False,
            do_predict=True,
        ))

    # =========== load artifacts
    parser = job_runner.get_args_parser()
    args = parser.parse_args(arg_list)

    processor = runner_utils.get_data_processor(
        datasets=datasets,
        training_type=training_type,
        meta_data_types=meta_data_types,
        resource_dir=resource_dir)

    model_classes = job_runner.MODEL_CLASSES[args.bert_model_type]

    # load trained model and tokenizer
    config_class, model_class, tokenizer_class = model_classes
    tokenizer = tokenizer_class.from_pretrained(
        model_package_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(model_package_dir)

    # prepare model
    model.to(device)
    model.eval()

    return args.bert_model_type, args.local_rank, tokenizer, model, processor


def load_dataset(local_rank, tokenizer, processor, lines, batch_size):

    bert_base_model_type = BertBaseModelType.bert
    logger.info("========================= Start loading dataset")
    partition = "test"
    examples = processor.create_examples(
        feature.read_json_lines(lines), partition)

    label_list = processor.get_labels()
    id2label = {}
    label2id = {}
    for i, label in enumerate(label_list):
        id2label[i] = label
        label2id[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        feat = feature.convert_example_to_feature(
            example,
            processor.training_type,
            processor.meta_data_types,
            label2id,
            ex_index < 5,
            MAX_SEQ_LENGTH,
            tokenizer,
            char2ids_dict=None)
        features.append(feat)

    include_char_data = bert_base_model_type == BertBaseModelType.char_bert
    dataset = feature.convert_features_to_dataset(
        local_rank, features, evaluate=True, include_char_data=include_char_data)

    sampler = SequentialSampler(dataset)

    batch_collate_fn = batch_collate_fn_char_bert if bert_base_model_type == BertBaseModelType.char_bert else batch_collate_fn_bert
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            collate_fn=batch_collate_fn)
    logger.info("========================= Done loading dataset")
    return dataset, examples, dataloader, id2label, label2id


def make_request(url: str, post: bool = False, data=None, headers=None):
    import requests
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data


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
            [f"`{col}` INT COMMENT '{column_name_2_comment[col]}'" for col in column_names])
        sql = f'''CREATE TABLE {table_name} (\n{columns_query}\n)'''
        print(f"creating table with query:\n {sql}")
        cursor.execute(sql)
    except mysql.connector.Error as error:
        print("Failed to create table {}".format(error))
        raise error

    try:
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
        raise error

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


def create_tables(
        host,
        port,
        database_name,
        tables_data):

    # create database
    create_database(
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
        cleanup_table(
            table_name=table_name,
            database_name=database_name,
            host=host,
            port=port,
        )
        logger.warning(f"deleted table {table_name}")

        ingest_to_mysql(
            table_name=table_name,
            database_name=database_name,
            host=host,
            port=port,
            column_names=column_names,
            column_name_2_comment=column_name_2_comment,
            dataset=[])


def create_database(
    database_name,
    host="10.0.1.178",
    port="7733",
    user="root",
    password="Sudodata-123",
):
    # alias vm_178_mysql_l="mysql -h 10.0.1.178 -u root -pSudodata-123 -P 7733"
    # alias vm_178_mysql="mysql -h 10.0.1.178 -u root -pSudodata -P 16969"
    # alias vm_140_mysql="mysql -h 10.0.1.140 -u root -pSudodata-123 -P 33039"
    # alias vm_67_mysql="mysql -h 10.0.1.67 -u root -pSudodata-123 -P 33039"
    # alias vm_me_mysql="mysql -h 10.0.1.48 -u root -pSudodata-123 -P 33039"

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


def cleanup_table(
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
        cursor.execute(f"use {database_name}")
        # delete table if exists
        cursor.execute(f"drop table {table_name}")
    except Exception as e:
        print(f"table {table_name} does not exists")
    connection.commit()


def get_test_data_dir():
    pandora_path = os.path.dirname(pandora.__file__)
    pandora_work_dir = Path(pandora_path).parent.absolute()

    test_data_dir = os.path.join(
        pandora_work_dir, "test_data")
    return test_data_dir
