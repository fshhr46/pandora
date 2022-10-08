import torch
import os
import requests
import json
import pathlib

from torch.utils.data import DataLoader, SequentialSampler

import pandora.tools.mps_utils as mps_utils
import pandora.packaging.feature as feature
import pandora.tools.runner_utils as runner_utils
import pandora.service.job_runner as job_runner

from pandora.tools.common import logger
from pandora.dataset.sentence_data import Dataset
from pandora.packaging.feature import batch_collate_fn


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


def load_model(device):

    home = str(pathlib.Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    task_name = "sentence"
    mode_type = "bert"
    bert_base_model_name = "bert-base-chinese"
    arg_list = job_runner.get_training_args(
        task_name=task_name,
        mode_type=mode_type,
        bert_base_model_name=bert_base_model_name,
    )

    TEST_DATASETS = [
        Dataset.short_sentence
        # "pandora_demo_meta_100_10"
    ]

    datasets = TEST_DATASETS

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
        resource_dir=resource_dir, datasets=datasets)

    model_classes = job_runner.MODEL_CLASSES[args.model_type]

    # load trained model and tokenizer
    model_package_dir = os.path.join(args.output_dir, "torchserve_package")
    config_class, model_class, tokenizer_class = model_classes
    tokenizer = tokenizer_class.from_pretrained(
        model_package_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(model_package_dir)

    # prepare model
    model.to(device)
    model.eval()

    return args.model_type, args.local_rank, tokenizer, model, processor


def load_dataset(local_rank, tokenizer, processor, lines, batch_size):

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
    features = feature.convert_examples_to_features(
        examples, label_list, max_seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)
    dataset = runner_utils.convert_features_to_dataset(
        local_rank, features, True)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                            collate_fn=batch_collate_fn)
    logger.info("========================= Done loading dataset")
    return dataset, dataloader, id2label, label2id


def make_request(url: str, post: bool = False, data=None, headers=None):
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data
