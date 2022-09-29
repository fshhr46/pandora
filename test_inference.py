import torch
import os
import requests
import json
import logging
import pathlib

from torch.utils.data import DataLoader, SequentialSampler
from captum.attr import visualization

import pandora.packaging.feature as feature
import pandora.tools.runner_utils as runner_utils
import pandora.tools.mps_utils as mps_utils
import pandora.packaging.inference as inference
import pandora.service.job_runner as job_runner

from pandora.tools.common import logger
from pandora.callback.progressbar import ProgressBar
from pandora.tools.common import init_logger
from pandora.dataset.sentence_data import Dataset
from pandora.packaging.feature import batch_collate_fn

# TODO: Setup M1 chip
if torch.cuda.is_available():
    device_name = "cuda"
elif mps_utils.has_mps:
    device_name = "mps"
else:
    device_name = "cpu"
logger.info(f"device_name is {device_name}")
device = torch.device(device_name)

MAX_SEQ_LENGTH = 64
HANDLER_MODE = "sequence_classification"


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


def load_model():

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


def inference_online(data_obj):
    # time.sleep(1)
    model_name = "short_sentence"
    version = "1"
    url = "http://localhost:38080/predictions"
    url = f"{url}/{model_name}/{version}"
    result = make_request(url, False,
                          data=data_obj,
                          headers={'content-type': "application/x-www-form-urlencoded"})
    return result


def test_online(lines):
    incorrect = 0
    pbar = ProgressBar(n_total=len(lines), desc='comparing')
    for step, line in enumerate(lines):

        # Read baseline file data
        obj = json.loads(line)
        pred_offline = obj["pred"][0]
        label = obj["label"][0]

        # Run inference online
        request_data = {
            "data": obj["text"],
            "column_name": obj.get("column_name")
        }

        res = inference_online(request_data)
        pred_online = res["class"]

        # Compare
        if pred_online != pred_offline:
            incorrect += 1
            logger.info("")
            logger.info("=========")
            logger.info(
                f"pred_online: {pred_online}, pred_offline: {pred_offline}")
            logger.info(f"label: {label}")
            logger.info(res)
            logger.info(obj)
        pbar(step)
    logger.info(f"incorrect: {incorrect}")
    return incorrect


def test_offline_train(lines, batch_size=100):
    model_type, local_rank, tokenizer, model, processor = load_model()
    dataset, _, id2label, _ = load_dataset(
        local_rank, tokenizer, processor, lines, batch_size=batch_size)

    predictions = job_runner.predict(
        model_type, model,
        id2label, dataset,
        local_rank, device,
        batch_size=batch_size,
    )
    assert len(predictions) == len(dataset)

    incorrect = 0
    pbar = ProgressBar(n_total=len(lines), desc='comparing')
    for step, (pred, line) in enumerate(zip(predictions, lines)):

        # Read baseline file data
        obj = json.loads(line)
        pred_offline = obj["pred"][0]
        label = obj["label"][0]

        # Read prediction from result
        pred_online = pred["tags"][0]

        # Compare
        if pred_online != pred_offline:
            incorrect += 1
            logger.info("")
            logger.info("=========")
            logger.info(
                f"pred_online: {pred_online}, pred_offline: {pred_offline}")
            logger.info(f"label: {label}")
            logger.info(pred)
            logger.info(obj)
            raise
        pbar(step)
    logger.info(f"incorrect: {incorrect}")
    return incorrect


def test_offline(lines, batch_size=20):
    _, local_rank, tokenizer, model, processor = load_model()
    _, dataloader, id2label, _ = load_dataset(
        local_rank, tokenizer, processor, lines, batch_size=batch_size)
    incorrect = 0
    total = 0
    pbar = ProgressBar(n_total=len(lines), desc='comparing')
    for step, input_batch in enumerate(dataloader):
        input_batch = tuple(t.to(device) for t in input_batch)
        input_batch_with_index = (
            input_batch[0], input_batch[1], input_batch[2], [])
        # TODO: Fix hard coded "sequence_classification"
        inferences = inference.run_inference(
            input_batch=input_batch_with_index,
            mode=HANDLER_MODE,
            model=model)
        results = inference.format_outputs(
            inferences=inferences, id2label=id2label)

        sub_lines = lines[step * batch_size: (step+1) * batch_size]
        assert len(results) == len(sub_lines)
        for res, line in zip(results, sub_lines):
            # Read baseline file data
            obj = json.loads(line)
            pred_offline = obj["pred"][0]
            label = obj["label"][0]
            pred_online = res["class"]

            # Compare
            if pred_online != pred_offline:
                incorrect += 1
                logger.info("")
                logger.info("=========")
                logger.info(
                    f"pred_online: {pred_online}, pred_offline: {pred_offline}")
                logger.info(f"label: {label}")
                logger.info(res)
                logger.info(obj)
            pbar(total)
            total += 1
    logger.info(f"incorrect: {incorrect}")
    return incorrect


def test_get_insights(lines, batch_size=40, n_steps=50, dump_output=False, visualize_output=False):
    _, local_rank, tokenizer, model, processor = load_model()
    _, dataloader, id2label, label2id = load_dataset(
        local_rank, tokenizer, processor, lines, batch_size=batch_size)
    total = 0
    pbar = ProgressBar(n_total=len(lines), desc='Attributing')
    json_objs = []
    for step, input_batch in enumerate(dataloader):
        input_batch = tuple(t.to(device) for t in input_batch)
        input_batch_with_index = (
            input_batch[0], input_batch[1], input_batch[2], [])
        # TODO: Fix hard coded "sequence_classification"
        inferences = inference.run_inference(
            input_batch=input_batch_with_index,
            mode=HANDLER_MODE,
            model=model)
        results = inference.format_outputs(
            inferences=inferences, id2label=id2label)

        sub_lines = lines[step * batch_size: (step+1) * batch_size]
        assert len(results) == len(sub_lines)

        targets = [label2id[res["class"]] for res in results]
        # TODO: Fix hard coded "sequence_classification"
        insights = inference.run_get_insights(
            # configs
            mode=HANDLER_MODE,
            embedding_name="bert",
            captum_explanation=True,
            # model related
            model=model,
            tokenizer=tokenizer,
            # device
            device=device,
            # input related
            input_batch=input_batch_with_index,
            target=targets,
            n_steps=n_steps)
        torch.cuda.empty_cache()
        for res, insight, line in zip(results, insights, sub_lines):

            # Read baseline file data
            obj = json.loads(line)
            label = obj["label"][0]
            sentence = obj["text"]

            pred_online = res["class"]
            probability = res["probability"]

            request_data = {
                "data": obj["text"],
                "column_name": obj.get("column_name")
            }
            logger.info("")
            logger.info(
                "======================================================")
            # logger.info(f"request_data is {request_data}")
            # logger.info(f"pred_online: {pred_online}")
            # logger.info(f"label: {label}")

            response = insight
            delta = response["delta"]

            non_pad_words = list(
                filter(lambda word: word != tokenizer.pad_token, response["words"]))
            non_pad_attributions = response["importances"][:len(
                non_pad_words)]
            positions = list(range(len(non_pad_words)))
            combined = list(
                zip(non_pad_words, positions, non_pad_attributions))
            sorted_attributions = sorted(
                combined, key=lambda tp: tp[2], reverse=True)
            attributions = torch.tensor(response["importances"])

            obj = {
                "sentence": sentence,
                "probability": probability,
                "pred_online": pred_online,
                "label": label,
                "attributions_sum": attributions.sum().item(),
                "delta": delta,
                "sorted_attributions": sorted_attributions,
            }
            json_objs.append(obj)
            # logger.info(json.dumps(obj,
            #                        ensure_ascii=False,
            #                        indent=4))
            pbar(total)
            total += 1

    label_2_keywords = build_keyword_dict(json_objs)
    if dump_output:
        with open("attributions.json", 'w') as f:
            for json_obj in json_objs:
                json.dump(json_obj, f, ensure_ascii=False)
        with open("keywords.json", 'w') as f:
            json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)
    if visualize_output:
        return visualize_insights(json_objs=json_objs)


def build_keyword_dict(json_objs):

    label_to_keyword_attrs = {}
    for obj in json_objs:
        label = obj["label"]
        if label not in label_to_keyword_attrs:
            label_to_keyword_attrs[label] = {}

        attributions = label_to_keyword_attrs[label]
        for entry in obj["sorted_attributions"]:
            word, pos, attribution = entry
            attributions[word] = attributions.get(word, 0) + attribution

    label_to_keyword_attrs_sorted = {}
    for label, keywords in label_to_keyword_attrs.items():
        keywords_sorted = sorted(
            keywords.items(), key=lambda k_v: k_v[1], reverse=True)
        label_to_keyword_attrs_sorted[label] = keywords_sorted
    return label_to_keyword_attrs_sorted


def visualize_insights(json_objs):
    vis_data_records_ig = []
    for obj in json_objs:
        sorted_attributions = obj["sorted_attributions"]
        sorted_attributions = sorted(
            sorted_attributions, key=lambda tp: tp[1])
        attributions = torch.tensor(
            [item[2] for item in sorted_attributions], dtype=torch.float32)
        sentence = obj["sentence"][:MAX_SEQ_LENGTH-2]
        assert len(sentence) == len(
            attributions), f"{len(sentence)} == {len(attributions)}"

        vis_data_records_ig.append(
            visualization.VisualizationDataRecord(
                attributions,
                obj["probability"],
                obj["pred_online"],
                obj["label"],
                "[PAD]",
                obj["attributions_sum"],
                sentence,
                obj["delta"]))
    return visualization.visualize_text(vis_data_records_ig)


def run_test():
    init_logger(log_file=None, log_file_level=logging.DEBUG)

    home = str(pathlib.Path.home())
    # =========== test
    test_file = f"{home}/workspace/resource/datasets/synthetic_data/test.json"
    test_file = f"{home}/workspace/resource/outputs/bert-base-chinese/synthetic_data_1000/predict/test_submit.json"
    test_file = f"{home}/workspace/resource/outputs/bert-base-chinese/synthetic_data/predict/test_submit.json"
    test_file = f"{home}/workspace/resource/outputs/bert-base-chinese/short_sentence/predict/test_submit.json"
    # test_file = "/home/haoranhuang/workspace/resource/outputs/bert-base-chinese/pandora_demo_meta_100_10/predict/test_submit.json"

    lines = open(test_file).readlines()

    # Test inferencing by calling an online model registered in torchserve
    # assert test_online(lines) == 0

    # Test inferencing by loading a model offline and calling predict in training pipeline (batch)
    # assert test_offline_train(lines) == 0

    # Test inferencing by loading a model offline and calling inference.run_inference offline
    # assert test_offline(lines) == 0

    # Run get insights
    test_get_insights(lines, 5, 30, True, True)


if __name__ == '__main__':
    run_test()
