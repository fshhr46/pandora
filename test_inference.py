import pandora.packaging.inference as inference
from pandora.tools.common import init_logger
import torch
import os
import requests
import json
import pprint

from pathlib import Path

import pandora.service.job_runner as job_runner
from pandora.dataset.sentence_data import Dataset
import pandora.packaging.feature as feature
import pandora.tools.runner_utils as runner_utils
from pandora.tools.common import logger
from pandora.callback.progressbar import ProgressBar

TEST_DATASETS = [
    Dataset.synthetic_data
]

datasets = TEST_DATASETS

device = torch.device("cuda")


def load_model():

    home = str(Path.home())
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


def load_dataset(local_rank, tokenizer, processor, lines):

    partition = "test"
    examples = processor.create_examples(
        feature.read_json_lines(lines), partition)
    label_list = processor.get_labels()
    id2label = {}
    for i, label in enumerate(label_list):
        id2label[i] = label
    features = feature.convert_examples_to_features(
        examples, label_list, max_seq_length=128, tokenizer=tokenizer)
    dataset = runner_utils.convert_features_to_dataset(
        local_rank, features, True)
    return dataset, id2label


def make_request(url: str, post: bool = False, data=None):
    if post:
        url_obj = requests.post(url, data=data)
    else:
        url_obj = requests.get(url, data=data)
    text = url_obj.text
    data = json.loads(text)
    return data


def inference_online(text: str, pp):
    # time.sleep(1)
    model_name = "synthetic"
    version = "1"
    url = f"http://localhost:18080/predictions/{model_name}/{version}"
    result = make_request(url, False, {"data": text})
    return result


def test_online(lines):
    pp = pprint.PrettyPrinter(indent=4)

    incorrect = 0
    for line in lines:
        obj = json.loads(line)
        text = obj["text"]
        res = inference_online(text, pp)
        pred = res["class"]
        label = obj["label"][0]
        if pred != label:
            incorrect += 1
            logger.info("")
            logger.info(pred)
            logger.info(label)
            logger.info(res)
            logger.info(obj)
    logger.info(f"incorrect: {incorrect}")


def test_offline(lines):
    pp = pprint.PrettyPrinter(indent=4)
    model_type, local_rank, tokenizer, model, processor = load_model()
    dataset, id2label = load_dataset(
        local_rank, tokenizer, processor, lines)

    predictions = job_runner.predict(
        model_type, model,
        id2label, dataset,
        local_rank, device,
    )
    assert len(predictions) == len(dataset)

    incorrect = 0
    for pred, line in zip(predictions, lines):
        obj = json.loads(line)
        if pred["tags"][0] != obj["label"][0]:
            incorrect += 1
            logger.info("========")
            logger.info(pred)
            logger.info(obj)
    logger.info(f"incorrect: {incorrect}")


def compare(lines):
    _, local_rank, tokenizer, model, processor = load_model()
    dataset, id2label = load_dataset(
        local_rank, tokenizer, processor, lines)
    incorrect = 0
    pbar = ProgressBar(n_total=len(dataset), desc='inferencing')
    for step, (line, data_entry) in enumerate(zip(lines, dataset)):
        # time.sleep(1)
        # logger.info("========")
        obj = json.loads(line)
        pred = obj["pred"][0]
        label = obj["label"][0]

        # off-load tensors to device
        data_entry = tuple(t.to(device) for t in data_entry)

        input_ids_batch = data_entry[0][None, :].to(device)
        attention_mask_batch = data_entry[1][None, :].to(device)
        token_type_ids_batch = data_entry[2][None, :].to(device)

        # TODO: Fix hard coded "sequence_classification"
        # input_ids_batch, attention_mask_batch, token_type_ids, indexes
        batch = (input_ids_batch, attention_mask_batch,
                 token_type_ids_batch, [])
        inf_results = inference.run_inference(
            "sequence_classification", model, id2label, batch)
        assert len(inf_results) == 1
        inf_offline = inf_results[0]["class"]

        assert len(inf_offline) == len(pred)
        if inf_offline != pred:
            incorrect += 1
            logger.info("")
            logger.info(label)
            logger.info(inf_offline)
            logger.info(pred)
            logger.info(inf_results)
        pbar(step)
    logger.info(f"incorrect: {incorrect}")


if __name__ == '__main__':
    init_logger(log_file=None)
    # =========== test
    # test_file = "/home/haoranhuang/workspace/resource/datasets/synthetic_data/test.json"
    # test_file = "/home/haoranhuang/workspace/resource/outputs/bert-base-chinese/synthetic_data_1000/predict/test_submit.json"
    test_file = "/home/haoranhuang/workspace/resource/outputs/bert-base-chinese/synthetic_data/predict/test_submit.json"
    data = open(test_file)
    # data = [
    #     json.dumps({"text": "440602199110191812",
    #                "label": ["个人信息"], "pred": ["个人信息"]}, ),
    #     json.dumps({"text": "13702557628", "label": ["个人信息"]}),
    #     json.dumps({"text": "137-02557628", "label": ["个人信息"]}),
    # ]
    lines = [line for line in data]
    test_offline(lines)
    # test_online(lines)
    # compare(lines)
