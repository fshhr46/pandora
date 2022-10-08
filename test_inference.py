
import json
import logging
import pathlib

import pandora.packaging.inference as inference
import pandora.service.job_runner as job_runner
import pandora.tools.test_utils as test_utils

from pandora.tools.common import logger
from pandora.callback.progressbar import ProgressBar
from pandora.tools.common import init_logger

device = test_utils.get_device()


def inference_online(data_obj):
    # time.sleep(1)
    model_name = "short_sentence"
    version = "1"
    url = "http://localhost:38080/predictions"
    url = f"{url}/{model_name}/{version}"
    result = test_utils.make_request(url, False,
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
    model_type, local_rank, tokenizer, model, processor = test_utils.load_model(
        device)
    dataset, _, id2label, _ = test_utils.load_dataset(
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
    _, local_rank, tokenizer, model, processor = test_utils.load_model(device)
    _, dataloader, id2label, _ = test_utils.load_dataset(
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
            mode=test_utils.HANDLER_MODE,
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
    assert test_offline_train(lines) == 0

    # Test inferencing by loading a model offline and calling inference.run_inference offline
    assert test_offline(lines) == 0


if __name__ == '__main__':
    run_test()
