import os
import json
import requests
from typing import Dict, Tuple, List

import torch

from pandora.tools.common import logger
from pandora.packaging.feature import TrainingType, ModelType

import pandora.dataset.poseidon_data as poseidon_data
import pandora.dataset.dataset_utils as dataset_utils
import pandora.service.job_utils as job_utils


def extract_keyword(
        server_dir: str,
        job_id: str) -> Tuple[bool, Dict, str]:
    dataset_path = job_utils.get_dataset_file_path(server_dir, job_id)
    if not os.path.isfile(dataset_path):
        return False, f"dataset file {dataset_path} not exists"
    try:
        # load dataset file
        _, _, tags_by_id, data_by_tag_ids, dataset = poseidon_data.load_poseidon_dataset(
            dataset_path)

        # get training type
        training_type = TrainingType(dataset["data_type"])
        logger.info(f"training_type is {training_type}")

        # outputs
        keywords = []

        for tag_id, data_by_col_id in data_by_tag_ids.items():
            logger.info("====")
            logger.info(tag_id)

            for col_id, col_data_entries in data_by_col_id.items():
                logger.info(col_id)
                for data_entry in col_data_entries:
                    logger.info(data_entry)
                    # json.dump(obj, ensure_ascii=False,
                    #     cls=dataset_utils.DataEntryEncoder)

                    request_data = {
                        "data": data_entry.text,
                        "meta_data": data_entry.meta_data,
                    }
                    pred_result = inference_online(
                        host="5y72098x40.zicp.fun",
                        port="38080",
                        data_obj=request_data,
                        model_name="39",
                    )
                    logger.info(pred_result)

                    pred_online = pred_result["class"]
                    probability = pred_result["probability"]

                    logger.info("")
                    logger.info(
                        "======================================================")
                    # logger.info(f"pred_online: {pred_online}")
                    # logger.info(f"label: {label}")

                    request_data["target"] = 0
                    insight = explanations_online(
                        host="5y72098x40.zicp.fun",
                        port="38080",
                        data_obj=request_data,
                        model_name="39",
                    )
                    logger.info(pred_result)
                    words = insight["words"]
                    attributions = insight["importances"]
                    delta = insight["delta"]

                    positions = list(range(len(words)))
                    combined = list(zip(words, positions, attributions))
                    sorted_attributions = sorted(
                        combined, key=lambda tp: tp[2], reverse=True)

                    obj = {
                        "text": data_entry.text,
                        "probability": probability,
                        "pred_online": pred_online,
                        "label": data_entry.label[0],
                        "attributions_sum": torch.tensor(attributions).sum().item(),
                        "delta": delta,
                        "sorted_attributions": sorted_attributions,
                    }
                    logger.info(obj)
                    # json_objs.append(obj)
            continue

    except ValueError as e:
        return False, e
    return True, ""


def make_request(url: str, post: bool = False, data=None, headers=None):
    if post:
        url_obj = requests.post(url, data=data, headers=headers)
    else:
        url_obj = requests.get(url, data=data, headers=headers)
    text = url_obj.text
    data = json.loads(text)
    return data


def inference_online(
        host: str,
        port: str,
        data_obj,
        model_name: str,
        model_version: str = None):
    url = f"http://{host}:{port}/predictions/{model_name}"
    if model_version:
        url = f"{url}/{model_version}"
    result = make_request(url, False,
                          data=data_obj,
                          headers={'content-type': "application/x-www-form-urlencoded"})
    return result


def explanations_online(
        host: str,
        port: str,
        data_obj,
        model_name: str,
        model_version: str = None):
    url = f"http://{host}:{port}/explanations/{model_name}"
    if model_version:
        url = f"{url}/{model_version}"
    result = make_request(url, False,
                          data=data_obj,
                          headers={'content-type': "application/x-www-form-urlencoded"})
    return result
