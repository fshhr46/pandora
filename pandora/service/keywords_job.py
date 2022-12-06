from genericpath import isfile
import os
import json
import requests
import jieba
import jieba.posseg as pseg
from pandora.tools.common import init_logger, logger


from typing import Dict, Tuple, List

import torch
import torch.multiprocessing as mp

from pandora.tools.common import logger
from pandora.packaging.feature import (
    TrainingType,
    create_example,
    get_text_from_example,
)
from pandora.service.job_utils import JobStatus, JobType

import pandora.dataset.poseidon_data as poseidon_data
import pandora.service.job_utils as job_utils

KEYWORD_FILE_NAME = "keywords.json"


class KeywordExtractionJob(object):

    def __init__(
            self,
            output_dir: str,
            job_id: str,
            host: str,
            port: str,
            dataset_path: str,
            keyword_file_path: str,
            model_name: str,
            model_version: str = None):

        self.output_dir = output_dir
        self.job_id = job_id
        self.host = host
        self.port = port
        self.dataset_path = dataset_path
        self.keyword_file_path = keyword_file_path
        self.model_name = model_name
        self.model_version = model_version

    def __call__(self, *args, **kwds) -> None:
        extract_keywords(
            output_dir=self.output_dir,
            job_id=self.job_id,
            host=self.host,
            port=self.port,
            dataset_path=self.dataset_path,
            keyword_file_path=self.keyword_file_path,
            model_name=self.model_name,
            model_version=self.model_version)


def start_keyword_extraction_job(
        server_dir: str,
        job_id: str,
        host: str,
        port: str,
        model_name: str,
        model_version: str = None):
    status = get_status(server_dir=server_dir, job_id=job_id)
    if status != JobStatus.not_started:
        message = f"You can only start a job with {JobStatus.not_started} status. current status {status}."
        logger.info(message)
        return False, message, ""

    output_dir = job_utils.get_job_output_dir(
        server_dir, prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id)
    log_path = job_utils.get_log_path(output_dir, JobType.keywords)

    dataset_path = job_utils.get_dataset_file_path(
        server_dir, job_utils.KEYWORD_JOB_PREFIX, job_id)
    keyword_file_path = get_keyword_file_path(server_dir, job_id)

    # Check if dataset is prepared
    if not os.path.isfile(dataset_path):
        return False, f"dataset file {dataset_path} not exists", ""

    init_logger(log_file=log_path)
    logger.info("start running extract_keywords")

    _, _, training_type, _, meta_data_types = poseidon_data.load_poseidon_dataset_file(
        dataset_path)
    if training_type != TrainingType.meta_data:
        return False, f"training_type {training_type} is not supported", ""
    if len(meta_data_types) != 1:
        return False, f"multiple meta data types {meta_data_types} are not supported", ""

    job = KeywordExtractionJob(
        output_dir=output_dir,
        job_id=job_id,
        host=host,
        port=port,
        dataset_path=dataset_path,
        keyword_file_path=keyword_file_path,
        model_name=model_name,
        model_version=model_version
    )
    mp.set_start_method("spawn", force=True)
    job_process = mp.Process(
        name=job_utils.get_job_folder_name_by_id(
            prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id), target=job)
    job_process.daemon = True
    job_process.start()
    message = f'Started keyword extraction job with ID {job_id}'
    logger.info(message)
    return True, "", keyword_file_path


def stop_job(job_id: str) -> Tuple[bool, str]:
    active_jobs = list_keyword_jobs()
    job_name = job_utils.get_job_folder_name_by_id(
        prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    for name, job in active_jobs.items():
        if job_name == name:
            logger.info(f"Found active training job with ID {job_id}")
            job.terminate()
            return True, f"Successfully stopped training job with ID  {job_id}"
    return False, f"Failed to find active training job with ID {job_id}"


def cleanup_artifacts(server_dir: str, job_id: str) -> Tuple[bool, str]:
    if is_job_running(job_id=job_id):
        return False, f"can no delete a running job_id {job_id}."
    return job_utils.cleanup_artifacts(
        server_dir, prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id)


def get_status(server_dir: str, job_id: str) -> JobStatus:
    if is_job_running(job_id=job_id):
        return JobStatus.running
    else:
        output_dir = job_utils.get_job_output_dir(
            server_dir, prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id)
        log_path = job_utils.get_log_path(output_dir, JobType.keywords)
        if os.path.exists(output_dir) and os.path.isfile(log_path):
            # keyword file generation marks job is completed
            keyword_file_path = get_keyword_file_path(server_dir, job_id)
            if os.path.isfile(keyword_file_path):
                return JobStatus.completed
            else:
                return JobStatus.terminated
        else:
            return JobStatus.not_started


def get_job_progress(server_dir: str, job_id: str) -> float:
    return None


def get_keywords(server_dir: str, job_id: str):
    keyword_file_path = get_keyword_file_path(server_dir, job_id)
    if os.path.isfile(keyword_file_path):
        with open(keyword_file_path, encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"keyword_file_path not exists. {keyword_file_path}")


def get_keyword_file_path(server_dir: str, job_id: str) -> str:
    output_dir = job_utils.get_job_output_dir(
        server_dir, prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id)
    return os.path.join(output_dir, KEYWORD_FILE_NAME)


def is_job_running(job_id: str) -> bool:
    job_name = job_utils.get_job_folder_name_by_id(
        prefix=job_utils.KEYWORD_JOB_PREFIX, job_id=job_id)
    return job_name in list_keyword_jobs().keys()


def list_keyword_jobs() -> Dict[str, mp.Process]:
    return job_utils.list_running_jobs(job_utils.KEYWORD_JOB_PREFIX)


def list_all_keyword_jobs(server_dir: str) -> Dict[str, str]:
    return job_utils.list_all_jobs(server_dir, job_utils.KEYWORD_JOB_PREFIX)


def extract_keywords(
        output_dir: str,
        job_id: str,
        host: str,
        port: str,
        dataset_path: str,
        keyword_file_path: str,
        model_name: str,
        model_version: str = None) -> Tuple[bool, Dict, str]:

    if not os.path.isfile(dataset_path):
        return False, f"dataset file {dataset_path} not exists"
    try:
        # load dataset file
        _, _, tags_by_id, data_by_tag_ids, dataset = poseidon_data.load_poseidon_dataset(
            dataset_path)

        # get training type
        _, _, training_type, _, meta_data_types = poseidon_data.load_poseidon_dataset_file(
            dataset_path)

        # outputs
        keywords = []

        json_objs = []
        for tag_id, data_by_col_id in data_by_tag_ids.items():
            logger.info("====")
            logger.info(tag_id)

            for col_id, col_data_entries in data_by_col_id.items():
                logger.info(col_id)
                for data_entry in col_data_entries:
                    logger.info(data_entry)
                    # json.dump(obj, ensure_ascii=False,
                    #     cls=dataset_utils.DataEntryEncoder)

                    request_data = data_entry.meta_data
                    request_data["data"] = data_entry.text
                    line = {
                        "labels": data_entry.label,
                        "sentence": data_entry.text,
                        "meta_data": data_entry.meta_data
                    }
                    example = create_example(
                        id="",
                        line=line
                    )
                    text = get_text_from_example(
                        example,
                        training_type,
                        meta_data_types)
                    pred_result = inference_online(
                        host=host,
                        port=port,
                        data_obj=request_data,
                        model_name=model_name,
                        model_version=model_version,
                    )
                    logger.info(pred_result)

                    pred_online = pred_result["class"]
                    probability = pred_result["probability"]

                    label = data_entry.label[0]
                    label2id = {label: int(i)
                                for i, label in enumerate(pred_result["softmax"].keys())}
                    request_data["target"] = label2id[label]
                    insight = explanations_online(
                        host=host,
                        port=port,
                        data_obj=request_data,
                        model_name=model_name,
                        model_version=model_version,
                    )
                    logger.info(insight)
                    words = insight["words"]
                    attributions = insight["importances"]
                    delta = insight["delta"]

                    positions = list(range(len(words)))
                    combined = list(zip(words, positions, attributions))
                    sorted_attributions = sorted(
                        combined, key=lambda tp: tp[2], reverse=True)

                    obj = {
                        "text": text,
                        "probability": probability,
                        "pred_online": pred_online,
                        "label": label,
                        "attributions_sum": torch.tensor(attributions).sum().item(),
                        "delta": delta,
                        "sorted_attributions": sorted_attributions,
                    }
                    logger.info(obj)
                    json_objs.append(obj)

        # 只有中文模型才使用jieba分词器
        use_jieba = model_name == "bert-base-chinese"
        label_2_keywords = build_keyword_dict(json_objs, use_jieba)
        # output_file = os.path.join(output_dir, "keywords.json")
        with open(keyword_file_path, 'w') as f:
            json.dump(label_2_keywords, f, ensure_ascii=False)

    except ValueError as e:
        return False, e
    return True, keyword_file_path


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


def filter_by_word_type(segment_type: str):
    # paddle模式词性标注对应表如下：

    # paddle模式词性和专名类别标签集合如下表，其中词性标签 24 个（小写字母），专名类别标签 4 个（大写字母）。

    # 标签	 含义	    标签	含义	    标签	含义	    标签	含义
    # n	    普通名词	f	    方位名词	  s	   处所名词     t	    时间
    # nr    人名	   ns	    地名	    nt	  机构名	   nw	  作品名
    # nz	其他专名	v	    普通动词	 vd	   动副词	    vn	    名动词
    # a	    形容词	    ad	    副形词	    an	   名形词	    d	    副词
    # m	    数量词	    q	    量词	    r	   代词	        p	    介词
    # c	    连词	    u	    助词	    xc	   其他虚词	    w	    标点符号
    # PER	人名	    LOC	    地名	    ORG     机构名	    TIME	时间
    return segment_type in [
        # "n", "ns", "s", "t"
        # "nr", "v", "nt", "nw"
        # "nz", "vn"
    ]


def build_keyword_dict(json_objs, use_jieba=True, do_average=False):
    if use_jieba:
        jieba.enable_paddle()

    label_to_keyword_attrs = {}
    label_to_keyword_count = {}
    for obj in json_objs:
        label = obj["label"]
        if label not in label_to_keyword_attrs:
            label_to_keyword_attrs[label] = {}
            label_to_keyword_count[label] = {}

        index = 0
        attributions_sorted_by_index = sorted(
            obj["sorted_attributions"], key=lambda k_v: k_v[1])

        per_label_attributions = label_to_keyword_attrs[label]
        per_label_counts = label_to_keyword_count[label]
        # 词粒度(中文)
        if use_jieba:
            sentence = obj["text"]
            segs = list(pseg.cut(sentence, use_paddle=True))
            # tokens = [entry[0] for entry in attributions_sorted_by_index]
            assert sum([len(seg.word) for seg in segs]) == len(sentence)
            index = 0
            # logger.info(segs)
            for seg in segs:
                # Filter word types that doesn't matter
                if filter_by_word_type(seg.flag):
                    index += len(seg.word)
                    continue
                seg_attribution = 0
                # logger.info(seg)
                # logger.info(attributions_sorted_by_index)
                # logger.info(attributions_sorted_by_index[index])
                for i in range(len(seg.word)):
                    # logger.info(index)
                    seg_attribution += attributions_sorted_by_index[index][2]
                    index += 1
                per_label_attributions[seg.word] = per_label_attributions.get(
                    seg.word, 0) + seg_attribution
                per_label_counts[seg.word] = per_label_counts.get(
                    seg.word, 0) + 1
        else:
            # 字(token)粒度(英文)
            for entry in obj["sorted_attributions"]:
                word, _, attribution = entry
                per_label_attributions[word] = per_label_attributions.get(
                    word, 0) + attribution
                per_label_counts[word] = per_label_counts.get(word, 0) + 1

    label_to_keyword_attrs_sorted = {}
    for label, keywords in label_to_keyword_attrs.items():
        keyword_attributions = keywords.items()
        if do_average:
            keyword_counts = label_to_keyword_count[label]
            averaged_keyword_attributions = []
            for k, v in keyword_attributions:
                if keyword_counts[k] != 1:
                    logger.info(f"label: {label}, k {k}: {keyword_counts[k]}")
                averaged_v = 0 if v == 0 else 1.0 * v / keyword_counts[k]
                averaged_keyword_attributions.append([k, averaged_v])
            keyword_attributions = averaged_keyword_attributions
        keywords_sorted = sorted(
            keyword_attributions, key=lambda k_v: k_v[1], reverse=True)
        label_to_keyword_attrs_sorted[label] = keywords_sorted
    return label_to_keyword_attrs_sorted
