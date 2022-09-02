#!/usr/bin/python
# coding:utf8

import json
import os
from pathlib import Path
from pandora.processors.feature import SentenceProcessor
from pandora.tools.common import init_logger, logger
from pandora.dataset.sentence_data import Dataset


def main():
    resource_dir = os.path.join(Path.home(), "workspace", "resource")
    datasets = [
        Dataset.column_data,
        Dataset.short_sentence,
        Dataset.long_sentence,
    ]
    datasets.sort()
    dataset_names = "_".join(datasets)
    predict_dir = os.path.join(resource_dir,
                               f"outputs/bert-base-chinese/sentence/{dataset_names}/bert/predict/softmax/")
    predict_path = os.path.join(predict_dir, "test_submit.json")
    # truth_path = os.path.join(resource_dir, "CLUEdatasets/cluener/dev.json")
    truth_path = predict_path

    pre_lines = [json.loads(line.strip())
                 for line in open(predict_path) if line.strip()]
    truth_lines = [json.loads(line.strip())
                   for line in open(truth_path) if line.strip()]
    # validation
    assert len(pre_lines) == len(truth_lines)

    build_report(pre_lines=pre_lines,
                 truth_lines=truth_lines,
                 report_dir=predict_dir,
                 datasets=datasets)

    build_report_sklearn(pre_lines=pre_lines,
                         truth_lines=truth_lines,
                         report_dir=None,
                         datasets=datasets)


def build_report_sklearn(pre_lines, truth_lines, report_dir=None, datasets=None):

    processor = SentenceProcessor(datasets_to_include=datasets)
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    all_preds = [label2id[line["pred"][0]] for line in pre_lines]
    all_truths = [label2id[line["label"][0]] for line in truth_lines]
    from sklearn.metrics import f1_score
    f = f1_score(all_truths, all_preds, average='macro')
    from sklearn.metrics import classification_report
    summary = classification_report(all_truths, all_preds, output_dict=True)
    all_stats = {}
    for label, id in label2id.items():
        data = summary.pop(str(id))
        all_stats[label] = data
    print_result(all_stats=all_stats, summary=summary, report_dir=report_dir)


def get_f1_score_label(pre_lines, truth_lines, label="organization"):
    """
    打分函数
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    counts_t = 0
    counts_p = 0
    for pre_line, gold_line in zip(pre_lines, truth_lines):
        preds = pre_line["pred"]
        preds.sort()
        truths = gold_line["label"]
        truths.sort()
        if label in truths and label in preds:
            TP += 1
            counts_t += 1
            counts_p += 1
        elif label not in truths and label in preds:
            FP += 1
            counts_p += 1
        elif label in truths and label not in preds:
            FN += 1
            counts_t += 1
        else:
            TN += 1

    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    all_preds = TP + FP + TN + FN
    acc = (TP + TN) / all_preds if all_preds > 0 else 0
    stats = {
        "label": label,
        "f1":  f,
        "precision": p,
        "recall": r,
        "acc": acc,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "counts_t": counts_t,
        "counts_p": counts_p,
    }
    return stats


def build_report(pre_lines, truth_lines, report_dir=None, datasets=None):
    num_preds = len(pre_lines)
    all_stats = {}
    summary = {
        "label": "all",
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "counts_t": 0,
        "counts_p": 0,
    }
    processor = SentenceProcessor(datasets_to_include=datasets)
    label_list = processor.get_labels()
    sum_f1 = 0
    sum_p = 0
    sum_r = 0
    for label in label_list:
        stats = get_f1_score_label(
            pre_lines, truth_lines, label=label)
        all_stats[label] = stats
        sum_f1 += stats["f1"]
        sum_p += stats["precision"]
        sum_r += stats["recall"]

        summary["TP"] += stats["TP"]
        summary["FP"] += stats["FP"]
        summary["TN"] += stats["TN"]
        summary["FN"] += stats["FN"]
        summary["counts_t"] += stats["counts_t"]
        summary["counts_p"] += stats["counts_p"]
    summary["f1"] = sum_f1 / len(label_list)
    summary["acc"] = summary["TP"] / num_preds if num_preds > 0 else 0
    summary["precision"] = sum_p / len(label_list)
    summary["recall"] = sum_r / len(label_list)
    print_result(all_stats=all_stats, summary=summary, report_dir=report_dir)


def print_result(all_stats, summary, report_dir=None):
    logger.info("========== Summary ==========")
    json_str = json.dumps(all_stats, indent=4, ensure_ascii=False)
    logger.info(f"\n stats: \n{json_str}")
    logger.info(
        f"\nsummary stats: \n{json.dumps(summary, indent=4, ensure_ascii=False)}")

    if report_dir:
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, "w") as report_f:
            json.dump(all_stats, report_f, indent=4, ensure_ascii=False)
        report_all_path = os.path.join(report_dir, "report_all.json")
        with open(report_all_path, "w") as report_f:
            json.dump(summary, report_f, indent=4, ensure_ascii=False)
        logger.info(f"report was saved in dir: {report_dir}")


if __name__ == "__main__":
    init_logger()
    main()
