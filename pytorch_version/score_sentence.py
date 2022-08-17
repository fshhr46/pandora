#!/usr/bin/python
# coding:utf8

import json
import os
from pathlib import Path
from processors.cls_sentence import SentenceProcessor
from tools.common import init_logger, logger
from tools.sentence_data import Dataset


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

    build_report(predict_path=predict_path,
                 truth_path=truth_path,
                 report_dir=predict_dir,
                 datasets=datasets)


def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    counts_t = 0
    counts_p = 0
    for pre_line, gold_line in zip(pre_lines, gold_lines):
        preds = pre_line["pred"]
        truths = gold_line["label"]
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


def build_report(predict_path, truth_path, report_dir, datasets=None):
    pre_lines = [json.loads(line.strip())
                 for line in open(predict_path) if line.strip()]
    gold_lines = [json.loads(line.strip())
                  for line in open(truth_path) if line.strip()]
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
    labels_to_check = processor.get_labels()
    sum_f1 = 0
    sum_acc = 0
    sum_p = 0
    sum_r = 0
    for label in labels_to_check:
        stats = get_f1_score_label(
            pre_lines, gold_lines, label=label)
        all_stats[label] = stats
        sum_f1 += stats["f1"]
        sum_acc += stats["acc"]
        sum_p += stats["precision"]
        sum_r += stats["recall"]

        summary["TP"] += stats["TP"]
        summary["FP"] += stats["FP"]
        summary["TN"] += stats["TN"]
        summary["FN"] += stats["FN"]
        summary["counts_t"] += stats["counts_t"]
        summary["counts_p"] += stats["counts_p"]
    summary["f1"] = sum_f1 / len(labels_to_check)
    summary["acc"] = sum_acc / len(labels_to_check)
    summary["precision"] = sum_p / len(labels_to_check)
    summary["recall"] = sum_r / len(labels_to_check)

    logger.info("========== Summary ==========")
    json_str = json.dumps(all_stats, indent=4, ensure_ascii=False)
    logger.info(f"\n stats: \n{json_str}")
    logger.info(
        f"\nsummary stats: \n{json.dumps(summary, indent=4, ensure_ascii=False)}")

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
