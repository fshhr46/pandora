#!/usr/bin/python
# coding:utf8

import json
import os
from pandora.tools.common import logger


def build_report_sklearn(pre_lines, truth_lines, label_list, report_dir=None):
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
    log_result(all_stats=all_stats, summary=summary, report_dir=report_dir)


def _get_empty_stats():
    return {
        "f1":  None,
        "precision": None,
        "recall": None,
        "acc": None,
        "TP": 0,
        "FP": 0,
        "TN": 0,
        "FN": 0,
        "counts_t": 0,
        "counts_p": 0,
    }


def get_f1_score_label(pre_lines, truth_lines, label):
    """
    打分函数
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    counts_t = 0
    counts_p = 0
    for pre_line, truth_line in zip(pre_lines, truth_lines):
        preds = pre_line["pred"]
        preds.sort()
        truths = truth_line["label"]
        truths.sort()
        # DOC Hack: when calculating F-1 for rejected class
        # label here would be set to None
        if label is None:
            # Calculating for unseen class
            # For DOC: determine
            if pre_line['rejected']:
                preds = [None]
            if pre_line['unseen_class']:
                truths = [None]

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

    stats = _get_empty_stats()
    stats.update({
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
    })
    return stats


def build_stats(
        pre_lines,
        truth_lines,
        label_list):
    num_preds = len(pre_lines)
    all_stats = {}
    summary = _get_empty_stats()
    sum_f1 = 0
    sum_p = 0
    sum_r = 0

    for label in label_list:
        stats = get_f1_score_label(
            pre_lines, truth_lines, label=label)
        all_stats[label] = stats

        # Add data to summary
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
    # summary = sort_stats(summary)
    return all_stats, summary


def build_report(
        pre_lines,
        truth_lines,
        label_list,
        build_doc_report: bool = False,
        report_dir=None):

    # TODO: Hack, calculate score for DOC using label = None
    if build_doc_report:
        doc_stats = get_f1_score_label(
            pre_lines, truth_lines, label=None)
    else:
        doc_stats = None

    # Build report for the main task
    all_stats, summary = build_stats(
        pre_lines, truth_lines, label_list)

    log_result(all_stats=all_stats, summary=summary,
               doc_stats=doc_stats, report_dir=report_dir)


def sort_stats(stats):
    return dict(sorted(stats.items(), key=lambda item: item[1], reverse=True))


def log_result(all_stats, summary, doc_stats=None, report_dir=None):
    logger.info("========== Summary ==========")
    all_stats_json_str = json.dumps(all_stats, indent=4, ensure_ascii=False)
    logger.info(f"\n stats: \n{all_stats_json_str}")

    summary_json_str = json.dumps(summary, indent=4, ensure_ascii=False)
    logger.info(
        f"\nsummary stats: \n{summary_json_str}")

    doc_stats_json_str = None
    if doc_stats:
        doc_stats_json_str = json.dumps(
            doc_stats, indent=4, ensure_ascii=False)
        logger.info(
            f"\nDOC stats: \n{doc_stats_json_str}")

    if report_dir:
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, "w") as f:
            f.write(all_stats_json_str)
        report_all_path = os.path.join(report_dir, "report_all.json")
        with open(report_all_path, "w") as f:
            f.write(summary_json_str)

        if doc_stats_json_str:
            report_doc_path = os.path.join(report_dir, "report_doc.json")
            with open(report_doc_path, "w") as f:
                f.write(doc_stats_json_str)

        logger.info(f"report was saved in dir: {report_dir}")
