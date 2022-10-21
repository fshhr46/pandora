import os
import jieba
import torch
import json
import logging
import pathlib
import jieba.posseg as pseg

from captum.attr import visualization
from pandora.dataset.sentence_data import Dataset

import pandora.packaging.inference as inference
import pandora.tools.test_utils as test_utils

from pandora.tools.common import logger
from pandora.callback.progressbar import ProgressBar
from pandora.tools.common import init_logger
from pandora.packaging.feature import TrainingType, MetadataType

device = test_utils.get_device()


def test_get_insights(
        lines,
        datasets,
        model_package_dir,
        training_type,
        meta_data_types,
        batch_size=40,
        n_steps=50,
        output_dir=None,
        visualize_output=False):

    _, local_rank, tokenizer, model, processor = test_utils.load_model(
        device,
        datasets,
        model_package_dir,
        training_type,
        meta_data_types,
    )

    _, examples, dataloader, id2label, label2id = test_utils.load_dataset(
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
            mode=test_utils.HANDLER_MODE,
            model=model)
        pred_results = inference.format_outputs(
            inferences=inferences, id2label=id2label)

        sub_examples = examples[step * batch_size: (step+1) * batch_size]
        assert len(pred_results) == len(sub_examples)

        targets = [label2id[res["class"]] for res in pred_results]

        # TODO: Fix hard coded "sequence_classification"
        insights = inference.run_get_insights(
            # configs
            mode=test_utils.HANDLER_MODE,
            embedding_name="bert",
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
        for pred_result, insight, example in zip(pred_results, insights, sub_examples):

            # Read baseline file data
            label = example.labels[0]
            text = example.text

            pred_online = pred_result["class"]
            probability = pred_result["probability"]

            logger.info("")
            logger.info(
                "======================================================")
            # logger.info(f"pred_online: {pred_online}")
            # logger.info(f"label: {label}")

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
            json_objs.append(obj)
            import time
            # time.sleep(1)
            logger.info(json.dumps(obj,
                                   ensure_ascii=False,
                                   indent=4))
            pbar(total)
            total += 1

    label_2_keywords = build_keyword_dict(json_objs)
    if output_dir:
        with open(os.path.join(output_dir, "attributions.json"), 'w') as f:
            for json_obj in json_objs:
                json.dump(json_obj, f, ensure_ascii=False)
                f.write("\n")
        with open(os.path.join(output_dir, "keywords.json"), 'w') as f:
            json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)
    if visualize_output:
        return visualize_insights(json_objs=json_objs)


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
        # jieba.enable_parallel(4)

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
        # 词粒度
        if use_jieba:
            sentence = obj["text"][:test_utils.MAX_SEQ_LENGTH - 2]
            segs = list(pseg.cut(sentence, use_paddle=True))
            # tokens = [entry[0] for entry in attributions_sorted_by_index]
            assert sum([len(seg.word) for seg in segs]) == len(sentence)
            index = 0
            for seg in segs:
                # Filter word types that doesn't matter
                if filter_by_word_type(seg.flag):
                    index += len(seg.word)
                    continue
                seg_attribution = 0
                # logger.info(seg)
                # logger.info(attributions_sorted_by_index[index])
                for i in range(len(seg.word)):
                    seg_attribution += attributions_sorted_by_index[index][2]
                    index += 1
                per_label_attributions[seg.word] = per_label_attributions.get(
                    seg.word, 0) + seg_attribution
                per_label_counts[seg.word] = per_label_counts.get(
                    seg.word, 0) + 1
        else:
            # 字粒度
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


def visualize_insights(json_objs):
    vis_data_records_ig = []
    for obj in json_objs:
        sorted_attributions = obj["sorted_attributions"]
        sorted_attributions = sorted(
            sorted_attributions, key=lambda tp: tp[1])
        attributions = torch.tensor(
            [item[2] for item in sorted_attributions], dtype=torch.float32)
        sentence = obj["text"][:test_utils.MAX_SEQ_LENGTH-2]
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
    base_folder = "/home/haoranhuang/workspace/resource/outputs/bert-base-chinese/pandora_demo_1019_fix_10_10_meta_comment/"
    data_file = "/home/haoranhuang/workspace/resource/datasets/pandora_demo_1019_fix_10_10/test.json"
    model_package_dir = os.path.join(base_folder, "torchserve_package")

    lines = open(data_file).readlines()
    attribution_output_dir = os.path.join(base_folder, "attribution")
    pathlib.Path(attribution_output_dir).mkdir(parents=True, exist_ok=True)

    training_type = TrainingType.meta_data
    meta_data_types = [
        MetadataType.column_comment
    ]

    datasets = [
        # Dataset.short_sentence
        "pandora_demo_1019_fix_10_10"
    ]

    # Run get insights
    test_get_insights(
        lines=lines,
        datasets=datasets,
        model_package_dir=model_package_dir,
        training_type=training_type,
        meta_data_types=meta_data_types,
        batch_size=1,
        n_steps=50,
        output_dir=attribution_output_dir,
        visualize_output=False)

    # Test merge attributions
    attribution_file = os.path.join(
        attribution_output_dir, "attributions.json")
    lines = open(attribution_file).readlines()
    json_objs = [json.loads(line) for line in lines]

    # with open(f"{home}/workspace/resource/attribution/keywords_char_averaged.json", 'w') as f:
    #     label_2_keywords = build_keyword_dict(
    #         json_objs, use_jieba=False, do_average=True)
    #     json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)
    # with open(f"{home}/workspace/resource/attribution/keywords_word_averaged.json", 'w') as f:
    #     label_2_keywords = build_keyword_dict(
    #         json_objs, use_jieba=True, do_average=True)
    #     json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)

    # with open(os.path.join(attribution_output_dir, f"keywords_char.json"), 'w') as f:
    #     label_2_keywords = build_keyword_dict(
    #         json_objs, use_jieba=False, do_average=False)
    #     json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)
    # with open(os.path.join(attribution_output_dir, f"keywords_word.json"), 'w') as f:
    #     label_2_keywords = build_keyword_dict(
    #         json_objs, use_jieba=True, do_average=False)
    #     json.dump(label_2_keywords, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    run_test()
