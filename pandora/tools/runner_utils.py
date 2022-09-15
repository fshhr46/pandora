import os
import json
import pandora.tools.report as report
from pandora.tools.common import json_to_text


from pandora.processors.feature import SentenceProcessor
from pandora.processors.feature import convert_examples_to_features
from pandora.processors.feature import (
    convert_examples_to_features,
    RandomDataSampler,
)

import torch
from torch.utils.data import TensorDataset

from pandora.tools.common import logger


def build_train_report(examples, predictions, report_dir, processor):
    logger.info(" ")
    output_predic_file = os.path.join(report_dir, "test_prediction.json")
    output_submit_file = os.path.join(report_dir, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in predictions:
            writer.write(json.dumps(record) + '\n')

    test_submit = []
    for x, y in zip(examples, predictions):
        json_d = {}
        json_d['guid'] = x.guid
        json_d['text'] = x.sentence
        json_d['words'] = x.words
        json_d['label'] = x.labels
        json_d['pred'] = y['tags_sentence']
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

    pre_lines = [json.loads(line.strip())
                 for line in open(output_submit_file) if line.strip()]
    label_list = processor.get_labels()
    report.build_report(pre_lines=pre_lines,
                        truth_lines=pre_lines,
                        label_list=label_list,
                        report_dir=report_dir)


def get_data_processor(datasets, resource_dir: str):
    return SentenceProcessor(resource_dir=resource_dir, datasets=datasets)


def prepare_data(args, tokenizer, processor):
    train_dataset = eval_dataset = test_dataset = None
    train_examples = eval_examples = test_examples = None
    if args.do_train:
        sampler = None
        if args.sample_size:
            sampler = RandomDataSampler(
                seed=args.seed,
                sample_size=args.sample_size)
        train_dataset, train_examples = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type='train',  evaluate=False, processor=processor, sampler=sampler)
    if args.do_eval:
        eval_dataset, eval_examples = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type='dev', evaluate=True, processor=processor)

    if args.do_predict:
        test_dataset, test_examples = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type="test", evaluate=False, processor=processor)
    return {
        "datasets": {
            "train": train_dataset,
            "eval": eval_dataset,
            "test": test_dataset,
        },
        "examples": {
            "train": train_examples,
            "eval": eval_examples,
            "test": test_examples,
        },
    }


def load_examples(data_dir, processor, data_type):
    if data_type == 'train':
        examples = processor.get_train_examples(data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(data_dir)
    elif data_type == 'test':
        examples = processor.get_test_examples(data_dir)
    else:
        raise ValueError(f"invalid data_type {data_type}")
    return examples


def load_and_cache_examples(args, task, tokenizer, data_type, evaluate: bool, processor, sampler=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # Load data features from cache or dataset file
    feature_dim = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    base_model_name = list(
        filter(None, args.model_name_or_path.split('/'))).pop()
    cached_features_file = os.path.join(
        args.output_dir, f'cached_softmax-{data_type}_{base_model_name}_{feature_dim}_{task}')
    examples = load_examples(args.data_dir, processor, data_type)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if sampler:
            examples = sampler.sample(examples=examples)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length,
                                                cls_token_at_end=False,
                                                pad_on_left=False,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in [
                                                    "xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids(
                                                    [tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    # Only support single label now.
    all_label_ids = torch.tensor(
        [f.sentence_labels[0] for f in features], dtype=torch.long)

    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset, examples


def get_training_log_path(output_dir):
    return os.path.join(output_dir, 'training_job.log')
