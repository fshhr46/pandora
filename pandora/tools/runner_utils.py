import os
import json
from typing import List
from pandora.packaging.feature import (
    MetadataType,
    TrainingType,
    convert_features_to_dataset
)

from pandora.packaging.constants import CHARBERT_CHAR_VOCAB
import pandora.tools.report as report
from pandora.tools.common import json_to_text


from pandora.packaging.model import BertBaseModelType
from pandora.packaging.feature import (
    convert_example_to_feature,
    get_text_from_example,
    load_char_vocab,
    RandomDataSampler,
    SentenceProcessor,
)

import torch

from pandora.tools.common import logger


def build_train_report(
        examples,
        training_type,
        meta_data_types,
        predictions,
        report_dir,
        processor):
    logger.info(" ")
    output_predic_file = os.path.join(report_dir, "test_prediction.json")
    output_submit_file = os.path.join(report_dir, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in predictions:
            writer.write(json.dumps(record) + '\n')

    test_submit = []
    for example, pred in zip(examples, predictions):
        text = get_text_from_example(
            example=example,
            training_type=training_type,
            meta_data_types=meta_data_types,
        )
        json_d = {}
        json_d['guid'] = example.id
        json_d['text'] = text
        json_d['label'] = example.labels
        json_d['pred'] = pred['tags']
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

    pre_lines = [json.loads(line.strip())
                 for line in open(output_submit_file) if line.strip()]
    label_list = processor.get_labels()
    report.build_report(pre_lines=pre_lines,
                        truth_lines=pre_lines,
                        label_list=label_list,
                        report_dir=report_dir)


def get_data_processor(
        datasets,
        training_type: TrainingType,
        meta_data_types: List[MetadataType],
        resource_dir: str):
    return SentenceProcessor(
        training_type=training_type,
        meta_data_types=meta_data_types,
        resource_dir=resource_dir,
        datasets=datasets)


def prepare_data(args,
                 tokenizer,
                 processor,
                 bert_model_type: BertBaseModelType):
    train_dataset = eval_dataset = test_dataset = None
    train_examples = eval_examples = test_examples = None

    # char bert setup
    if bert_model_type == BertBaseModelType.char_bert:
        char2ids_dict = load_char_vocab(CHARBERT_CHAR_VOCAB)
    else:
        char2ids_dict = None

    if args.do_train:
        sampler = None
        if args.sample_size:
            sampler = RandomDataSampler(
                seed=args.seed,
                sample_size=args.sample_size)
        train_dataset, train_examples = load_and_cache_examples(
            args, tokenizer, data_type='train',  evaluate=False, processor=processor, char2ids_dict=char2ids_dict, sampler=sampler)
    if args.do_eval:
        eval_dataset, eval_examples = load_and_cache_examples(
            args, tokenizer, data_type='dev', evaluate=True, processor=processor, char2ids_dict=char2ids_dict)

    if args.do_predict:
        test_dataset, test_examples = load_and_cache_examples(
            args, tokenizer, data_type="test", evaluate=True, processor=processor, char2ids_dict=char2ids_dict)
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


def load_and_cache_examples(args,
                            tokenizer,
                            data_type,
                            evaluate: bool,
                            processor,
                            char2ids_dict,
                            sampler=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # Load data features from cache or dataset file
    feature_dim = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    base_model_name = list(
        filter(None, args.model_name_or_path.split('/'))).pop()
    cached_features_file = os.path.join(
        args.output_dir, f'cached_softmax-{data_type}_{base_model_name}_{feature_dim}')
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
        label2id = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d",
                            ex_index, len(examples))
            feat = convert_example_to_feature(
                example,
                training_type=processor.training_type,
                meta_data_types=processor.meta_data_types,
                label2id=label2id,
                log_data=ex_index < 5,
                max_seq_length=args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length,
                tokenizer=tokenizer,
                # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids(
                    [tokenizer.pad_token])[0],
                pad_token_segment_id=0,
                char2ids_dict=char2ids_dict
            )
            features.append(feat)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)
    include_char_data = char2ids_dict is not None
    dataset = convert_features_to_dataset(
        args.local_rank, features, evaluate, include_char_data)
    return dataset, examples
