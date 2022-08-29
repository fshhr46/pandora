from ast import arg
import errno
from pathlib import Path
import argparse
import glob
import json
import logging
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.optimizater.adamw import AdamW
from callback.progressbar import ProgressBar
from models.bert_for_sentence import BertForSentence
from models.transformers import WEIGHTS_NAME, BertConfig
from processors.cls_sentence import (
    batch_collate_fn,
    convert_examples_to_features,
    RandomDataSampler,
)
from processors.cls_sentence import convert_examples_to_features
from processors.cls_sentence import cls_processors as processors
from processors.utils_cls import SentenceTokenizer
from tools.common import init_logger, logger
from tools.common import seed_everything, json_to_text
import score_sentence
from tools.sentence_data import Dataset

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                 for conf in (BertConfig,)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSentence, SentenceTokenizer),
}


DATASETS_TO_INCLUDE = list(Dataset)
DATASETS_TO_INCLUDE = [
    Dataset.column_data,
    # Dataset.short_sentence,
    # Dataset.long_sentence,
]
DATASETS_TO_INCLUDE.sort()


def get_training_args(
    task_name,
    mode_type,
    bert_base_model_name,
    # optional parameters
    sample_size: int = 0,
) -> List[str]:
    batch_size = 24
    checkpoint_steps = 500
    arg_list = [f"--model_type={mode_type}",
                f"--model_name_or_path={bert_base_model_name}",
                f"--task_name={task_name}",
                "--do_lower_case",
                "--loss_type=ce",
                "--train_max_seq_length=128",
                "--eval_max_seq_length=512",
                f"--per_gpu_train_batch_size={batch_size}",
                f"--per_gpu_eval_batch_size={batch_size}",
                "--learning_rate=3e-5",
                "--num_train_epochs=4.0",
                f"--logging_steps={checkpoint_steps}",
                f"--save_steps={checkpoint_steps}",
                "--seed=42",
                "--eval_all_checkpoints",
                "--overwrite_cache",
                # "--overwrite_output_dir",
                ]

    if sample_size:
        arg_list.append(
            f"--sample_size={sample_size}",
        )
    return arg_list


def set_actions(
    do_train: bool,
    do_eval: bool,
    do_predict: bool,
):
    args = []
    if do_train:
        args.append("--do_train")
    if do_eval:
        args.append("--do_eval")
    if do_predict:
        args.append("--do_predict")
    return args


def get_default_dirs(
    resource_dir,
    cache_dir,
    task_name,
    bert_base_model_name,
) -> List[str]:
    if torch.has_mps:
        assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    datasets_str = ", ".join(DATASETS_TO_INCLUDE)
    logger.info(f"DATASETS_TO_INCLUDE are:\n\t {datasets_str}")
    dataset_names = "_".join(DATASETS_TO_INCLUDE)
    output_dir = os.path.join(resource_dir, "outputs",
                              bert_base_model_name, task_name, dataset_names)

    data_dir = os.path.join(resource_dir, "datasets", task_name)
    os.makedirs(os.path.join(
        resource_dir, "prev_trained_model"), exist_ok=True)
    pre_trained_model_dir = os.path.join(resource_dir, "prev_trained_model")
    model_cache_dir = os.path.join(pre_trained_model_dir, "transformers")
    try:
        os.symlink(cache_dir, model_cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(model_cache_dir)
            os.symlink(cache_dir, model_cache_dir)
        else:
            raise e

    args = [f"--data_dir={data_dir}",
            f"--output_dir={output_dir}/",
            f"--cache_dir={cache_dir}", ]
    return args


def main():

    home = str(Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    task_name = "sentence"
    mode_type = "bert"
    bert_base_model_name = "bert-base-chinese"
    arg_list = get_training_args(
        task_name=task_name,
        mode_type=mode_type,
        bert_base_model_name=bert_base_model_name,
    )

    arg_list.extend(
        get_default_dirs(
            resource_dir,
            cache_dir,
            task_name=task_name,
            bert_base_model_name=bert_base_model_name,
        ))
    arg_list.extend(
        set_actions(
            do_train=True,
            do_eval=True,
            do_predict=True,
        ))
    train_eval_test(arg_list)


def get_data_processor(task_name):
    return processors[task_name](datasets_to_include=DATASETS_TO_INCLUDE)


def train_eval_test(arg_list):

    parser = get_args_parser()
    args = parser.parse_args(arg_list)
    args, config, tokenizer, model, model_classes = setup(args=args)
    init_logger(log_file=args.output_dir +
                '/{}-{}.log'.format(args.model_type, args.task_name))
    config_class, model_class, tokenizer_class = model_classes

    # Training
    if args.do_train:

        sampler = None
        if args.sample_size:
            sampler = RandomDataSampler(args.sample_size)
        train_dataset = load_and_cache_examples(
            args, args.task_name, tokenizer, data_type='train', sampler=sampler)
        global_step, tr_loss = train(
            args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # local rank = 0 means you are the master node
    # local rank = -1 means distributed training is disabled.
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                '/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(
                    global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split(
                '-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split(
                '/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            prefix = os.path.join("predict", prefix)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            os.makedirs(os.path.join(args.output_dir, prefix), exist_ok=True)
            predict(args, model, tokenizer, prefix=prefix)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=batch_collate_fn)

    if args.max_steps > 0:
        training_steps_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        training_steps_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=training_steps_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com\nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size()
                   if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", training_steps_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # Added here for reproductibility (even between python 2 and 3)
    seed_everything(args.seed)
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            token_type_ids = None
            # XLM and RoBERTa don"t use segment_ids
            if args.model_type != "distilbert" and args.model_type in [
                    "bert", "xlnet"]:
                token_type_ids = batch[2]
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": token_type_ids,
                      "labels": batch[3]
                      }
            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logger.info(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        evaluate(args, model, tokenizer)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = (model.module if hasattr(
                        model, "module") else model)
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir)
        logger.info(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=batch_collate_fn)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            token_type_ids = None
            # XLM and RoBERTa don"t use segment_ids
            if args.model_type != "distilbert" and args.model_type in [
                    "bert", "xlnet"]:
                token_type_ids = batch[2]
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": token_type_ids,
                      "labels": batch[3]
                      }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=1).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        pbar(step)
    logger.info(' ')
    eval_loss = eval_loss / nb_eval_steps
    results = {}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key,
                    value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    return results


def predict(args, model, tokenizer, prefix="", data_type="test"):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    batch_size = 1
    test_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, data_type=data_type)
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(
        test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=batch_collate_fn)
    # Predict!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)
    assert batch_size == 1

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            token_type_ids = None
            # XLM and RoBERTa don"t use segment_ids
            if args.model_type != "distilbert" and args.model_type in [
                    "bert", "xlnet"]:
                token_type_ids = batch[2]
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "token_type_ids": token_type_ids,
                      "labels": None
                      }
            outputs = model(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1).tolist()

        tags = [args.id2label[x] for x in preds]
        json_d = {}
        json_d['tags_sentence'] = tags
        results.append(json_d)
        pbar(step)
    logger.info(" ")
    report_dir = os.path.join(pred_output_dir, prefix)
    output_predic_file = os.path.join(report_dir, "test_prediction.json")
    output_submit_file = os.path.join(report_dir, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

    test_submit = []
    examples = load_examples(
        args.data_dir, get_data_processor(args.task_name), data_type)
    for x, y in zip(examples, results):
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
    score_sentence.build_report(pre_lines=pre_lines,
                                truth_lines=pre_lines,
                                report_dir=report_dir,
                                datasets=DATASETS_TO_INCLUDE)


def load_examples(data_dir, processor, data_type):
    if data_type == 'train':
        examples = processor.get_train_examples(data_dir)
    elif data_type == 'dev':
        examples = processor.get_dev_examples(data_dir)
    else:
        examples = processor.get_test_examples(data_dir)
    return examples


def load_and_cache_examples(args, task, tokenizer, data_type, sampler=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    processor = get_data_processor(task_name=task)
    # Load data features from cache or dataset file
    feature_dim = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    base_model_name = list(
        filter(None, args.model_name_or_path.split('/'))).pop()
    cached_features_file = os.path.join(
        args.output_dir, f'cached_softmax-{data_type}_{base_model_name}_{feature_dim}_{task}')
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = load_examples(args.data_dir, processor, data_type)
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
    return dataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.", )
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()), )
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS), )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    # Other parameters
    parser.add_argument('--markup', default='bios',
                        type=str, choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce',
                        type=str, choices=['lsr', 'focal', 'ce'])
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.", )
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8,
                        type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int,
                        default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="number of samples for each class")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https:/\nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port", type=str,
                        default="", help="For distant debugging.")
    return parser


def setup(args):
    # create output dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # TODO: Setup M1 chip
        if torch.cuda.is_available() and not args.no_cuda:
            device_name = "cuda"
        elif torch.has_mps:
            device_name = "mps"
        else:
            device_name = "cpu"
        logger.info(f"device_name is {device_name}")
        device = torch.device(device_name)
        # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )

    # Set seed
    seed_everything(args.seed)

    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = get_data_processor(task_name=args.task_name)
    logger.info(f"task_name is {args.task_name}")
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    logger.info(f"model_type is {args.model_type}")

    # Load and initialize config, tokenizer and the seed model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # 'sentence': (BertConfig, BertForSentence, CNerTokenizer),
    #  {'num_labels': 34, 'loss_type': 'ce', 'cache_dir': '/Users/haoranhuang/.cache/torch/transformers'}
    # args: removing "cache_dir" and "num_labels", loss_type is left
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, loss_type=args.loss_type,
                                          cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None, )
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info(
        f"\n========\nTraining/evaluation parameters {args}\n========\n")
    return args, config, tokenizer, model, MODEL_CLASSES[args.model_type]


if __name__ == "__main__":
    main()
