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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pandora.callback.lr_scheduler import get_linear_schedule_with_warmup
from pandora.callback.optimizater.adamw import AdamW
from pandora.callback.progressbar import ProgressBar
from pandora.models.transformers import WEIGHTS_NAME, BertConfig
from pandora.processors.feature import (
    batch_collate_fn,
)


from pandora.packaging.model import BertForSentence
from pandora.processors.feature import cls_processors
from pandora.dataset import sentence_data
from pandora.packaging.tokenizer import SentenceTokenizer
from pandora.tools.common import init_logger, logger
import pandora.tools.common as common_utils
import pandora.tools.runner_utils as runner_utils
import pandora.tools.mps_utils as mps_utils
import pandora.dataset.dataset_utils as dataset_utils


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                 for conf in (BertConfig,)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSentence, SentenceTokenizer),
}


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
    bert_base_model_name,
    datasets,
) -> List[str]:
    if mps_utils.has_mps:
        assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    datasets_str = ", ".join(datasets)
    logger.info(f"datasets are:\n\t {datasets_str}")
    dataset_names = "_".join(datasets)
    output_dir = os.path.join(resource_dir, "outputs",
                              bert_base_model_name, dataset_names)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = dataset_utils.get_partitioned_data_folder(resource_dir)
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


def train_eval_test(arg_list, resource_dir: str, datasets: List[str]):
    parser = get_args_parser()
    args = parser.parse_args(arg_list)

    processor = runner_utils.get_data_processor(
        resource_dir=resource_dir, datasets=datasets)

    args, config, tokenizer, model, model_classes = setup(
        args=args, processor=processor)
    config_class, model_class, tokenizer_class = model_classes

    data = runner_utils.prepare_data(args, tokenizer, processor=processor)
    dataset_partitions = data["datasets"]
    train_dataset, eval_dataset, test_dataset = \
        dataset_partitions["train"], dataset_partitions["eval"], dataset_partitions["test"]
    examples = data["examples"]
    # Training
    if train_dataset:
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
        # save as binary
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # save as json
        with open(os.path.join(args.output_dir, "index_to_name.json"), "w") as id2label_f:
            json.dump(args.id2label, id2label_f, indent=4, ensure_ascii=False)

    # Evaluation
    results = {}
    if eval_dataset and args.local_rank in [-1, 0]:
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
            result = evaluate(args, model, eval_dataset, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(
                    global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    # Predict
    if test_dataset and args.local_rank in [-1, 0]:
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

            report_dir = os.path.join(args.output_dir, prefix)
            if not os.path.exists(report_dir) and args.local_rank in [-1, 0]:
                os.makedirs(report_dir)
            predictions = predict(args, model, test_dataset, prefix=prefix)
            test_examples = examples["test"]
            runner_utils.build_train_report(
                test_examples, predictions, report_dir,
                processor=processor)


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
    common_utils.seed_everything(args.seed)
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


def evaluate(args, model, eval_dataset, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=batch_collate_fn)
    # Eval!
    logger.info(f"***** Running evaluation %s ***** prefix: {prefix}")
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
    logger.info(f"***** Eval results %s ***** prefix: {prefix}")
    info = "-".join([f' {key}: {value:.4f} ' for key,
                    value in results.items()])
    logger.info(info)
    logger.info(f"***** Entity results %s ***** prefix: {prefix}")
    return results


def predict(args, model, test_dataset, prefix=""):
    batch_size = 1
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(
        test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=batch_collate_fn)
    # Predict!
    logger.info(f"***** Running prediction %s ***** prefix: {prefix}")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", batch_size)
    assert batch_size == 1

    predictions = []
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
        predictions.append(json_d)
        pbar(step)
    return predictions


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(cls_processors.keys()))
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


def setup(args, processor):
    # check if output dir already exists and override is not set
    log_path = runner_utils.get_training_log_path(args.output_dir)
    if os.path.exists(log_path) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    init_logger(log_file=runner_utils.get_training_log_path(args.output_dir))
    # create output dirs
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        elif mps_utils.has_mps:
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
    common_utils.seed_everything(args.seed)

    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in cls_processors:
        raise ValueError("Task not found: %s" % (args.task_name))

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
