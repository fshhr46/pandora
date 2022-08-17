#!python
import errno
import glob
import os

import torch

from models.transformers import WEIGHTS_NAME
from tools.common import logging, logger

from pathlib import Path


def get_runner(predict_type):
    if predict_type == "crf":
        import run_ner_crf as runner
    elif predict_type == "softmax":
        import run_ner_softmax as runner
    elif predict_type == "span":
        import run_ner_span as runner
    else:
        raise ValueError("Invalid value type: %s" % (predict_type))
    return runner


def setup_running_env():
    if torch.has_mps:
        assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"

    bert_base_model_name = "bert-base-chinese"
    task_name = "cluener"
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "outputs",
                              bert_base_model_name, f"{task_name}_output")
    home = str(Path.home())
    cache_dir = os.path.join(home, ".cache/torch/transformers")
    data_dir = os.path.join(current_dir, "CLUEdatasets", task_name)
    os.makedirs(os.path.join(current_dir, "prev_trained_model"), exist_ok=True)
    pre_trained_model_dir = os.path.join(current_dir, "prev_trained_model")
    model_cache_dir = os.path.join(pre_trained_model_dir, "transformers")
    try:
        os.symlink(cache_dir, model_cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(model_cache_dir)
            os.symlink(cache_dir, model_cache_dir)
        else:
            raise e

    arg_list = ["--model_type=bert",
                f"--model_name_or_path={bert_base_model_name}",
                "--do_train",
                "--do_eval",
                f"--task_name={task_name}",
                "--do_predict",
                "--do_lower_case",
                "--loss_type=ce",
                f"--data_dir={data_dir}",
                "--train_max_seq_length=128",
                "--eval_max_seq_length=512",
                "--per_gpu_train_batch_size=24",
                "--per_gpu_eval_batch_size=24",
                "--learning_rate=3e-5",
                "--num_train_epochs=4.0",
                "--logging_steps=224",
                "--save_steps=224",
                f"--output_dir={output_dir}/",
                "--seed=42",
                f"--cache_dir={cache_dir}",
                "--overwrite_output_dir",
                # --predict_type=$PREDICT_TYPE
                ]
    return arg_list


def main():
    predict_type = "softmax"
    arg_list = setup_running_env()
    # predict_pretrained_model(predict_type, arg_list)
    # evaluate_pretrained_model(predict_type, arg_list)
    train_eval_test(predict_type, arg_list)


def predict_pretrained_model(predict_type, arg_list):
    runner = get_runner(predict_type)
    args, config, tokenizer, model, _ = runner.setup(arg_list=arg_list)
    prefix = os.path.join("predict", predict_type)
    os.makedirs(os.path.join(args.output_dir, prefix), exist_ok=True)
    runner.predict(args, model, tokenizer, prefix=prefix)


def evaluate_pretrained_model(predict_type, arg_list):
    runner = get_runner(predict_type)
    args, config, tokenizer, model, _ = runner.setup(arg_list=arg_list)
    prefix = os.path.join("evaluate", predict_type)
    os.makedirs(os.path.join(args.output_dir, prefix), exist_ok=True)
    runner.evaluate(args, model, tokenizer, prefix=prefix)


def train_eval_test(predict_type, arg_list):
    runner = get_runner(predict_type)
    args, config, tokenizer, model, model_classes = runner.setup(
        arg_list=arg_list)
    config_class, model_class, tokenizer_class = model_classes

    # Training
    if args.do_train:
        train_dataset = runner.load_and_cache_examples(
            args, args.task_name, tokenizer, data_type='train')
        global_step, tr_loss = runner.train(
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
            result = runner.evaluate(args, model, tokenizer, prefix=prefix)
            if global_step:
                result = {"{}_{}".format(global_step, k)
                                         : v for k, v in result.items()}
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
            prefix = os.path.join("predict", predict_type, prefix)
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            os.makedirs(os.path.join(args.output_dir, prefix), exist_ok=True)
            runner.predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
