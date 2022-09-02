import pandora.server.runner as runner
from pathlib import Path
import os


def main():

    home = str(Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    task_name = "sentence"
    mode_type = "bert"
    bert_base_model_name = "bert-base-chinese"
    arg_list = runner.get_training_args(
        task_name=task_name,
        mode_type=mode_type,
        bert_base_model_name=bert_base_model_name,
    )

    arg_list.extend(
        runner.get_default_dirs(
            resource_dir,
            cache_dir,
            task_name=task_name,
            bert_base_model_name=bert_base_model_name,
        ))
    arg_list.extend(
        runner.set_actions(
            do_train=True,
            do_eval=True,
            do_predict=True,
        ))
    runner.train_eval_test(arg_list)


if __name__ == "__main__":
    main()
