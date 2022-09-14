import pandora.service.job_runner as job_runner
from pathlib import Path
import os

from pandora.dataset.sentence_data import Dataset
import pandora.packaging.packager as packager


# TODO: TEST_DATASETS should be input parameter
TEST_DATASETS = list(Dataset)
TEST_DATASETS = [
    Dataset.synthetic_data,
    # Dataset.column_data,
    # Dataset.short_sentence,
    # Dataset.long_sentence,
]
TEST_DATASETS.sort()


def main():

    home = str(Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    task_name = "sentence"
    mode_type = "bert"
    bert_base_model_name = "bert-base-chinese"
    arg_list = job_runner.get_training_args(
        task_name=task_name,
        mode_type=mode_type,
        bert_base_model_name=bert_base_model_name,
    )

    arg_list.extend(
        job_runner.get_default_dirs(
            resource_dir,
            cache_dir,
            bert_base_model_name=bert_base_model_name,
            datasets=TEST_DATASETS,
        ))
    arg_list.extend(
        job_runner.set_actions(
            do_train=True,
            do_eval=True,
            do_predict=True,
        ))
    resource_dir = os.path.join(Path.home(), "workspace", "resource")
    job_runner.train_eval_test(arg_list, resource_dir=resource_dir,
                               datasets=TEST_DATASETS)

    # packaging
    dataset_names = "_".join(TEST_DATASETS)
    output_dir = os.path.join(resource_dir, "outputs",
                              bert_base_model_name, dataset_names)
    pkger = packager.ModelPackager(model_dir=output_dir)
    package_dir = pkger.build_model_package()
    print(f"package_dir is {package_dir}")


if __name__ == "__main__":
    main()
