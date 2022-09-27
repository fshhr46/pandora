import os
import shutil
from pathlib import Path

import pandora.service.job_runner as job_runner
import pandora.packaging.packager as packager

from pandora.dataset.sentence_data import Dataset


def main():

    home = str(Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    task_name = "sentence"
    model_type = "bert"
    bert_base_model_name = "bert-base-chinese"

    # Build dataset
    num_data_entry_train = 100
    num_data_entry_test = 10

    # dataset_name_prefix = "synthetic_data"
    dataset_name_prefix = "pandora_demo_meta"
    dataset_name = f"{dataset_name_prefix}_{num_data_entry_train}_{num_data_entry_test}"
    output_dir = os.path.join(resource_dir, "outputs",
                              bert_base_model_name, dataset_name)

    import build_synthetic_datasets as dataset_builder
    dataset_names = [
        Dataset.short_sentence
    ]
    # dataset_names = [dataset_name]

    default_datasets = [e.value for e in Dataset]
    for dataset_name in dataset_names:
        if dataset_name in default_datasets:
            continue
        dataset_builder.build_dataset(
            dataset_name=dataset_name,
            num_data_entry_train=num_data_entry_train,
            num_data_entry_test=num_data_entry_test,
        )

    # Set args
    arg_list = job_runner.get_training_args(
        task_name=task_name,
        mode_type=model_type,
        bert_base_model_name=bert_base_model_name,
    )

    arg_list.extend(
        job_runner.get_default_dirs(
            resource_dir,
            cache_dir,
            bert_base_model_name=bert_base_model_name,
            datasets=dataset_names,
        ))
    arg_list.extend(
        job_runner.set_actions(
            do_train=True,
            do_eval=True,
            do_predict=True,
        ))
    resource_dir = os.path.join(Path.home(), "workspace", "resource")

    # Start training
    args = job_runner.train_eval_test(arg_list, resource_dir=resource_dir,
                                      datasets=dataset_names)

    # packaging
    pkger = packager.ModelPackager(
        model_dir=output_dir,
        eval_max_seq_length=args.eval_max_seq_length)
    package_dir = packager.get_package_dir(model_dir=output_dir)

    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    assert not os.path.isdir(
        package_dir), f"package_dir {package_dir} already exists, please delete"
    pkger.build_model_package()
    print(f"package_dir is {package_dir}")


if __name__ == "__main__":
    main()
