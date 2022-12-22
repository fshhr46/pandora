import os
import shutil
from pathlib import Path

from pandora.packaging.losses import LossType
import pandora.service.job_runner as job_runner
import pandora.packaging.packager as packager
import pandora.tools.training_utils as training_utils

from pandora.dataset.sentence_data import Dataset
from pandora.packaging.feature import MetadataType, TrainingType
from pandora.packaging.classifier import ClassifierType
from pandora.packaging.model import BertBaseModelType


def main():

    home = str(Path.home())
    resource_dir = os.path.join(home, "workspace", "resource")
    cache_dir = os.path.join(home, ".cache/torch/transformers")

    bert_model_type = BertBaseModelType.char_bert
    bert_model_type = BertBaseModelType.bert
    bert_base_model_name = "char-bert"
    bert_base_model_name = "bert-base-uncased"
    bert_base_model_name = "bert-base-chinese"

    # Build dataset
    num_data_entry_train = 10
    num_data_entry_test = 10

    # Classifier
    classifier_type = ClassifierType.doc

    # Training args
    training_type = TrainingType.column_data
    training_type = TrainingType.mixed_data
    training_type = TrainingType.meta_data

    meta_data_types = [
        # MetadataType.column_name,
        MetadataType.column_comment,
        # MetadataType.column_descripition,
    ]

    # Setting num_epochs means choosing the default number based on training_type
    num_epochs = 0

    # if num_epochs is not passed, set num_epochs by training type
    if num_epochs == 0:
        num_epochs = training_utils.get_num_epochs(
            training_type,
            meta_data_types,
            bert_model_type=bert_model_type,
        )

    batch_size = 0
    if batch_size == 0:
        batch_size = training_utils.get_batch_size(
            training_type,
            meta_data_types,
            bert_model_type=bert_model_type,
        )

    # dataset_name_prefix = "synthetic_data"
    dataset_name_prefix = f"pandora_demo_1019_fix"
    dataset_name = f"{dataset_name_prefix}_{num_data_entry_train}_{num_data_entry_test}"

    import build_synthetic_datasets as dataset_builder
    dataset_names = [
        # Dataset.short_sentence
        Dataset.column_data
    ]
    dataset_names = [dataset_name]
    dataset_names = ["poseidon_gov_test_data"]
    dataset_names = ["poseidon_cn_mobile_data"]
    dataset_names = ["meta_data"]
    dataset_names = ["mixed_data"]
    output_dir = os.path.join(
        resource_dir,
        "outputs",
        bert_base_model_name,
        "_".join(dataset_names)
    )

    default_datasets = [e.value for e in Dataset]
    for dataset_name in dataset_names:
        if dataset_name in default_datasets:
            continue
        # dataset_builder.build_dataset(
        #     training_type=training_type,
        #     dataset_name=dataset_name,
        #     num_data_entry_train=num_data_entry_train,
        #     num_data_entry_test=num_data_entry_test,
        #     ingest_data=False,
        # )

    # Set dataset and model args
    num_folds = 0
    doc_pos_weight = 0.5
    arg_list = job_runner.get_training_args(
        # model args
        bert_model_type=bert_model_type,
        bert_base_model_name=bert_base_model_name,
        training_type=training_type,
        meta_data_types=meta_data_types,
        loss_type=LossType.focal_loss,
        classifier_type=classifier_type,
        doc_pos_weight=doc_pos_weight,
        num_folds=num_folds,
        # training args
        num_epochs=num_epochs,
        batch_size=batch_size,
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
    arg_list.append("--overwrite_output_dir")
    resource_dir = os.path.join(Path.home(), "workspace", "resource")

    # Start training
    args = job_runner.run_e2e_modeling(arg_list, resource_dir=resource_dir,
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
