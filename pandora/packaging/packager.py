import os
import json
import pandora.tools.common as common
import pathlib

import pandora.tools.common as common
from pandora.packaging.constants import (
    PACKAGE_DIR_NAME,
    REGISTER_SCRIPT_NAME,
    FEATURE_NAME,
    CONSTANT_NAME,
    MODEL_FILES_TO_COPY,
    PACKAGING_DONE_FILE,
    MODEL_FILE_NAME,
    HANDLER_NAME,
    MODEL_CONFIG_FILE_NAME,
    SERUP_CONF_FILE_NAME,
    INDEX2NAME_FILE_NAME,
    VOCAB_FILE_NAME,
    BERT_MODEL_NAME,
    CHAR_BERT_MODEL_NAME,
    TOKENIZER_NAME,
    INFERENCE_NAME,
    CHARBERT_CHAR_VOCAB,
    CHARBERT_TERM_VOCAB,
    PACKAGE_SCRIPT_NAME,
)


def get_package_dir(model_dir: str):
    return os.path.join(model_dir, PACKAGE_DIR_NAME)


def done_packaging(model_dir: str):
    done_file = os.path.join(get_package_dir(
        model_dir), PACKAGING_DONE_FILE)
    return os.path.isfile(done_file)


class ModelPackager(object):
    def __init__(self,
                 model_dir: str,
                 eval_max_seq_length: int) -> None:
        self.model_dir = model_dir
        self.eval_max_seq_length = eval_max_seq_length

    def build_model_package(self):
        assert os.path.isdir(
            self.model_dir), f"model_dir {self.model_dir} is not a directory"

        # create package dir
        package_dir = get_package_dir(self.model_dir)
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)

        # create package file
        self.create_package_script(package_dir)

        curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
        # copy register.sh file
        ts_url = os.environ.get("TORCHSERVE_URL")
        # replace ts_url
        if ts_url:
            default_ts_url = "localhost:8081"
            old_reg_file = os.path.join(curr_dir, REGISTER_SCRIPT_NAME)
            new_reg_file = os.path.join(package_dir, REGISTER_SCRIPT_NAME)
            with open(old_reg_file) as f_in:
                with open(new_reg_file, "w") as f_out:
                    for line in f_in.readlines():
                        if default_ts_url in line:
                            line = line.replace(default_ts_url, ts_url)
                        f_out.write(line)
        else:
            common.copy_file(curr_dir, package_dir, REGISTER_SCRIPT_NAME)

        # copy handler, model and tokenizer, and other python files
        # TODO: Make a list out of this
        common.copy_file(curr_dir, package_dir, HANDLER_NAME)
        common.copy_file(curr_dir, package_dir, BERT_MODEL_NAME)
        common.copy_file(curr_dir, package_dir, CHAR_BERT_MODEL_NAME)
        common.copy_file(curr_dir, package_dir, TOKENIZER_NAME)
        common.copy_file(curr_dir, package_dir, INFERENCE_NAME)
        common.copy_file(curr_dir, package_dir, FEATURE_NAME)
        common.copy_file(curr_dir, package_dir, CONSTANT_NAME)

        # copy char_bert vocab files
        common.copy_file(curr_dir, package_dir, CHARBERT_CHAR_VOCAB)
        common.copy_file(curr_dir, package_dir, CHARBERT_TERM_VOCAB)

        # copy pandora as dependency
        # pandora_dir = os.path.dirname(pandora.__file__)
        # pandora_zip_path = os.path.join(package_dir, PANDORA_DEPENDENCY)
        # common.zipdir(dir_to_zip=pandora_dir, output_path=pandora_zip_path)

        for file_name in MODEL_FILES_TO_COPY:
            common.copy_file(self.model_dir, package_dir, file_name)

        # mark done
        open(os.path.join(package_dir, PACKAGING_DONE_FILE), "w")
        return package_dir

    def get_command(self):
        base_cmd = [
            "torch-model-archiver",
            "--force",
            f"--model-name $model_name",
            f"--version $model_version",
            f"--serialized-file {MODEL_FILE_NAME}",
            f"--handler {HANDLER_NAME}",
            "--extra-files"]
        extra_files = [
            MODEL_CONFIG_FILE_NAME, SERUP_CONF_FILE_NAME,
            INDEX2NAME_FILE_NAME, VOCAB_FILE_NAME,
            BERT_MODEL_NAME, CHAR_BERT_MODEL_NAME,
            TOKENIZER_NAME,
            INFERENCE_NAME, FEATURE_NAME,
            CONSTANT_NAME,
            CHARBERT_CHAR_VOCAB, CHARBERT_TERM_VOCAB]
        return f'{" ".join(base_cmd)} {",".join(extra_files)}'

    def create_package_script(self, package_dir):
        # copy sample file to package dir
        curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
        common.copy_file(curr_dir, package_dir, PACKAGE_SCRIPT_NAME)
        # append command to dir
        script_path = os.path.join(package_dir, PACKAGE_SCRIPT_NAME)
        with open(script_path, "a") as script_f:
            script_f.write(self.get_command())
            script_f.write("\n")
