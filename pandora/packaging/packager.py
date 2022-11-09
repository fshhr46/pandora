import os
import json
import pandora.tools.common as common
import pathlib

from . import constants


def get_package_dir(model_dir: str):
    return os.path.join(model_dir, constants.PACKAGE_DIR_NAME)


def done_packaging(model_dir: str):
    done_file = os.path.join(get_package_dir(
        model_dir), constants.PACKAGING_DONE_FILE)
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
        common.copy_file(curr_dir, package_dir, constants.REGISTER_SCRIPT_NAME)

        # copy handler, model and tokenizer
        # TODO: Make a list out of this
        common.copy_file(curr_dir, package_dir, constants.HANDLER_NAME)
        common.copy_file(curr_dir, package_dir, constants.MODEL_NAME)
        common.copy_file(curr_dir, package_dir, constants.TOKENIZER_NAME)
        common.copy_file(curr_dir, package_dir, constants.INFERENCE_NAME)
        common.copy_file(curr_dir, package_dir, constants.FEATURE_NAME)

        # copy char_bert vocab files
        common.copy_file(curr_dir, package_dir, constants.CHARBERT_CHAR_VOCAB)
        common.copy_file(curr_dir, package_dir, constants.CHARBERT_TERM_VOCAB)

        # copy pandora as dependency
        # pandora_dir = os.path.dirname(pandora.__file__)
        # pandora_zip_path = os.path.join(package_dir, PANDORA_DEPENDENCY)
        # common.zipdir(dir_to_zip=pandora_dir, output_path=pandora_zip_path)

        for file_name in constants.MODEL_FILES_TO_COPY:
            common.copy_file(self.model_dir, package_dir, file_name)

        # mark done
        open(os.path.join(package_dir, constants.PACKAGING_DONE_FILE), "w")
        return package_dir

    def get_command(self):
        base_cmd = [
            "torch-model-archiver",
            "--force",
            f"--model-name $model_name",
            f"--version $model_version",
            f"--serialized-file {constants.MODEL_FILE_NAME}",
            f"--handler {constants.HANDLER_NAME}",
            "--extra-files"]
        extra_files = [
            constants.MODEL_CONFIG_FILE_NAME, constants.SERUP_CONF_FILE_NAME,
            constants.INDEX2NAME_FILE_NAME, constants.VOCAB_FILE_NAME,
            constants.MODEL_NAME, constants.TOKENIZER_NAME,
            constants.INFERENCE_NAME, constants.FEATURE_NAME,
            constants.CHARBERT_CHAR_VOCAB, constants.CHARBERT_TERM_VOCAB]
        return f'{" ".join(base_cmd)} {",".join(extra_files)}'

    def create_package_script(self, package_dir):
        # copy sample file to package dir
        curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
        common.copy_file(curr_dir, package_dir, constants.PACKAGE_SCRIPT_NAME)
        # append command to dir
        script_path = os.path.join(package_dir, constants.PACKAGE_SCRIPT_NAME)
        with open(script_path, "a") as script_f:
            script_f.write(self.get_command())
            script_f.write("\n")
