import os
import json
import pandora.tools.common as common
import pathlib
import shutil

import pandora

PACKAGE_DIR_NAME = "torchserve_package"

# model dir files
MODEL_FILE_NAME = "pytorch_model.bin"
MODEL_CONFIG_FILE_NAME = "config.json"
VOCAB_FILE_NAME = "vocab.txt"
INDEX2NAME_FILE_NAME = "index_to_name.json"

MODEL_FILES_TO_COPY = [MODEL_FILE_NAME, MODEL_CONFIG_FILE_NAME,
                       VOCAB_FILE_NAME, INDEX2NAME_FILE_NAME]

# torchserve related names
PANDORA_DEPENDENCY = "pandora.zip"
SERUP_CONF_FILE_NAME = "setup_config.json"
HANDLER_NAME = "handler.py"
REGISTER_SCRIPT_NAME = "register.sh"
PACKAGE_SCRIPT_NAME = "package.sh"
PACKAGING_DONE_FILE = "package.done"


def get_package_dir(model_dir: str):
    return os.path.join(model_dir, PACKAGE_DIR_NAME)


def done_packaging(model_dir: str):
    done_file = os.path.join(get_package_dir(model_dir), PACKAGING_DONE_FILE)
    return os.path.isfile(done_file)


class ModelPackager(object):
    def __init__(self,
                 model_dir: str,) -> None:
        self.model_dir = model_dir

    def create_setup_config_file(self, package_dir):
        setup_conf = {
            "model_name": "bert-base-chinese",
            # "mode": "sequence_classification",
            "mode": "sequence_classification",
            "do_lower_case": True,
            "num_labels": "10",
            "save_mode": "pretrained",
            "max_length": "150",
            "captum_explanation": False,  # TODO: make this True
            "embedding_name": "bert",
            "FasterTransformer": False,  # TODO: make this True
            "model_parallel": False  # Beta Feature, set to False for now.
        }
        setup_conf_path = os.path.join(package_dir, SERUP_CONF_FILE_NAME)
        with open(setup_conf_path, "w") as setup_conf_f:
            json.dump(setup_conf, setup_conf_f, indent=4)

    def build_model_package(self):
        assert os.path.isdir(
            self.model_dir), f"model_dir {self.model_dir} is not a directory"

        # create package dir
        package_dir = get_package_dir(self.model_dir)
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)

        # create torchserve config file
        self.create_setup_config_file(package_dir)

        # create package file
        self.create_package_script(package_dir)

        curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
        # copy register.sh file
        common.copy_file(curr_dir, package_dir, REGISTER_SCRIPT_NAME)

        # copy handler.py file
        common.copy_file(curr_dir, package_dir, HANDLER_NAME)

        # copy pandora as dependency
        pandora_dir = os.path.dirname(pandora.__file__)
        pandora_zip_path = os.path.join(package_dir, PANDORA_DEPENDENCY)
        common.zipdir(dir_to_zip=pandora_dir, output_path=pandora_zip_path)

        for file_name in MODEL_FILES_TO_COPY:
            common.copy_file(self.model_dir, package_dir, file_name)

        # mark done
        open(os.path.join(package_dir, PACKAGING_DONE_FILE), "w")
        return package_dir

    def get_command(self):
        return f"torch-model-archiver \
            --model-name $model_name \
            --version $model_version \
            --serialized-file {MODEL_FILE_NAME} \
            --handler {HANDLER_NAME} \
            --extra-files \
            \"{MODEL_CONFIG_FILE_NAME},{SERUP_CONF_FILE_NAME},{INDEX2NAME_FILE_NAME},{VOCAB_FILE_NAME},{PANDORA_DEPENDENCY}\""

    def create_package_script(self, package_dir):
        # copy sample file to package dir
        curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
        common.copy_file(curr_dir, package_dir, PACKAGE_SCRIPT_NAME)
        # append command to dir
        script_path = os.path.join(package_dir, PACKAGE_SCRIPT_NAME)
        with open(script_path, "a") as script_f:
            script_f.write(self.get_command())
