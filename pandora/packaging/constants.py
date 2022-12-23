
PACKAGE_DIR_NAME = "torchserve_package"

# model dir files
MODEL_FILE_NAME = "pytorch_model.bin"
MODEL_CONFIG_FILE_NAME = "config.json"
VOCAB_FILE_NAME = "vocab.txt"
INDEX2NAME_FILE_NAME = "index_to_name.json"
SERUP_CONF_FILE_NAME = "setup_config.json"
PROCESSOR_PROFILE_FILE_NAME = "processor_profile.json"

# char-bert related
CHARBERT_CHAR_VOCAB = "char-bert_char_vocab.txt"
CHARBERT_TERM_VOCAB = "char-bert_term_vocab.txt"

MODEL_FILES_TO_COPY = [MODEL_FILE_NAME, MODEL_CONFIG_FILE_NAME,
                       VOCAB_FILE_NAME, INDEX2NAME_FILE_NAME,
                       SERUP_CONF_FILE_NAME]


# handler and python files
# PANDORA_DEPENDENCY = "pandora.zip"
HANDLER_NAME = "handler.py"
BERT_MODEL_NAME = "model.py"
CHAR_BERT_MODEL_NAME = "char_bert_model.py"
TOKENIZER_NAME = "tokenizer.py"
CLASSIFIER_NAME = "classifier.py"
INFERENCE_NAME = "inference.py"
FEATURE_NAME = "feature.py"
CONSTANT_NAME = "constants.py"

# torchserve related names
REGISTER_SCRIPT_NAME = "register.sh"
PACKAGE_SCRIPT_NAME = "package.sh"
PACKAGING_DONE_FILE = "package.done"
