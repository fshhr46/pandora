import collections
import json
from multiprocessing.sharedctypes import Value
import os
import copy
import random
import pathlib
import logging

from enum import Enum
from typing import List, Dict

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class MetadataType(str, Enum):
    def __str__(self):
        return str(self.value)
    column_name = "column_name"
    column_comment = "column_comment"
    column_descripition = "column_descripition"


class ModelTaskType(str, Enum):
    def __str__(self):
        return str(self.value)
    tag = "tag"  # tagging model
    classification = "classification"  # classification_ratings model


class TrainingType(str, Enum):
    def __str__(self):
        return str(self.value)
    meta_data = "meta_data"  # use meta data for training (like column names)
    column_data = "column_data"  # use data entries column
    mixed_data = "mixed_data"  # use metadata and column data together


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id, labels, sentence, meta_data):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. Multiple labels for the sentence
        """
        # TODO: Remove id from example
        self.id = id
        self.labels = labels
        self.sentence = sentence
        self.meta_data = meta_data

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            input_len,
            segment_ids,
            sentence_labels,
            char_input_ids,
            start_ids,
            end_ids):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sentence_labels = sentence_labels
        self.input_len = input_len
        self.char_input_ids = char_input_ids
        self.start_ids = start_ids
        self.end_ids = end_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def batch_collate_fn_bert(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # bert model
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_labels


def batch_collate_fn_char_bert(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """

    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_char_input_ids, all_start_ids, all_end_ids, \
        all_labels = map(torch.stack, zip(*batch))

    # This is to save GPU memory by shrinking the size of tensors.
    # Chart Bert doesn't do this
    # max_len = max(all_lens).item()
    # all_input_ids = all_input_ids[:, :max_len]
    # all_attention_mask = all_attention_mask[:, :max_len]
    # all_token_type_ids = all_token_type_ids[:, :max_len]
    # all_labels = all_labels[:]
    # all_char_input_ids = all_char_input_ids[:, :max_len]
    # all_start_ids = all_start_ids[:, :max_len]
    # all_end_ids = all_end_ids[:, :max_len]

    return all_input_ids, all_attention_mask, all_token_type_ids, all_lens, \
        all_char_input_ids, all_start_ids, all_end_ids, \
        all_labels


def get_text_from_example(
        example,
        training_type,
        meta_data_types):
    # create meta_data_text
    meta_data_types = sorted(meta_data_types)
    meta_data_vals = []
    for meta_data_type in meta_data_types:
        # TODO: Fix this hack: manually replacing the underscores.
        if meta_data_type == MetadataType.column_name:
            meta_data_vals.append(
                example.meta_data[meta_data_type].replace("_", " "))
        else:
            meta_data_vals.append(example.meta_data[meta_data_type])
    meta_data_text = "|".join(meta_data_vals)

    # append meta_data_text to sentence

    sentence = example.sentence
    if training_type == TrainingType.column_data:
        assert sentence, "text data is required for column_data model"
        text = sentence
    elif training_type == TrainingType.meta_data:
        assert meta_data_text, "meta_data_text data is required for meta_data model"
        text = meta_data_text
    elif training_type == TrainingType.mixed_data:
        assert sentence, "text data is required for mixed_data model"
        assert meta_data_text, "meta_data_text data is required for mixed_data model"
        combined_text = f"{meta_data_text}|{sentence}"
        text = combined_text
    else:
        raise ValueError(f"invalid training_type {training_type}")
    return text


def convert_tokens_to_char_ids(
    tokens,
    tokenizer,
    max_seq_length,
    char2ids_dict,
):
    # convert input_ids to char_input_ids
    #all_seq_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    char_ids = []
    start_ids = []
    end_ids = []
    char_maxlen = max_seq_length * 6

    #print(f"Input tokens: {' '.join(tokens)}")
    for idx, token in enumerate(tokens):  # char_ids for debug
        if len(char_ids) >= char_maxlen:
            break
        token = token.strip("##")
        # Special tokens
        if token in [tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token,
                     tokenizer.cls_token, tokenizer.mask_token]:
            start_ids.append(len(char_ids))
            end_ids.append(len(char_ids))
            char_ids.append(0)
        else:
            for char_idx, c in enumerate(token):
                if len(char_ids) >= char_maxlen:
                    break

                if char_idx == 0:
                    start_ids.append(len(char_ids))
                if char_idx == len(token) - 1:
                    end_ids.append(len(char_ids))

                if c in char2ids_dict:
                    cid = char2ids_dict[c]
                else:
                    cid = char2ids_dict["<unk>"]
                char_ids.append(cid)

        if len(char_ids) < char_maxlen:
            char_ids.append(0)
        # if True:
        #    print(f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')
    #print(f'len of char_ids: {len(char_ids)}')

    if len(char_ids) > char_maxlen:
        char_ids = char_ids[:char_maxlen]
    else:
        pad_len = char_maxlen - len(char_ids)
        char_ids = char_ids + [0] * pad_len
    while len(start_ids) < max_seq_length:
        start_ids.append(char_maxlen-1)
    while len(end_ids) < max_seq_length:
        end_ids.append(char_maxlen-1)

    assert len(char_ids) == char_maxlen
    assert len(start_ids) == max_seq_length
    assert len(end_ids) == max_seq_length
    return char_ids, start_ids, end_ids


# Convert example to feature, and pad them
def convert_example_to_feature(
        example,
        training_type: TrainingType,
        meta_data_types: List[MetadataType],
        label2id,
        log_data: bool,
        max_seq_length,
        tokenizer, pad_token=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        char2ids_dict: Dict = None,
        mask_padding_with_zero=True):

    text = get_text_from_example(
        example,
        training_type,
        meta_data_types)
    # sentence labels
    sentence_labels = [label2id[x] for x in example.labels]

    # Extract tokens
    # Input token IDs from tokenizer
    tokens = tokenizer.tokenize(text)
    # assert tokens_words == tokens_sentence
    # tokens = tokens_sentence

    # TODO: Remove duplicate
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # TODO: Fix this. Currently special token is removed.
    # However, this is not necessary for sentence classification
    special_tokens_count = 2
    if len(tokens) > max_seq_length:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        input_ids = input_ids[: (max_seq_length - special_tokens_count)]

    # Input mask
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # TODO: Remove Segment ID, it is not needed for classification task
    # Segment ID
    segment_ids = [sequence_a_segment_id] * len(tokens)

    # Zero-pad up to the sequence length.
    input_len = len(input_ids)
    padding_length = max_seq_length - len(input_ids)

    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length

    if char2ids_dict is not None:
        char_input_ids, start_ids, end_ids = convert_tokens_to_char_ids(
            tokens=tokens,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            char2ids_dict=char2ids_dict,
        )
    else:
        char_input_ids, start_ids, end_ids = None, None, None

    # Check length
    assert len(
        input_ids) == max_seq_length, f"len(input_ids): {len(input_ids)}, max_seq_length {max_seq_length}"
    assert len(
        input_mask) == max_seq_length, f"len(input_ids): {len(input_mask)}, max_seq_length {max_seq_length}"
    assert len(
        segment_ids) == max_seq_length, f"len(input_ids): {len(segment_ids)}, max_seq_length {max_seq_length}"

    # Only support single label now (or no label)
    assert len(sentence_labels) <= 1
    if log_data:
        logger.info("*** Example ***")
        logger.info("id: %s", example.id)
        logger.info("num tokens: %s", len(tokens))
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids size: %s", len(input_ids))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join(
            [str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join(
            [str(x) for x in segment_ids]))
        logger.info("sentence_labels: %s", " ".join(
            [str(x) for x in sentence_labels]))
        logger.info("char_input_ids: %s", " ".join(
            [str(x) for x in char_input_ids]))
        logger.info("start_ids: %s", " ".join(
            [str(x) for x in start_ids]))
        logger.info("end_ids: %s", " ".join(
            [str(x) for x in end_ids]))

    feature = InputFeatures(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        input_mask=torch.tensor(input_mask, dtype=torch.long),
        input_len=torch.tensor(input_len, dtype=torch.long),
        segment_ids=torch.tensor(segment_ids, dtype=torch.long),
        sentence_labels=torch.tensor(sentence_labels, dtype=torch.long),
        char_input_ids=torch.tensor(
            char_input_ids, dtype=torch.long) if char_input_ids else None,
        start_ids=torch.tensor(
            start_ids, dtype=torch.long) if start_ids else None,
        end_ids=torch.tensor(end_ids, dtype=torch.long) if end_ids else None)
    return feature


def convert_features_to_dataset(local_rank, features, evaluate, include_char_data):
    if local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # Convert to Tensors and build dataset
    all_input_ids = torch.stack(
        [f.input_ids for f in features])
    all_input_mask = torch.stack(
        [f.input_mask for f in features])
    all_segment_ids = torch.stack(
        [f.segment_ids for f in features])

    # Only support single label now.
    all_label_ids = torch.stack(
        [f.sentence_labels[0] for f in features])

    all_lens = torch.stack([f.input_len for f in features])

    # add char-bert tensors
    if include_char_data:
        all_char_input_ids = torch.stack(
            [f.char_input_ids for f in features])
        all_start_ids = torch.stack(
            [f.start_ids for f in features])
        all_end_ids = torch.stack(
            [f.end_ids for f in features])
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lens,
            all_char_input_ids, all_start_ids, all_end_ids,
            all_label_ids)
    else:
        dataset = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_lens,
            all_label_ids)
    return dataset


def create_example(id, line):
    labels = line['labels']
    sentence = str(line['sentence'])
    meta_data = line['meta_data']

    return InputExample(
        id=id,
        labels=labels,
        sentence=sentence,
        meta_data=meta_data)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        raise NotImplementedError()

    @classmethod
    def _read_text(self, input_file):
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        raise NotImplementedError()

    @classmethod
    def _create_labels(self, line):
        raise NotImplementedError()


class SentenceProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def __init__(
            self,
            training_type: TrainingType,
            meta_data_types: List[MetadataType],
            resource_dir,
            datasets: List[str]) -> None:
        super().__init__()
        self.training_type = training_type
        self.meta_data_types = meta_data_types
        self.resource_dir = resource_dir
        self.datasets = datasets

    def _get_examples_all_dir(self, data_dir, partition) -> List[InputExample]:
        all_examples = []
        for dataset in self.datasets:
            all_examples.extend(
                self.create_examples(self._read_json(
                    os.path.join(data_dir, dataset, f"{partition}.json")), partition)
            )
        return all_examples

    def get_train_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        return self._get_examples_all_dir(data_dir, "train")

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        return self._get_examples_all_dir(data_dir, "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        return self._get_examples_all_dir(data_dir, "test")

    def get_labels(self):
        """See base class."""
        all_labels = []
        for dataset in self.datasets:
            if dataset:
                data_dir = os.path.join(self.resource_dir, "datasets")
            # TODO: This is hacky relying on "dataset" variable being a empty string
            else:
                data_dir = self.resource_dir
            label_file = open(os.path.join(
                data_dir, dataset, "labels.json"))
            all_labels.extend(json.load(label_file))
        return all_labels

    def create_examples(self, lines, dataset_type: str) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            example = create_example(
                id=guid,
                line=line
            )
            examples.append(example)
        return examples

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r') as f:
            return read_json_lines(f)

    @classmethod
    def _create_labels(self, line):
        return _create_labels(line)

    def build_char_inputs(self, input_ids, sub_index_to_ori_tok, start, rng, labels):
        all_seq_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # if True:
        if self.args.output_debug:
            print(f"all_seq_tokens: {' '.join(all_seq_tokens)}")
        char_ids = []
        start_ids = []
        end_ids = []
        char_maxlen = self.args.block_size * self.args.char_maxlen_for_word
        for idx, token in enumerate(all_seq_tokens):
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")
            if token == self.tokenizer.unk_token:
                tok_orig_index = idx+start - 1
                if tok_orig_index in sub_index_to_ori_tok:  # -1 for CLS
                    orig_token = sub_index_to_ori_tok[tok_orig_index]
                    #print(f'UNK: {token} to orig_tokens: {orig_token}')
                    token = orig_token
            if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"] or labels[idx] != -1:
                start_ids.append(len(char_ids))
                end_ids.append(len(char_ids))
                char_ids.append(0)
            else:
                for char_idx, c in enumerate(token):
                    if len(char_ids) >= char_maxlen:
                        break

                    if char_idx == 0:
                        start_ids.append(len(char_ids))
                    if char_idx == len(token) - 1:
                        end_ids.append(len(char_ids))

                    if c in self.char2ids_dict:
                        cid = self.char2ids_dict[c]
                    else:
                        cid = self.char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)
            # if True:
            if self.args.output_debug:
                print(
                    f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')
        #print(f'len of char_ids: {len(char_ids)}')
        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < self.args.block_size:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < self.args.block_size:
            end_ids.append(char_maxlen-1)
        # if True:
        if self.args.output_debug:
            print(f'char_ids : {" ".join(map(str, char_ids))}')
            print(f'start_ids: {" ".join(map(str, start_ids))}')
            print(f'end_ids  : {" ".join(map(str, end_ids))}')
        #max_start = max(start_ids)
        #max_end   = max(end_ids)
        # if max_start > char_maxlen or max_end > char_maxlen:
        #    print("Error sequence information")
        #    exit(0)
        return char_ids, start_ids, end_ids


def read_json_lines(json_lines_itr):
    lines = []
    for text_line in json_lines_itr:
        line_obj = json.loads(text_line.strip())
        lines.append(read_json_line(line_obj))
    return lines


def read_json_line(line_obj: Dict):
    # 把句子拆成一个一个单词
    # >> > words = "abcde你好啊"
    # >> > list(words)
    # ['a', 'b', 'c', 'd', 'e', '你', '好', '啊']
    # >> >
    # TODO: Fix this column name concat
    sentence = line_obj['text']
    meta_data = line_obj['meta_data']
    labels = _create_labels(line_obj)
    return {
        "labels": labels,
        "sentence": sentence,
        "meta_data": meta_data}


def _create_labels(line_obj):
    # label is a list of sentence level classes
    # words is not used as this is sentence level classification
    return line_obj.get('label', [])


class RandomDataSampler(object):
    """Sampler to select data.
    """

    def __init__(self, seed: int, sample_size: int) -> None:
        self.sample_size = sample_size

    def group_data_by_labels(self,
                             examples: List[InputExample]) -> Dict[str, InputExample]:
        examples_by_label = {}
        for example in examples:
            for label in example.labels:
                if label not in examples_by_label:
                    examples_by_label[label] = []
                examples_by_label[label].append(example)
        return examples_by_label

    def sample(self, examples: List[InputExample]) -> List[InputExample]:
        examples_by_label = self.group_data_by_labels(examples)
        output_examples = []
        for _, data in examples_by_label.items():
            output_examples.extend(random.sample(data, self.sample_size))
        return output_examples


def extract_feature_from_request(
        request_data,
        training_type: TrainingType,
        meta_data_types: List[MetadataType],
        label2id,
        max_seq_length,
        tokenizer,
        char2ids_dict=None):
    """This function extract feature from a request data.
    Request data is a dictionary with "data" and "column_name" keys

    Args:
        request_data (int): Batches of tokens IDs of text
    Returns:
        feature
    """

    input_text = request_data.get("data")
    if isinstance(input_text, (bytes, bytearray)):
        input_text = input_text.decode("utf-8")
    logger.debug(f"input_text is {input_text}")

    column_name = request_data.get("column_name")
    if isinstance(column_name, (bytes, bytearray)):
        column_name = column_name.decode("utf-8")
    logger.debug(f"column_name is {column_name}")

    column_comment = request_data.get("column_comment")
    if isinstance(column_comment, (bytes, bytearray)):
        column_comment = column_comment.decode("utf-8")
    logger.debug(f"column_comment is {column_comment}")

    column_description = request_data.get("column_description")
    if isinstance(column_description, (bytes, bytearray)):
        column_description = column_description.decode("utf-8")
    logger.debug(f"column_description is {column_description}")

    meta_data = {
        "column_name": column_name,
        "column_comment": column_comment,
        "column_description": column_description,
    }
    line = read_json_line(
        line_obj={"text": input_text, "meta_data": meta_data})
    # TODO: fix this hack: id=""
    example = create_example(
        id="",
        line=line
    )
    feat = convert_example_to_feature(
        example,
        training_type,
        meta_data_types,
        label2id,
        log_data=False,
        max_seq_length=int(max_seq_length),
        tokenizer=tokenizer,
        char2ids_dict=char2ids_dict)
    return feat


def load_char_vocab():
    curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())
    fname = os.path.join(curr_dir, "bert_char_vocab")
    return load_line_to_ids_dict(fname)


def load_line_to_ids_dict(fname):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(fname, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        vocab[char] = index
    return vocab


def build_inputs_from_batch(batch, include_labels, include_char_data):
    inputs = {"input_ids": batch[0],
              "attention_mask": batch[1],
              "token_type_ids": batch[2],
              }

    # This is currently not used.
    all_lens = batch[3]
    if include_char_data:
        inputs["char_input_ids"] = batch[4]
        inputs["start_ids"] = batch[5]
        inputs["end_ids"] = batch[6]

    if include_labels:
        inputs["labels"] = batch[-1]
    return inputs
