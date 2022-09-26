import csv
import json
import os
import copy
import random
from typing import List, Dict
import torch

import logging

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, id, words, labels, sentence):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. Multiple labels for the sentence
        """
        # TODO: Remove id from example
        self.id = id
        self.words = words
        self.labels = labels
        self.sentence = sentence

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

    def __init__(self, input_ids, input_mask, input_len, segment_ids, sentence_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sentence_labels = sentence_labels
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def batch_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(
        torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def convert_examples_to_features(examples, label_list,
                                 max_seq_length, tokenizer,
                                 *args, **kwargs):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,

    label_list: all labels
    """
    label2id = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        feature = convert_example_to_feature(
            example, label2id, ex_index < 5,
            max_seq_length, tokenizer,
            *args, **kwargs)
        features.append(feature)
    return features


# Convert example to feature, and pad them
def convert_example_to_feature(example, label2id, log_data,
                               max_seq_length, tokenizer, pad_token=0, pad_token_segment_id=0,
                               sequence_a_segment_id=0, mask_padding_with_zero=True,):
    # sentence labels
    sentence_labels = [label2id[x] for x in example.labels]

    # Input token IDs from tokenizer
    tokens = tokenizer.tokenize(example.words)

    special_tokens_count = 2
    if len(tokens) > max_seq_length:
        tokens = tokens[: (max_seq_length - special_tokens_count)]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Input mask
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Segment ID
    segment_ids = [sequence_a_segment_id] * len(tokens)

    # Zero-pad up to the sequence length.
    input_len = len(input_ids)
    padding_length = max_seq_length - len(input_ids)

    input_ids += [pad_token] * padding_length
    input_mask += [0 if mask_padding_with_zero else 1] * padding_length
    segment_ids += [pad_token_segment_id] * padding_length

    # Check length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # Only support single label now (or no label)
    assert len(sentence_labels) <= 1
    if log_data:
        logger.info("*** Example ***")
        logger.info("id: %s", example.id)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join(
            [str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join(
            [str(x) for x in segment_ids]))
        logger.info("sentence_labels: %s", " ".join(
            [str(x) for x in sentence_labels]))

    feature = InputFeatures(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        input_mask=torch.tensor(input_mask, dtype=torch.long),
        input_len=torch.tensor(input_len, dtype=torch.long),
        segment_ids=torch.tensor(segment_ids, dtype=torch.long),
        sentence_labels=torch.tensor(sentence_labels, dtype=torch.long))
    return feature


def create_example(id, line):
    words = line['words']
    labels = line['labels']
    text = line['sentence']
    return InputExample(
        id=id, words=words, labels=labels, sentence=text)


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
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text'].decode("utf-8")
                # 把句子拆成一个一个单词
                # >> > words = "abcde你好啊"
                # >> > list(words)
                # ['a', 'b', 'c', 'd', 'e', '你', '好', '啊']
                # >> >
                words = list(text)
                labels = cls._create_labels(line, words)
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _create_labels(self, line, words):
        label_entities = line.get('label', None)
        labels = ['O'] * len(words)
        if label_entities is not None:
            for key, value in label_entities.items():
                for sub_name, sub_index in value.items():
                    for start_index, end_index in sub_index:
                        assert ''.join(
                            words[start_index:end_index + 1]) == sub_name
                        if start_index == end_index:
                            labels[start_index] = 'S-' + key
                        else:
                            labels[start_index] = 'B-' + key
                            labels[start_index + 1:end_index +
                                   1] = ['I-' + key] * (len(sub_name) - 1)
        return labels


class SentenceProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def __init__(self, resource_dir, datasets: List[str]) -> None:
        super().__init__()
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

    @classmethod
    def create_examples(cls, lines, dataset_type) -> List[InputExample]:
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            example = create_example(id=guid, line=line)
            examples.append(example)
        return examples

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r') as f:
            return read_json_lines(f)

    @classmethod
    def _create_labels(self, line, words):
        return _create_labels(line, words)


def read_json_lines(json_lines_itr):
    lines = []
    for text_line in json_lines_itr:
        line_obj = json.loads(text_line.strip())
        lines.append(read_json_line(line_obj))
    return lines


def read_json_line(line_obj: Dict):
    text = line_obj['text']
    # 把句子拆成一个一个单词
    # >> > words = "abcde你好啊"
    # >> > list(words)
    # ['a', 'b', 'c', 'd', 'e', '你', '好', '啊']
    # >> >
    # TODO: Fix this column name concat
    column_name = line_obj.get('column_name')
    if column_name:
        text = f"{column_name}, {text}"
    words = list(text)
    labels = _create_labels(line_obj, words)
    return {"words": words, "labels": labels, "sentence": text}


def _create_labels(line_obj, words):
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


# TODO: Delete this
cls_processors = {
    "sentence": SentenceProcessor,
}
