import json
import os
import copy
import torch

from pathlib import Path
from .utils_ner import DataProcessor
from tools import sentence_data
import logging

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, sentence):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. Multiple labels for the sentence
        """
        self.guid = guid
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


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                 sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                 sequence_a_segment_id=0, mask_padding_with_zero=True,):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,

    label_list: all labels
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # sentence labels
        sentence_labels = [label_map[x] for x in example.labels]

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

        # Only support single label now
        assert len(sentence_labels) == 1
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join(
                [str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join(
                [str(x) for x in segment_ids]))
            logger.info("sentence_labels: %s", " ".join(
                [str(x) for x in sentence_labels]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_len=input_len,
                segment_ids=segment_ids,
                sentence_labels=sentence_labels))
    return features


class SentenceProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def __init__(self, resource_dir=None, datasets_to_include=None) -> None:
        super().__init__()
        if not datasets_to_include:
            datasets_to_include = [
                sentence_data.Dataset.column_data,
                sentence_data.Dataset.long_sentence,
                sentence_data.Dataset.short_sentence,
            ]
        if not resource_dir:
            resource_dir = os.path.join(Path.home(), "workspace", "resource")
        self.resource_dir = resource_dir
        self.datasets_to_include = datasets_to_include

    def _get_examples_all_dir(self, data_dir, partition):
        all_examples = []
        for dataset in self.datasets_to_include:
            all_examples.extend(
                self._create_examples(self._read_json(
                    os.path.join(data_dir, dataset, f"{partition}.json")), partition)
            )
        return all_examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._get_examples_all_dir(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._get_examples_all_dir(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._get_examples_all_dir(data_dir, "test")

    def get_labels(self):
        """See base class."""
        all_labels = []
        data_dir = sentence_data.get_output_data_folder(self.resource_dir)
        for dataset in self.datasets_to_include:
            label_file = open(os.path.join(data_dir, dataset, "labels.json"))
            all_labels.extend(json.load(label_file))
        return all_labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            words = line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(
                guid=guid, words=words, labels=labels, sentence=line['sentence']))
        return examples

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                # 把句子拆成一个一个单词
                # >> > words = "abcde你好啊"
                # >> > list(words)
                # ['a', 'b', 'c', 'd', 'e', '你', '好', '啊']
                # >> >
                words = list(text)
                labels = cls._create_labels(line, words)
                lines.append(
                    {"words": words, "labels": labels, "sentence": text})
        return lines

    @classmethod
    def _create_labels(self, line, words):
        # label is a list of sentence level classes
        return line.get('label', None)


cls_processors = {
    "sentence": SentenceProcessor,
}
