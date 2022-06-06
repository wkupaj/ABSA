import csv
import os

import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_file(self, filename):
        '''
        read file
        '''
        data = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                assert len(row) == 6, "error on line {}. Found {} calls".format(row, len(row))
                data.append(row)

        return data


class AeProcessor(DataProcessor):
    """Processor for the Aspect Extraction ."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.csv")), data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.csv")), data_dir, "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.csv")), data_dir, "test")

    def _create_examples(self, lines, data_dir, set_type):
        examples = []

        d = {}
        for l in lines:
            if l[0] in d:
                d[l[0]][0].append(l[3])
            else:
                d[l[0]] = [[l[3]], l[0], l[5]]

        labels = self.get_labels(data_dir)
        dict_labels = {}
        for i, label in enumerate(labels):
            dict_labels[label] = i

        for i, entry in enumerate(d.values()):
            guid = "%s-%s-%s" % (set_type, i, entry[0])
            text_a = entry[2]
            text_b = None
            mask = [0] * len(labels)
            for j in entry[0]:
                mask[dict_labels[j]] = 1
            label = mask
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _get_labels(sentences):
        label_set = set([])
        l = []
        for t in sentences:
            l.append(t[3])
        label_set.update(l)
        return sorted(list(label_set))

    def get_labels(self, data_dir):
        label_set = set([])
        label_set.update(AeProcessor._get_labels(self._read_file(os.path.join(data_dir, "train.csv"))))
        label_set.update(AeProcessor._get_labels(self._read_file(os.path.join(data_dir, "valid.csv"))))
        label_set.update(AeProcessor._get_labels(self._read_file(os.path.join(data_dir, "test.csv"))))
        return sorted(list(label_set))


class AscProcessor(DataProcessor):
    """Processor for the Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.csv")), data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.csv")), data_dir, "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.csv")), data_dir, "test")

    def _create_examples(self, lines, data_dir, set_type):
        examples = []

        labels = self.get_labels(data_dir)
        dict_labels = {}
        for i, label in enumerate(labels):
            dict_labels[label] = i

        for i, entry in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, i, entry[0])
            text_a = entry[3]
            text_b = entry[5]
            mask = [0] * len(labels)
            mask[dict_labels[entry[4]]] = 1
            label = mask
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    @staticmethod
    def _get_labels(sentences):
        label_set = set([])
        l = []
        for t in sentences:
            l.append(t[4])
        label_set.update(l)
        return sorted(list(label_set))

    def get_labels(self, data_dir):
        label_set = set([])
        label_set.update(AscProcessor._get_labels(self._read_file(os.path.join(data_dir, "train.csv"))))
        label_set.update(AscProcessor._get_labels(self._read_file(os.path.join(data_dir, "valid.csv"))))
        label_set.update(AscProcessor._get_labels(self._read_file(os.path.join(data_dir, "test.csv"))))
        return sorted(list(label_set))


def convert_examples_to_features(examples, max_seq_length, encode_method):
    features = []
    for (ex_index, example) in enumerate(examples):


        labels = example.label
        token_ids = []

        if example.text_b is None:
            textlist = example.text_a.split(' ')
        else:
            textlist = example.text_a.split("#") + example.text_b.split(' ')

        for i, word in enumerate(textlist):
            tokens = encode_method(word.strip())
            token_ids.extend(tokens)

        if len(token_ids) >= max_seq_length - 1:
            token_ids = token_ids[0:(max_seq_length - 2)]

        input_mask = [1] * len(token_ids)

        while len(token_ids) < max_seq_length:
            token_ids.append(1)
            input_mask.append(0)

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(InputFeatures(input_ids=token_ids, input_mask=input_mask, label_id=labels))
    return features


def create_dataset(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_label_ids)
