# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

## Required parameters
data_dir = '../data/snips'
log_dir = '../logs/snips-intent'
bert_model_dir = '../checkpoint/uncased_L-12_H-768_A-12'
bert_config_file = os.path.join(bert_model_dir, 'bert_config.json')
vocab_file = os.path.join(bert_model_dir, 'vocab.txt')
init_checkpoint = os.path.join(bert_model_dir, 'bert_model.ckpt')

## Other parameters
do_lower_case = True
do_train = True
do_eval = True
do_predict = True
max_seq_length = 46
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 3
warmup_proportion = 0.1
save_checkpoints_steps = 1000
log_step_count_steps = 10
save_summary_steps = 1


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


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Processor for the ATIS data set."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_path = os.path.join(self.data_dir, "train.tsv")
        self.dev_path = os.path.join(self.data_dir, "dev.tsv")
        self.test_path = os.path.join(self.data_dir, "test.tsv")

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_examples(self):
        return self._create_examples(self._read_tsv(self.train_path), "train")

    def get_dev_examples(self):
        return self._create_examples(self._read_tsv(self.dev_path), "dev")

    def get_test_examples(self):
        return self._create_examples(self._read_tsv(self.test_path), "test")

    def get_labels_info(self):
        labels = []
        label_map = {}
        label_map_file = os.path.join(log_dir, "label_map.txt")
        lines = self._read_tsv(self.train_path) + \
                self._read_tsv(self.dev_path) + \
                self._read_tsv(self.test_path)

        for line in lines:
            labels += line[0].strip().split()

        labels = sorted(set(labels), reverse=True)
        num_labels = sorted(set(labels), reverse=True).__len__()

        with tf.gfile.GFile(label_map_file, "w") as writer:
            for (i, label) in enumerate(labels):
                label_map[label] = i
                writer.write("{}:{}\n".format(i, label))
        return label_map, num_labels


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(examples, label_map, max_seq_length,
                                            tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder,
                                batch_size):
    """Creates an `input_fn` closure to be passed to Estimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        return example

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids)

    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels)

        predicted_class = tf.argmax(logits, axis=-1, output_type=tf.int32)

        accuracy = tf.metrics.accuracy(labels=label_ids,
                                       predictions=predicted_class,
                                       weights=is_real_example, name="acc_op")
        tf.summary.scalar("accuracy", accuracy[1])

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op
            )
        elif mode == tf.estimator.ModeKeys.EVAL:

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops={"accuracy": accuracy}
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"predicted_class": predicted_class,
                             "true_class": label_ids}
            )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not do_train and not do_eval and not do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(log_dir)
    processor = DataProcessor(data_dir)
    label_map, num_labels = processor.get_labels_info()

    tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
        model_dir=log_dir,
        session_config=config,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=log_step_count_steps,
        save_summary_steps=save_summary_steps
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    # Training
    if do_train:
        train_file = os.path.join(log_dir, "train.tf_record")
        if not tf.gfile.Exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, label_map, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=False,
            batch_size=train_batch_size
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # Evaluate
    if do_eval:
        eval_examples = processor.get_dev_examples()
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(log_dir, "eval.tf_record")
        if not tf.gfile.Exists(eval_file):
            file_based_convert_examples_to_features(
                eval_examples, label_map, max_seq_length, tokenizer, eval_file)
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=eval_batch_size
        )

        result = estimator.evaluate(input_fn=eval_input_fn)

        output_eval_file = os.path.join(log_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    # Predict
    if do_predict:

        predict_examples = processor.get_test_examples()
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(log_dir, "predict.tf_record")
        if not tf.gfile.Exists(predict_file):
            file_based_convert_examples_to_features(
                predict_examples, label_map, max_seq_length, tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=False,
            batch_size=predict_batch_size
        )

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(log_dir, "test_results.tsv")
        label_map_new = {v: k for k, v in label_map.items()}
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            writer.write("line, true, predicted \n")
            tf.logging.info("***** Predict results *****")

            true_classes = []
            predicted_classes = []

            for i, item in enumerate(result):
                true_class = item["true_class"]
                predicted_class = item["predicted_class"]

                true_classes.append(true_class)
                predicted_classes.append(predicted_class)

                if predicted_class != true_class:
                    output_line = "{}, {}, {}\n".format(i + 1, label_map_new[true_class],
                                                        label_map_new[predicted_class])
                    writer.write(output_line)
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)
        tf.logging.info("Test Accuracy: %s", accuracy)


if __name__ == "__main__":
    tf.app.run()
