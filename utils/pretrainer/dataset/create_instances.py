import random
import numpy as np
import collections
import torch
import tensorflow as tf

import indexed_dataset

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("input_file_prefix", None,
                    "Input text/entity file.")
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
vocab_words_size = 14399

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


class TrainingInstance(object):
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.input_ids = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels


def create_training_instances(input_file, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    all_documents = []
    with tf.gfile.GFile(input_file + "_token", "r") as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = [int(x) for x in line.strip().split()]
            if line[0] != 0:
                all_documents.append(line)
            # for testing
            if len(all_documents) > 100:
                break
    rng.shuffle(all_documents)
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, rng))
    return instances


def jump_in_document(document, i):
    pos = 1
    while i > 0:
        pos = pos + 1 + document[pos]
        i -= 1
    return pos


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, rng):
    document = all_documents[document_index]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < document[0]:
        current_chunk.append(i)
        current_length += document[jump_in_document(document, i)]
        if i == document[0] - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in current_chunk[:a_end]:
                    pos = jump_in_document(document, j)
                    tokens_a.extend(document[pos + 1:pos + 1 + document[pos]])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, random_document[0] - 1)
                    for j in range(random_start, random_document[0]):
                        pos = jump_in_document(random_document, j)
                        tokens_b.extend(random_document[pos + 1:pos + 1 + random_document[pos]])
                        if len(tokens_b) >= target_b_length:
                            break

                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in current_chunk[a_end:]:
                        pos = jump_in_document(document, j)
                        tokens_b.extend(document[pos + 1:pos + 1 + document[pos]])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = [101] + tokens_a + [102] + tokens_b + [102]
                segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == 101 or token == 102:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None
        if rng.random() < 0.8:
            masked_token = 103  # [MASK]
        else:
            if rng.random() < 0.5:
                masked_token = tokens[index]
            else:
                masked_token = rng.randint(0, vocab_words_size - 1)
        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, output_file, vocab_file):
    # read vocab
    vocab_words = []
    with tf.gfile.GFile(vocab_file, 'r') as fin:
        for line in fin:
            vocab_words.append(line.strip())

    ds = indexed_dataset.IndexedDatasetBuilder(output_file + ".bin")
    for (inst_index, instance) in enumerate(instances):
        input_mask = [1] * len(instance.input_ids)
        segment_ids = list(instance.segment_ids)
        input_ids = list(instance.input_ids)
        assert len(input_ids) <= max_seq_length
        if len(input_ids) < max_seq_length:
            rest = max_seq_length - len(input_ids)
            input_ids.extend([0] * rest)
            input_mask.extend([0] * rest)
            segment_ids.extend([0] * rest)

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = list(instance.masked_lm_labels)
        masked_lm_labels = np.ones(len(input_ids), dtype=int) * -1
        masked_lm_labels[masked_lm_positions] = masked_lm_ids
        masked_lm_labels = list(masked_lm_labels)
        # masked_lm_labels[0] = -1

        next_sentence_label = 1 if instance.is_random_next else 0

        ds.add_item(torch.IntTensor(input_ids + input_mask + segment_ids + masked_lm_labels + [next_sentence_label]))

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [vocab_words[x] for x in instance.input_ids]))

            unmask = list(instance.input_ids)
            for i, x in enumerate(masked_lm_labels):
                if x != -1:
                    unmask[i] = x
            tf.logging.info("unmask_tokens: %s" % " ".join(
                [vocab_words[x] for x in unmask]))
            tf.logging.info("input_mask: %s" % " ".join(
                [str(x) for x in input_mask]))
            tf.logging.info("segment: %s" % " ".join(
                [str(x) for x in segment_ids]))
            tf.logging.info("next_sentence: %d" % next_sentence_label)

    ds.finalize(output_file + ".idx")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("*** Reading from input files ***")
    tf.logging.info("%s", FLAGS.input_file_prefix)

    rng = random.Random(FLAGS.random_seed)

    instances = create_training_instances(
        FLAGS.input_file_prefix, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    tf.logging.info("*** Writing to output files ***")
    tf.logging.info("%s", FLAGS.output_file)
    write_instance_to_example_files(instances, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, FLAGS.output_file, FLAGS.vocab_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file_prefix")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
