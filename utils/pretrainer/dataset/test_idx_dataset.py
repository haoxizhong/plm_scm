import numpy as np
import indexed_dataset
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
import iterators
import tensorflow as tf

max_seq = 256

dataset = indexed_dataset.IndexedDataset('output/data_00')
print(len(dataset))
train_sampler = BatchSampler(RandomSampler(dataset), 1, True)

vocab_words = []
with tf.gfile.GFile("vocab.txt", 'r') as fin:
    for line in fin:
        vocab_words.append(line.strip())
d = {}
with tf.gfile.GFile("entity2id.txt", 'r') as fin:
    fin.readline()
    for line in fin:
        ent, idx = line.split()
        d[int(idx)] = ent


def collate_fn(x):
    x = torch.LongTensor([xx.numpy() for xx in x]) - 1
    entity_idx = x[:, 4 * max_seq:5 * max_seq]
    # ent_emb = embed(entity_idx+1) # -1 -> 0
    mask = x[:, 5 * max_seq:6 * max_seq]
    idx = np.unique(entity_idx.numpy())
    tf.logging.info(idx)
    return x[:, :max_seq], x[:, max_seq:2 * max_seq], x[:, 2 * max_seq:3 * max_seq], x[:,
                                                                                     3 * max_seq:4 * max_seq], mask, x[
                                                                                                                     :,
                                                                                                                     6 * max_seq:]


iterator = iterators.EpochBatchIterator(dataset, collate_fn, train_sampler)
tf.logging.set_verbosity(tf.logging.INFO)

from collections import defaultdict

d = defaultdict(int)

for i, batch in enumerate(iterator.next_epoch_itr()):
    if i == 10:
        break
    continue
    cnt = 0
    input_ids, input_mask, labels, entity, entity_mask = batch[0]
    input_ids = list(input_ids.numpy() - 1)
    input_mask = list(input_mask.numpy() - 1)
    labels = list(labels.numpy() - 1)
    entity = list(entity.numpy() - 1)
    entity_mask = list(entity_mask.numpy() - 1)
    # tf.logging.info("Masked Input: "+" ".join([vocab_words[x] for x in input_ids]))
    # tf.logging.info("Mask: "+" ".join([str(x) for x in input_mask]))
    for i, x in enumerate(labels):
        if x != -1:
            input_ids[i] = x
    tf.logging.info("Input: " + " ".join([vocab_words[x] for x in input_ids]))
    # mark = False
    for ent in entity:
        if ent != -1:
            cnt += 1
            # tf.logging.info(d[ent])
            # mark = True
    # if mark:
    #    tf.logging.info("Input: "+" ".join([vocab_words[x] for x in input_ids]))
    #    break
    # d[cnt] += 1
    # assert len(input_ids+input_mask+labels+entity+entity_mask) == 1280

exit(0)

tf.logging.info(sum([k * v for k, v in d.items()]))

res = sorted(d.items(), key=lambda x: x[0])

with open("distrib.txt", "w") as fout:
    for k, v in res:
        fout.write("{}\t{}\n".format(k, v))
