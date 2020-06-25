"""Contextual lenses on Pfam family classification dataset (first 100 families)."""


"""Connect TPU to VM instance."""
import argparse
from cloud_utils.tpu_init import connect_tpu
parser = argparse.ArgumentParser()
parser.add_argument('--tpu_name')
args = parser.parse_args()
tpu_name = args.tpu_name
connect_tpu(tpu_name=tpu_name)


import functools
import itertools
import os
import time

import flax
from flax import nn

import jax
import jax.numpy as jnp

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats

import sklearn.linear_model
import sklearn.metrics

import seaborn as sns

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import create_optimizer, train_step, \
train, create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import cross_entropy_loss


"""Data preprocessing
Code source: https://www.kaggle.com/drewbryant/starter-pfam-seed-random-split."""
data_partitions_dirpath = 'random_split/random_split/'
print('Available dataset partitions: ', os.listdir(data_partitions_dirpath))

def read_all_shards(partition='dev', data_dir=data_partitions_dirpath):
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    return pd.concat(shards)

test_df = read_all_shards('test')
dev_df = read_all_shards('dev')
train_df = read_all_shards('train')

partitions = {'test': test_df, 'dev': dev_df, 'train': train_df}
for name, df in partitions.items():
    print('Dataset partition "%s" has %d sequences' % (name, len(df)))

def mod_family_accession(family_accession):
  return family_accession[:family_accession.index('.')]

train_df['mod_family_accession'] = train_df.family_accession.apply(lambda x: mod_family_accession(x))
test_df['mod_family_accession'] = test_df.family_accession.apply(lambda x: mod_family_accession(x))
dev_df['mod_family_accession'] = dev_df.family_accession.apply(lambda x: mod_family_accession(x))

kept_family_accessions = []
for i in range(1, 101):
  kept_family_accessions.append('PF00' + '0'*(3-len(str(i))) + str(i))

def keep_family_accessions(df, kept_family_accessions):
  
  dfs_by_family_accessions = []
  
  for family_accession in kept_family_accessions:
    df_by_family_accession = df[df['mod_family_accession']==family_accession]
    dfs_by_family_accessions.append(df_by_family_accession)
  
  return pd.concat(dfs_by_family_accessions)

train_df = keep_family_accessions(train_df, kept_family_accessions)
test_df = keep_family_accessions(test_df, kept_family_accessions)
dev_df = keep_family_accessions(dev_df, kept_family_accessions)

family_ids = sorted(set(train_df['family_id'].values))
family_id_to_index = {}
for i, family_id in enumerate(family_ids):
  family_id_to_index[family_id] = i

train_df['index'] = train_df.family_id.apply(lambda x: family_id_to_index[x])
test_df['index'] = test_df.family_id.apply(lambda x: family_id_to_index[x])
dev_df['index'] = dev_df.family_id.apply(lambda x: family_id_to_index[x])

test_indexes = test_df['index'].values

num_families = len(set(train_df['family_id']))

def pad_seq(seq, pad_char='-'):
  SEQ_LEN = 512
  seq = seq[:SEQ_LEN]
  padded_seq = seq + pad_char*(SEQ_LEN-len(seq))
  return padded_seq

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'X', 'U', 'B', 'O', 'Z', '-'
]
num_categories = len(AMINO_ACID_VOCABULARY)

def residues_to_one_hot_inds(amino_acid_residues):
  return np.array([AMINO_ACID_VOCABULARY.index(char) for char in amino_acid_residues])

train_df['one_hot_inds'] = train_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))
test_df['one_hot_inds'] = test_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))
dev_df['one_hot_inds'] = dev_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))

print(train_df.head())


"""Data batching."""
def create_data_iterator(df, batch_size, epochs=1, buffer_size=None, seed=0):

  if buffer_size is None:
    buffer_size = len(df)

  one_hot_inds = list(df['one_hot_inds'].values)
  one_hot_inds = tf.data.Dataset.from_tensor_slices(one_hot_inds)

  indexes = df['index'].values
  indexes = tf.data.Dataset.from_tensor_slices(indexes)

  batches = tf.data.Dataset.zip((one_hot_inds, indexes)).batch(train_bs)
  batches = batches.shuffle(buffer_size=buffer_size, seed=seed).repeat(epochs).as_numpy_iterator()

  return batches


"""Model evaluation."""
def evaluate(predict_fn, test_data, true_indexes, title, loss_fn_kwargs):

  pred_indexes = []
  cross_entropy = 0.
  
  for batch in iter(test_data):

    X, Y = batch

    Y_hat = predict_fn(X)

    cross_entropy += cross_entropy_loss(Y, Y_hat, **loss_fn_kwargs)

    preds = jnp.argmax(jax.nn.log_softmax(Y_hat), axis=1)
    for pred in preds:
      pred_indexes.append(pred)
  pred_indexes = np.array(pred_indexes)
  
  acc = sklearn.metrics.accuracy_score(true_indexes, pred_indexes)
  
  results = {
      'title': title,
      'cross_entropy': cross_entropy,
      'accuracy': acc,
  }

  return results, preds


"""Experiments."""
epochs = 100
train_batches = create_data_iterator(train_df, batch_size=512, epochs=epochs)
test_batches = create_data_iterator(test_df, batch_size=512, buffer_size=1)
lr = 1e-3
wd = 0.
encoder_fn = cnn_one_hot_encoder
encoder_fn_kwargs = {
    'n_layers': 1,
    'n_features': [512],
    'n_kernel_sizes': [12]
}
reduce_fn = max_pool
reduce_fn_kwargs = {
    
}
loss_fn_kwargs = {
  'num_classes': num_families
}
cnn_max_pool_model = create_representation_model(encoder_fn=encoder_fn,
                                                  encoder_fn_kwargs=encoder_fn_kwargs,
                                                  reduce_fn=reduce_fn,
                                                  reduce_fn_kwargs=reduce_fn_kwargs,
                                                  num_categories=num_categories,
                                                  output_features=num_families)
cnn_max_pool_optimizer = train(model=cnn_max_pool_model,
                               train_data=train_batches,
                               loss_fn=cross_entropy_loss,
                               loss_fn_kwargs=loss_fn_kwargs,
                               learning_rate=lr,
                               weight_decay=wd)
del cnn_max_pool_model
cnn_max_pool_results, cnn_max_pool_preds = evaluate(predict_fn=cnn_max_pool_optimizer.target,
                                                    test_data=test_batches,
                                                    true_indexes=test_indexes,
                                                    title='CNN + Max Pool',
                                                    loss_fn_kwargs=loss_fn_kwargs)
print(cnn_max_pool_results)
