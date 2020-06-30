"""Utils for Pfam family classification experiments."""

import os 

import jax
import jax.numpy as jnp

import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats

import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier as knn

from train_utils import create_data_iterator

from loss_fns import cross_entropy_loss


# Data preprocessing.
# Code source: https://www.kaggle.com/drewbryant/starter-pfam-seed-random-split.
data_partitions_dirpath = 'random_split/random_split/'

def read_all_shards(partition='train', data_dir=data_partitions_dirpath):
  """Combines different CSVs into a single dataframe."""

  shards = []
  for fn in os.listdir(os.path.join(data_dir, partition)):
    with open(os.path.join(data_dir, partition, fn)) as f:
      shards.append(pd.read_csv(f, index_col=None))
  return pd.concat(shards)

def mod_family_accession(family_accession):
  """Reduces family accession to everything prior to '.'."""

  return family_accession[:family_accession.index('.')]

# Padding.
def pad_seq(seq, pad_char='-'):
  """Pads all sequenes to a length of 237."""

  SEQ_LEN = 512
  seq = seq[:SEQ_LEN]
  padded_seq = seq + pad_char*(SEQ_LEN-len(seq))
  return padded_seq

# Add one-hots.
AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'X', 'U', 'B', 'O', 'Z', '-'
]
def residues_to_one_hot_inds(amino_acid_residues):
  """Converts amino acid residues to one hot indices."""

  one_hot_inds = np.array([AMINO_ACID_VOCABULARY.index(char) for char in amino_acid_residues])

  return one_hot_inds

# Dictionary mapping family id to index.
family_ids = open('pfam_family_ids.txt', 'r').readlines()
family_id_to_index = {}
for i, family_id in enumerate(family_ids):
  family_id_to_index[family_id.replace('\n', '')] = i

def create_train_df(train_family_accessions):
  """Processes train data into a featurized dataframe."""

  train_df = read_all_shards('train')
  train_df['mod_family_accession'] = train_df.family_accession.apply(lambda x: mod_family_accession(x))
  train_df = train_df[train_df.mod_family_accession.isin(train_family_accessions)]
  train_df['index'] = train_df.family_id.apply(lambda x: family_id_to_index[x])
  train_df['one_hot_inds'] = train_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))

  return train_df

def create_test_df(test_family_accessions):
  """Processes test data into a featurized dataframe."""

  test_df = read_all_shards('test')
  test_df['mod_family_accession'] = test_df.family_accession.apply(lambda x: mod_family_accession(x))
  test_df = test_df[test_df.mod_family_accession.isin(test_family_accessions)]
  test_df['index'] = test_df.family_id.apply(lambda x: family_id_to_index[x])
  test_df['one_hot_inds'] = test_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))

  return test_df

def create_train_batches(train_family_accessions, batch_size, epochs=1, buffer_size=None, seed=0, drop_remainder=False):
  """Creates iterable object of train batches."""

  train_df = create_train_df(train_family_accessions)

  train_batches = create_data_iterator(df=train_df, input_col='one_hot_inds', output_col='index',
	  								   batch_size=batch_size, epochs=epochs, buffer_size=buffer_size, 
	  								   seed=seed, drop_remainder=drop_remainder)

  return train_batches


def create_test_batches(test_family_accessions, batch_size, epochs=1, buffer_size=1, seed=0, drop_remainder=False):
  """Creates iterable object of test batches."""

  test_df = create_test_df(test_family_accessions)

  test_batches = create_data_iterator(df=test_df, input_col='one_hot_inds', output_col='index',
	  								  batch_size=batch_size, epochs=epochs, buffer_size=buffer_size,
	  								  seed=seed, drop_remainder=drop_remainder)

  return test_batches


# Model evaluation.
def evaluate(predict_fn, test_family_accessions, title, loss_fn_kwargs, batch_size=512):
  """Computes predicted family ids and measures performance in cross entropy and accuracy."""

  test_df = create_test_df(test_family_accessions)
  test_indexes = test_df['index'].values
  test_batches = create_test_batches(test_family_accessions=test_family_accessions, batch_size=batch_size)

  pred_indexes = []
  cross_entropy = 0.
  
  for batch in iter(test_batches):

    X, Y = batch

    Y_hat = predict_fn(X)

    cross_entropy += cross_entropy_loss(Y, Y_hat, **loss_fn_kwargs)

    preds = jnp.argmax(Y_hat, axis=1)
    for pred in preds:
      pred_indexes.append(pred)
  
  pred_indexes = np.array(pred_indexes)
  
  acc = sklearn.metrics.accuracy_score(test_indexes, pred_indexes)
  
  results = {
      'title': title,
      'cross_entropy': cross_entropy,
      'accuracy': acc,
  }

  return results, pred_indexes


def compute_embeddings(encoder, data_batches):
  """Computes sequence embeddings according to a specified encoder."""

  vectors = []
  for batch in iter(data_batches):
    X, Y = batch
    X_embedded = encoder(X)
    for vec in np.array(X_embedded):
      vectors.append(vec)
  vectors = np.array(vectors)

  return vectors


def pfam_nearest_neighbors_classification(encoder, train_family_accessions, test_family_accessions, 
                                          batch_size=512, n_neighbors=1):
  """Nearest neighbors classification on Pfam families using specified encoder."""

  train_df = create_train_df(train_family_accessions)
  train_indexes = train_df['index'].values
  train_batches = create_train_batches(train_family_accessions, batch_size=batch_size, buffer_size=1)
  
  test_df = create_test_df(test_family_accessions)
  test_indexes = test_df['index'].values
  test_batches = create_test_batches(test_family_accessions, batch_size=batch_size)

  train_vectors = compute_embeddings(encoder, train_batches)
  test_vectors = compute_embeddings(encoder, test_batches)

  knn_classifier = knn(n_neighbors=n_neighbors)
  knn_classifier.fit(train_vectors, train_indexes)
  knn_predictions = knn_classifier.predict(test_vectors)

  knn_accuracy = sklearn.metrics.accuracy_score(test_indexes, knn_predictions)

  results = {
    str(n_neighbors)+"-nn accuracy": knn_accuracy
  }

  return results, knn_predictions, knn_classifier

