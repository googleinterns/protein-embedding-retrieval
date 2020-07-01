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


def create_pfam_df(family_accessions, test=False):
  """Processes Pfam data into a featurized dataframe."""

  if test:
    pfam_df = read_all_shards('test')
  else:
    pfam_df = read_all_shards('train')

  pfam_df['mod_family_accession'] = pfam_df.family_accession.apply(lambda x: mod_family_accession(x))
  pfam_df = pfam_df[pfam_df.mod_family_accession.isin(family_accessions)]
  pfam_df['index'] = pfam_df.family_id.apply(lambda x: family_id_to_index[x])
  pfam_df['one_hot_inds'] = pfam_df.sequence.apply(lambda x: residues_to_one_hot_inds(pad_seq(x)))

  return pfam_df


def create_pfam_batches(family_accessions, batch_size, test=False, samples=None, epochs=1,
                        drop_remainder=False, buffer_size=None, seed=0, random_state=0):
  """Creates iterable object of Pfam data batches."""

  pfam_df = create_pfam_df(family_accessions, test=test)
    
  if samples is not None:
    pfam_df = pfam_df.sample(frac=1, replace=False, random_state=random_state)
    pfam_df = pfam_df.groupby('mod_family_accession').head(samples).reset_index()
  
  pfam_indexes = pfam_df['index'].values

  pfam_batches = create_data_iterator(df=pfam_df, input_col='one_hot_inds', output_col='index',
	  								  batch_size=batch_size, epochs=epochs, buffer_size=buffer_size, 
	  								  seed=seed, drop_remainder=drop_remainder)

  return pfam_batches, pfam_indexes


# Model evaluation.
def pfam_evaluate(predict_fn, test_family_accessions, title, loss_fn_kwargs, batch_size=512):
  """Computes predicted family ids and measures performance in cross entropy and accuracy."""

  test_batches, test_indexes = create_pfam_batches(family_accessions=test_family_accessions, 
                                                   batch_size=batch_size, test=True, buffer_size=1)

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


def pfam_nearest_neighbors_classification(encoder, train_family_accessions, test_family_accessions, batch_size=512, 
                                          n_neighbors=1, train_samples=None, test_samples=None, restricted=False):
  """Nearest neighbors classification on Pfam families using specified encoder."""

  train_batches, train_indexes = create_pfam_batches(family_accessions=train_family_accessions, batch_size=batch_size,
                                                     samples=train_samples, buffer_size=1)
  test_batches, test_indexes = create_pfam_batches(family_accessions=test_family_accessions, batch_size=batch_size, 
                                                   test=True, samples=test_samples, buffer_size=1)

  train_vectors = compute_embeddings(encoder, train_batches)
  test_vectors = compute_embeddings(encoder, test_batches)

  knn_classifier = knn(n_neighbors=n_neighbors)
  knn_classifier.fit(train_vectors, train_indexes)
  knn_predictions = knn_classifier.predict(test_vectors)

  knn_accuracy = sklearn.metrics.accuracy_score(test_indexes, knn_predictions)

  results = {
    str(n_neighbors)+"-nn accuracy": knn_accuracy,
    'train_samples': train_samples,
    'test_samples': test_samples
  }

  return results, knn_predictions, knn_classifier

