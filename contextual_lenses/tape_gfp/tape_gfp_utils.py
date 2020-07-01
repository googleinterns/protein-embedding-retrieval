"""Utils for TAPE GFP experiments."""


import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats

import sklearn.metrics

from train_utils import create_data_iterator

from tape.tape.datasets import LMDBDataset


# Data processing.
def dataset_to_df(in_name):
  """Uses TAPE LMDBDataset module to convert dataset to pandas dataframe."""

  dataset = LMDBDataset(in_name)
  
  df = pd.DataFrame(list(dataset)[:])
  df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
  
  return df


# Padding.
def pad_seq(seq, pad_char='-'):
  """Pads all sequenes to a length of 237."""

  SEQ_LEN = 237
  padded_seq = seq + pad_char*(SEQ_LEN-len(seq))

  return padded_seq

# Open train/test data and add one-hots.
AMINO_ACID_VOCABULARY = [
	    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
	    'S', 'T', 'V', 'W', 'Y', '-'
]
def residues_to_one_hot_inds(amino_acid_residues):
  """Converts amino acid residues to one hot indices."""

  one_hot_inds = np.array([AMINO_ACID_VOCABULARY.index(char) for char in amino_acid_residues])
  
  return one_hot_inds


def create_gfp_df(test=False):
  """Processes GFP data into a featurized dataframe."""
  
  if test:
    gfp_df = dataset_to_df('tape/fluorescence/fluorescence_test.lmdb')
  else:
    gfp_df = dataset_to_df('tape/fluorescence/fluorescence_train.lmdb')
  
  gfp_df['padded_primary'] = gfp_df.primary.apply(lambda x: pad_seq(x))
  gfp_df['one_hot_inds'] = gfp_df.padded_primary.apply(lambda x: residues_to_one_hot_inds(x))

  return gfp_df


def create_gfp_batches(batch_size, epochs=1, test=False, buffer_size=None, seed=0, drop_remainder=False):
  """Creates iterable object of GFP batches."""
  
  if test:
    buffer_size = 1
  
  gfp_df = create_gfp_df(test=test)
    
  fluorescences = gfp_df['log_fluorescence'].values

  gfp_batches = create_data_iterator(df=gfp_df, input_col='one_hot_inds', output_col='log_fluorescence',
	  								 batch_size=batch_size, epochs=epochs, buffer_size=buffer_size, 
	  								 seed=seed, drop_remainder=drop_remainder)

  return gfp_batches, fluorescences


# Model evaluation.
def gfp_evaluate(predict_fn, title, batch_size=256):
  """Computes predicted fluorescences and measures performance in MSE and spearman correlation."""
  
  test_batches, test_fluorescences = create_gfp_batches(batch_size=batch_size, test=True, buffer_size=1)

  pred_fluorescences = []
  for batch in iter(test_batches):
    X, Y = batch
    preds = predict_fn(X)
    for pred in preds:
      pred_fluorescences.append(pred[0])
  pred_fluorescences = np.array(pred_fluorescences)
  
  spearmanr = scipy.stats.spearmanr(test_fluorescences, pred_fluorescences).correlation
  mse = sklearn.metrics.mean_squared_error(test_fluorescences, pred_fluorescences)
  plt.scatter(test_fluorescences, pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  plt.title(title)
  plt.savefig('figures/' + title.replace(' ', '').replace('+', ''))
  plt.show()

  bright_inds = np.where(test_fluorescences > 2.5)
  bright_test_fluorescences = test_fluorescences[bright_inds]
  bright_pred_fluorescences = pred_fluorescences[bright_inds]
  bright_spearmanr = scipy.stats.spearmanr(bright_test_fluorescences, bright_pred_fluorescences).correlation
  bright_mse = sklearn.metrics.mean_squared_error(bright_test_fluorescences, bright_pred_fluorescences)
  plt.scatter(bright_test_fluorescences, bright_pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  bright_title = title + ' (Bright)'
  plt.title(bright_title)
  plt.savefig('figures/' + bright_title.replace(' ', '').replace('+', ''))
  plt.show()

  dark_inds = np.where(test_fluorescences < 2.5)
  dark_test_fluorescences = test_fluorescences[dark_inds]
  dark_pred_fluorescences = pred_fluorescences[dark_inds]
  dark_spearmanr = scipy.stats.spearmanr(dark_test_fluorescences, dark_pred_fluorescences).correlation
  dark_mse = sklearn.metrics.mean_squared_error(dark_test_fluorescences, dark_pred_fluorescences)
  plt.scatter(dark_test_fluorescences, dark_pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  dark_title = title + ' (Dark)'
  plt.title(dark_title)
  plt.savefig('figures/' + dark_title.replace(' ', '').replace('+', ''))
  plt.show()

  results = {
      'title': title,
      'spearmanr': spearmanr,
      'mse': mse,
      'bright_spearmanr': bright_spearmanr,
      'bright_mse': bright_mse,
      'dark_spearmanr': dark_spearmanr,
      'dark_mse': dark_mse,
  }

  return results, pred_fluorescences

