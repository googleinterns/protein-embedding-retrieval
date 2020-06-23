"""Contextual lenses on TAPE fluoresence dataset."""


"""Connect TPU to VM instance."""
import argparse
from cloud_utils.tpu_init import connect_tpu
parser = argparse.ArgumentParser()
parser.add_argument('--tpu_name')
args = parser.parse_args()
tpu_name = args.tpu_name
connect_tpu(tpu_name=tpu_name)


import jax
import jax.numpy as jnp

import flax
from flax import nn

import numpy as np

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

from loss_fns import mse_loss

from tape.tape.datasets import LMDBDataset


"""Data processing."""
def dataset_to_df(in_name):
  dataset = LMDBDataset(in_name)
  df = pd.DataFrame(list(dataset)[:])
  df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
  df['binary_log_f'] = [int(b) for b in df['log_fluorescence'] > 2.5]
  return df


"""Padding."""
SEQ_LEN = 237
def pad_seq(seq, pad_char='-'):
  
  padded_seq = seq + pad_char*(SEQ_LEN-len(seq))
  padding_mask = []
  for _ in range(len(seq)):
    padding_mask.append([1])
  for _ in range(SEQ_LEN-len(seq)):
    padding_mask.append([0])
  padding_mask = np.array(padding_mask)

  return padded_seq, padding_mask


"""Open train data and add one-hots."""
train_df = dataset_to_df('tape/fluorescence/fluorescence_train.lmdb')
train_df.primary.apply(len).describe()

train_df['padded_primary'] = train_df.primary.apply(lambda x: pad_seq(x)[0])
train_df['padding_masks'] = train_df.primary.apply(lambda x: pad_seq(x)[1])
train_df.padded_primary.apply(len).describe()

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', '-'
]


def residues_to_one_hot_inds(amino_acid_residues):
  return np.array([AMINO_ACID_VOCABULARY.index(char) for char in amino_acid_residues])

train_df['one_hot_inds'] = train_df.padded_primary.apply(lambda x: residues_to_one_hot_inds(x))

print(train_df.head())


"""Open test data and add one-hots."""
test_df = dataset_to_df('tape/fluorescence/fluorescence_test.lmdb')
test_df.primary.apply(len).describe()

test_df['padded_primary'] = test_df.primary.apply(lambda x: pad_seq(x)[0])
test_df['padding_masks'] = test_df.primary.apply(lambda x: pad_seq(x)[1])
test_df.padded_primary.apply(len).describe()

test_df['one_hot_inds'] = test_df.padded_primary.apply(lambda x: residues_to_one_hot_inds(x))

print(test_df.head())


"""Data batching."""
def create_train_data(epochs, train_bs=256):
  train_one_hot_inds = list(train_df['one_hot_inds'].values)
  N = len(train_one_hot_inds)
  train_one_hot_inds = tf.data.Dataset.from_tensor_slices(train_one_hot_inds)

  train_fluorescences = train_df['log_fluorescence'].values
  train_fluorescences = tf.data.Dataset.from_tensor_slices(train_fluorescences)

  train_batches = tf.data.Dataset.zip((train_one_hot_inds, train_fluorescences))
  train_batches = train_batches.batch(train_bs).shuffle(buffer_size=N, seed=0).repeat(epochs).as_numpy_iterator()

  return train_batches


def create_test_data(epochs=1, test_bs=256):
  test_one_hot_inds = list(test_df['one_hot_inds'].values)
  test_one_hot_inds = tf.data.Dataset.from_tensor_slices(test_one_hot_inds)

  test_fluorescences = test_df['log_fluorescence'].values
  test_fluorescences = tf.data.Dataset.from_tensor_slices(test_fluorescences)

  test_batches = tf.data.Dataset.zip((test_one_hot_inds, test_fluorescences))
  test_batches = test_batches.batch(test_bs).repeat(epochs).as_numpy_iterator()

  return test_batches

test_fluorescences = test_df['log_fluorescence'].values


"""Model evaluation."""
def evaluate(optimizer, test_data, true_fluorescences, plot_title):

  pred_fluorescences = []
  for batch in iter(test_data):
    X, Y = batch
    preds = optimizer.target(X)
    for pred in preds:
      pred_fluorescences.append(pred[0])
  pred_fluorescences = np.array(pred_fluorescences)
  
  spearmanr = scipy.stats.spearmanr(true_fluorescences, pred_fluorescences).correlation
  mse = sklearn.metrics.mean_squared_error(true_fluorescences, pred_fluorescences)
  print("Full data set:")
  print("spearmanr: {}\nmse: {}".format(spearmanr, mse))
  print()
  plt.scatter(true_fluorescences, pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  plt.title(plot_title)
  plt.savefig('figures/' + plot_title.replace(' ', '').replace('+', ''))
  plt.show()

  bright_inds = np.where(true_fluorescences > 2.5)
  bright_true_fluorescences = true_fluorescences[bright_inds]
  bright_pred_fluorescences = pred_fluorescences[bright_inds]
  bright_spearmanr = scipy.stats.spearmanr(bright_true_fluorescences, bright_pred_fluorescences).correlation
  bright_mse = sklearn.metrics.mean_squared_error(bright_true_fluorescences, bright_pred_fluorescences)
  print("Bright mode only:")
  print("spearmanr: {}\nmse: {}".format(bright_spearmanr, bright_mse))
  print()
  plt.scatter(bright_true_fluorescences, bright_pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  bright_title = plot_title + ' (Bright)'
  plt.title(bright_title)
  plt.savefig('figures/' + bright_title.replace(' ', '').replace('+', ''))
  plt.show()

  dark_inds = np.where(true_fluorescences < 2.5)
  dark_true_fluorescences = true_fluorescences[dark_inds]
  dark_pred_fluorescences = pred_fluorescences[dark_inds]
  dark_spearmanr = scipy.stats.spearmanr(dark_true_fluorescences, dark_pred_fluorescences).correlation
  dark_mse = sklearn.metrics.mean_squared_error(dark_true_fluorescences, dark_pred_fluorescences)
  print("Dark mode only:")
  print("spearmanr: {}\nmse: {}".format(dark_spearmanr, dark_mse))
  print()
  plt.scatter(dark_true_fluorescences, dark_pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  dark_title = plot_title + ' (Dark)'
  plt.title(dark_title)
  plt.savefig('figures/' + dark_title.replace(' ', '').replace('+', ''))
  plt.show()

  results = {
      'spearmanr': spearmanr,
      'mse': mse,
      'pred_fluorescences': pred_fluorescences,
      'bright_spearmanr': bright_spearmanr,
      'bright_mse': bright_mse,
      'bright_pred_fluorescences': bright_pred_fluorescences,
      'dark_spearmanr': dark_spearmanr,
      'dark_mse': dark_mse,
      'dark_pred_fluorescences': dark_pred_fluorescences
  }

  return results


"""One-hot + positional embeddings + CNN + GatedConv lens."""
epochs = 50
train_batches = create_train_data(epochs, train_bs=256)
test_batches = create_test_data(test_bs=256)

lr = 1e-3
wd = 0.

encoder_fn = cnn_one_hot_pos_emb_encoder
encoder_fn_kwargs = {
    'N_layers': 1,
    'N_features': [512],
    'N_kernel_sizes': [12],
    'max_len': 512,
    'posemb_init': nn.initializers.normal(stddev=1e-6)
}
reduce_fn = gated_conv
reduce_fn_kwargs = {
    'rep_size': 256,
    'M_layers': 3,
    'M_features': [[512, 512], [512, 512]],
    'M_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
    'zero_rep_size': 256
}

cnn_pos_emb_gated_conv_model = create_representation_model(encoder_fn=encoder_fn,
                                                           encoder_fn_kwargs=encoder_fn_kwargs,
                                                           reduce_fn=reduce_fn,
                                                           reduce_fn_kwargs=reduce_fn_kwargs)

cnn_pos_emb_gated_conv_optimizer = train(model=cnn_pos_emb_gated_conv_model,
                                         train_data=train_batches,
                                         loss_fn=mse_loss,
                                         learning_rate=lr,
                                         weight_decay=wd)

del cnn_pos_emb_gated_conv_model

cnn_pos_emb_gated_conv_results = evaluate(optimizer=cnn_pos_emb_gated_conv_optimizer,
                                          test_data=test_batches,
                                          true_fluorescences=test_fluorescences,
                                          plot_title='OneHot + PosEmb + CNN + GatedConv')
