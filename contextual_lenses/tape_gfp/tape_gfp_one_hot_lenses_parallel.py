"""Contextual lenses on TAPE fluoresence dataset."""


"""Parse command line arguments."""
from parser import parse_args
tpu_name, save_dir, restore_dir = parse_args()


"""Connect TPU to VM instance."""
from cloud_utils.tpu_init import connect_tpu
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

from train_utils import create_optimizer, p_train, \
create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import mse_loss

from tape.tape.datasets import LMDBDataset


"""Data processing."""
def dataset_to_df(in_name):
  dataset = LMDBDataset(in_name)
  df = pd.DataFrame(list(dataset)[:])
  df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
  return df


"""Padding."""
def pad_seq(seq, pad_char='-'):
  SEQ_LEN = 237
  padded_seq = seq + pad_char*(SEQ_LEN-len(seq))

  return padded_seq


"""Open train data and add one-hots."""
train_df = dataset_to_df('tape/fluorescence/fluorescence_train.lmdb')
train_df.primary.apply(len).describe()

train_df['padded_primary'] = train_df.primary.apply(lambda x: pad_seq(x))

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

test_df['padded_primary'] = test_df.primary.apply(lambda x: pad_seq(x))
test_df['one_hot_inds'] = test_df.padded_primary.apply(lambda x: residues_to_one_hot_inds(x))

test_fluorescences = test_df['log_fluorescence'].values


"""Data batching."""
def create_data_iterator(df, batch_size, epochs=1, buffer_size=None, seed=0, drop_remainder=False):

  if buffer_size is None:
    buffer_size = len(df)

  one_hot_inds = list(df['one_hot_inds'].values)
  one_hot_inds = tf.data.Dataset.from_tensor_slices(one_hot_inds)

  fluorescences = df['log_fluorescence'].values
  fluorescences = tf.data.Dataset.from_tensor_slices(fluorescences)

  batches = tf.data.Dataset.zip((one_hot_inds, fluorescences)).shuffle(buffer_size=buffer_size, seed=seed)
  batches = batches.repeat(epochs).batch(batch_size=batch_size, drop_remainder=drop_remainder).as_numpy_iterator()

  return batches


"""Model evaluation."""
def evaluate(predict_fn, test_data, true_fluorescences, plot_title):

  pred_fluorescences = []
  for batch in iter(test_data):
    X, Y = batch
    preds = predict_fn(X)
    for pred in preds:
      pred_fluorescences.append(pred[0])
  pred_fluorescences = np.array(pred_fluorescences)
  
  spearmanr = scipy.stats.spearmanr(true_fluorescences, pred_fluorescences).correlation
  mse = sklearn.metrics.mean_squared_error(true_fluorescences, pred_fluorescences)
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
  plt.scatter(dark_true_fluorescences, dark_pred_fluorescences)
  plt.xlabel('True Fluorescences')
  plt.ylabel('Predicted Fluorescences')
  dark_title = plot_title + ' (Dark)'
  plt.title(dark_title)
  plt.savefig('figures/' + dark_title.replace(' ', '').replace('+', ''))
  plt.show()

  results = {
      'title': plot_title,
      'spearmanr': spearmanr,
      'mse': mse,
      'bright_spearmanr': bright_spearmanr,
      'bright_mse': bright_mse,
      'dark_spearmanr': dark_spearmanr,
      'dark_mse': dark_mse,
  }

  return results, pred_fluorescences


"""One-hot + positional embeddings + CNN + GatedConv lens."""
epochs = 50
train_batches = create_data_iterator(train_df, batch_size=64*8, epochs=epochs, drop_remainder=True)
test_batches = create_data_iterator(test_df, batch_size=256, buffer_size=1)
lr = 1e-3
wd = 0.
encoder_fn = cnn_one_hot_pos_emb_encoder
encoder_fn_kwargs = {
    'n_layers': 1,
    'n_features': [512],
    'n_kernel_sizes': [12],
    'max_len': 512,
    'posemb_init': nn.initializers.normal(stddev=1e-6)
}
reduce_fn = gated_conv
reduce_fn_kwargs = {
    'rep_size': 256,
    'm_layers': 3,
    'm_features': [[512, 512], [512, 512]],
    'm_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
    'conv_rep_size': 256
}
loss_fn_kwargs = {

}
cnn_pos_emb_gated_conv_model = create_representation_model(encoder_fn=encoder_fn,
                                                           encoder_fn_kwargs=encoder_fn_kwargs,
                                                           reduce_fn=reduce_fn,
                                                           reduce_fn_kwargs=reduce_fn_kwargs,
                                                           num_categories=21,
                                                           output_features=1)

cnn_pos_emb_gated_conv_optimizer = p_train(model=cnn_pos_emb_gated_conv_model,
                                           train_data=train_batches,
                                           loss_fn=mse_loss,
                                           loss_fn_kwargs=loss_fn_kwargs,
                                           learning_rate=lr,
                                           weight_decay=wd,
                                           restore_dir=restore_dir,
                                           save_dir=save_dir)

del cnn_pos_emb_gated_conv_model

cnn_pos_emb_gated_conv_results, cnn_pos_emb_gated_conv_pred_fluorescences = \
  evaluate(predict_fn=cnn_pos_emb_gated_conv_optimizer.target,
           test_data=test_batches,
           true_fluorescences=test_fluorescences,
           plot_title='OneHot + PosEmb + CNN + GatedConv')

print(cnn_pos_emb_gated_conv_results)
