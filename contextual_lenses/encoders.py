"""Encoder functions

Fixed and learnable transformations for embedding sequences.
"""


import flax
from flax import nn

import jax
from jax import lax
import jax.nn
import jax.numpy as jnp

import numpy as np

from operator import itemgetter


def one_hot_encoder(batch_inds, num_categories):
  """Applies one-hot encoding from jax.nn."""

  one_hots = jax.nn.one_hot(batch_inds, num_classes=num_categories)
  
  return one_hots


class CNN(nn.Module):
  """A simple 1D CNN model."""

  def apply(self, x, n_layers, n_features, n_kernel_sizes):
    
    x = jnp.expand_dims(x, axis=2)

    for layer in range(n_layers):
      features = n_features[layer]
      kernel_size = (n_kernel_sizes[layer], 1)
      x = nn.Conv(x, features=features, kernel_size=kernel_size)
      x = nn.relu(x)
    
    x = jnp.squeeze(x, axis=2)

    return x


def cnn_one_hot_encoder(batch_inds, num_categories, n_layers, n_features, n_kernel_sizes):
  """Applies one-hot encoding followed by 1D CNN."""

  one_hots = one_hot_encoder(batch_inds, num_categories)
  cnn_one_hots = CNN(one_hots, n_layers, n_features, n_kernel_sizes)
  
  return cnn_one_hots


# Positional embeddings
# Code source: https://github.com/google/flax/blob/aff10f032e892e28a1acf4dd4ee9dcc6cd39a606/examples/wmt/models.py.
def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.
  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.
  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def apply(self,
            inputs,
            inputs_positions=None,
            max_len=512,
            posemb_init=None,
            cache=None):
    """Applies AddPositionEmbs module.
    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.
    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.
      max_len: maximum possible length for the input.
      posemb_init: positional embedding initializer, if None, then use a
        fixed (non-learned) sinusoidal embedding table.
      cache: flax attention cache for fast decoding.
    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    length = inputs.shape[1]
    pos_emb_shape = (1, max_len, inputs.shape[-1])
    if posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=max_len)(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', pos_emb_shape, posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if cache:
      if self.is_initializing():
        cache.store(lambda: (4, (1, 1)))
      else:
        cache_entry = cache.retrieve(None)
        i = cache_entry.i
        cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


def one_hot_pos_emb_encoder(batch_inds, num_categories, max_len, posemb_init):
  """Applies one-hot encoding with positional embeddings."""
  
  one_hots = jax.nn.one_hot(batch_inds, num_classes=num_categories)
  one_hots_pos_emb = AddPositionEmbs(one_hots, max_len=max_len, posemb_init=posemb_init)
  
  return one_hots_pos_emb


def cnn_one_hot_pos_emb_encoder(batch_inds, num_categories, n_layers, n_features, n_kernel_sizes, max_len, posemb_init):
  """Applies one-hot encoding with positional embeddings followed by CNN."""

  one_hots_pos_emb = one_hot_pos_emb_encoder(batch_inds, num_categories, max_len=max_len, posemb_init=posemb_init)
  cnn_one_hots_pos_emb = CNN(one_hots_pos_emb, n_layers, n_features, n_kernel_sizes)
  
  return cnn_one_hots_pos_emb
