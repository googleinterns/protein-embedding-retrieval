"""Encoder functions

Fixed and learnable transformations for embedding sequences.
"""

import flax
from flax import nn

import jax
from jax import lax
import jax.nn
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

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


def cnn_one_hot_encoder(batch_inds, num_categories, n_layers, n_features,
                        n_kernel_sizes):
    """Applies one-hot encoding followed by 1D CNN."""

    one_hots = one_hot_encoder(batch_inds, num_categories)
    cnn_one_hots = CNN(one_hots, n_layers, n_features, n_kernel_sizes)

    return cnn_one_hots


def encoder_fn_name_to_fn(encoder_fn_name):
    """Returns encoder_fn corresponding to encoder_fn_name."""

    if encoder_fn_name is None or encoder_fn_name == 'transformer':
        encoder_fn = None
    elif encoder_fn_name == 'one_hot':
        encoder_fn = one_hot_encoder
    elif encoder_fn_name == 'cnn_one_hot':
        encoder_fn = cnn_one_hot_encoder
    elif encoder_fn_name == 'one_hot_pos_emb':
        encoder_fn = one_hot_pos_emb_encoder
    elif encoder_fn_name == 'cnn_one_hot_pos_emb':
        encoder_fn = cnn_one_hot_pos_emb_encoder
    else:
        raise ValueError('Incorrect encoder name specified.')

    return encoder_fn
