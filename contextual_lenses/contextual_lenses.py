"""Contextual lenses
    
Creates sequence length independent representation of embedded sequences
Original paper: https://arxiv.org/pdf/2002.08866.pdf.
"""


import flax
from flax import nn

import jax
import jax.nn
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

import numpy as np

from operator import itemgetter


def max_pool(x, padding_mask=None, pad_constant=1e8):
  """Apply padding, take maximum over sequence length axis."""

  if padding_mask is not None:
    x = x * padding_mask
    neg_mask = - pad_constant * (1 - padding_mask)
    x = x + neg_mask
   
  rep = jnp.max(x, axis=-2)
   
  return rep


def mean_pool(x, padding_mask=None):
  """Apply padding, take mean over sequence length axis."""

  if padding_mask is not None:
    x = x * padding_mask
    rep = jnp.sum(x, axis=-2) / jnp.sum(padding_mask, axis=-2)
  else:
    rep = jnp.mean(x, axis=-2)

  return rep


def linear_max_pool(x, rep_size, padding_mask=None):
  """Apply linear transformation + ReLU, apply padding,
     take maximum over sequence length.
  """
  
  x = nn.Dense(
        x,
        rep_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
  
  x = nn.relu(x)
  
  rep = max_pool(x, padding_mask=padding_mask)
  
  return rep


def linear_mean_pool(x, rep_size, padding_mask=None):
  """Apply linear transformation + ReLU, apply padding,
     take mean over sequence length.
  """
    
  x = nn.Dense(
        x,
        rep_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
  
  x = nn.relu(x)
  
  rep = mean_pool(x, padding_mask=padding_mask)
  
  return rep


class GatedConv(nn.Module):
  """Gated Convolutional lens followed by max pooling,
     see original paper for details.
  """

  def apply(self, x, rep_size, m_layers, m_features, m_kernel_sizes, conv_rep_size, padding_mask=None):
        
    H_0 = nn.relu(nn.Dense(x, conv_rep_size))
    G_0 = nn.relu(nn.Dense(x, conv_rep_size))
    H, G = jnp.expand_dims(H_0, axis=2), jnp.expand_dims(G_0, axis=2)

    for layer in range(1, m_layers+1):
      
      if layer < m_layers:
        H_features, G_features = m_features[layer-1]
      else:
        H_features, G_features = conv_rep_size, conv_rep_size
      
      H_kernel_size, G_kernel_size = m_kernel_sizes[layer-1]

      H = nn.Conv(H, features=H_features, kernel_size=(H_kernel_size, 1))
      G = nn.Conv(G, features=G_features, kernel_size=(G_kernel_size, 1)) 

      if layer < m_layers:
        H = nn.relu(H)
        G = nn.relu(G)
      else:
        H = nn.tanh(H)
        G = nn.sigmoid(G)

    H, G = jnp.squeeze(H, axis=2), jnp.squeeze(G, axis=2)
    
    F = H * G + G_0
    
    rep = linear_max_pool(F, padding_mask=padding_mask, rep_size=rep_size)
    
    return rep


def gated_conv(x, rep_size, m_layers, m_features, m_kernel_sizes, conv_rep_size, padding_mask=None):
  """Calls GatedConv method for use as a lens."""

  rep = GatedConv(x, rep_size=rep_size, m_features=m_features, m_layers=m_layers, m_kernel_sizes=m_kernel_sizes, 
                  conv_rep_size=conv_rep_size, padding_mask=padding_mask)
  
  return rep


def reduce_fn_name_to_fn(reduce_fn_name):
  """Returns reduce_fn corresponding to reduce_fn_name."""

  if reduce_fn_name == 'mean_pool':
    reduce_fn = mean_pool
  elif reduce_fn_name == 'max_pool':
    reduce_fn = max_pool
  elif reduce_fn_name == 'linear_mean_pool':
    reduce_fn = linear_mean_pool
  elif reduce_fn_name == 'linear_max_pool':
    reduce_fn = linear_max_pool
  elif reduce_fn_name == 'gated_conv':
    reduce_fn = gated_conv
  else:
    raise ValueError('Incorrect lens name specified.')

  return reduce_fn

