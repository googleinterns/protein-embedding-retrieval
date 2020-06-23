"""Contextual lenses

Creates sequence length independent representation of embedded sequences
Original paper: https://arxiv.org/pdf/2002.08866.pdf."""


import flax
from flax import nn

import jax
import jax.nn
import jax.numpy as jnp

import numpy as np


def max_pool(x, padding_mask=None, **reduce_fn_kwargs):
  """Apply padding, take maximum over sequence length axis."""

  if padding_mask is not None:
    x = x * padding_mask
    neg_mask = - 999999. * (1 - padding_mask)
    x = x + neg_mask
   
  rep = jnp.max(x, axis=-2)
   
  return rep


def mean_pool(x, padding_mask=None, **reduce_fn_kwargs):
  """Apply padding, take mean over sequence length axis."""

  if padding_mask is not None:
    x = x * padding_mask
    rep = jnp.sum(x, axis=-2) / jnp.sum(padding_mask, axis=-2)
  else:
    rep = jnp.mean(x, axis=-2)

  return rep


def linear_max_pool(x, padding_mask=None, **reduce_fn_kwargs):
  """Apply linear transformation + ReLU, apply padding,
     take maximum over sequence length."""
  
  rep_size = list(reduce_fn_kwargs.values())[0]

  x = nn.Dense(
        x,
        rep_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
  
  x = nn.relu(x)
  
  rep = max_pool(x, padding_mask)
  
  return rep


def linear_mean_pool(x, padding_mask=None, **reduce_fn_kwargs):
  """Apply linear transformation + ReLU, apply padding,
     take mean over sequence length."""
  
  rep_size = list(reduce_fn_kwargs.values())[0]
  
  x = nn.Dense(
        x,
        rep_size,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
  
  x = nn.relu(x)
  
  rep = mean_pool(x, padding_mask)
  
  return rep


class GatedConv(nn.Module):
  """Gated Convolutional lens followed by max pooling,
     see original paper for details."""

  def apply(self, x, padding_mask=None, **reduce_fn_kwargs):
    
    rep_size, M_layers, M_features, M_kernel_sizes, zero_rep_size = list(reduce_fn_kwargs.values())
    
    H_0 = nn.relu(nn.Dense(x, zero_rep_size))
    G_0 = nn.relu(nn.Dense(x, zero_rep_size))
    H, G = jnp.expand_dims(H_0, axis=2), jnp.expand_dims(G_0, axis=2)

    for layer in range(1, M_layers+1):
      
      if layer < M_layers:
        H_features, G_features = M_features[layer-1]
      else:
        H_features, G_features = zero_rep_size, zero_rep_size
      
      H_kernel_size, G_kernel_size = M_kernel_sizes[layer-1]

      H = nn.Conv(H, features=H_features, kernel_size=(H_kernel_size, 1))
      G = nn.Conv(G, features=G_features, kernel_size=(G_kernel_size, 1)) 

      if layer < M_layers:
        H = nn.relu(H)
        G = nn.relu(G)
      else:
        H = nn.tanh(H)
        G = nn.sigmoid(G)

    H, G = jnp.squeeze(H, axis=2), jnp.squeeze(G, axis=2)
    
    F = H * G + G_0
    
    rep = linear_max_pool(F, padding_mask=padding_mask, rep_size=rep_size)
    
    return rep


def gated_conv(x, padding_mask=None, **reduce_fn_kwargs):
  """Calls GatedConv method for use as a lens."""

  rep_size, M_layers, M_features, M_kernel_sizes, zero_rep_size = list(reduce_fn_kwargs.values())
  rep = GatedConv(x, rep_size=rep_size, M_layers=M_layers, M_features=M_features, 
                  M_kernel_sizes=M_kernel_sizes, padding_mask=padding_mask,
                  zero_rep_size=zero_rep_size)
  
  return rep
