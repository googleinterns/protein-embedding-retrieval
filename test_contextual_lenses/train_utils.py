"""Train utils."""
"""General tools for instantiating and training models."""


import flax
from flax import nn
from flax import optim

import jax
from jax import random
import jax.nn
import jax.numpy as jnp

import numpy as np


def create_optimizer(model, learning_rate, weight_decay):
  """Instantiates Adam optimizer."""

  optimizer_def = optim.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
  optimizer = optimizer_def.create(model)
  
  return optimizer


@jax.jit
def train_step(optimizer, X, Y):
  """Trains model (optimizer.target) using MSE loss."""

  def loss_fn(model):
    Y_hat = jnp.squeeze(model(X), axis=1)
    loss = jnp.mean((Y-Y_hat)**2)
    return loss
  
  grad_fn = jax.value_and_grad(loss_fn)
  _, grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  
  return optimizer


def train(model, train_data, learning_rate=1e-4, weight_decay=0.1):
  """Instantiates optimizer, applies train_step over training data.""" 
  
  optimizer = create_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
  
  del model
  
  for batch in iter(train_data):
    X, Y = batch
    optimizer = train_step(optimizer, X, Y)
  
  return optimizer


class RepresentationModel(nn.Module):

  def apply(self, x, encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs,
            num_categories=21):
    """Computes padding mask, encodes indices using embeddings, 
       applies lensing operation, predicts scalar value."""

    padding_mask = jnp.expand_dims(jnp.where(x < num_categories-1, 1, 0), axis=2)
    
    x = encoder_fn(x, num_categories=num_categories, **encoder_fn_kwargs)

    rep = reduce_fn(x, padding_mask=padding_mask, **reduce_fn_kwargs)

    out = nn.Dense(rep,
                   1,
                   kernel_init=nn.initializers.xavier_uniform(),
                   bias_init=nn.initializers.normal(stddev=1e-6)) 
    
    return out


def create_representation_model(encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs,
                                num_categories=21, key=random.PRNGKey(0)):
  """Instantiates a RepresentationModel object."""

  module = RepresentationModel.partial(encoder_fn=encoder_fn,
                                       encoder_fn_kwargs=encoder_fn_kwargs, 
                                       reduce_fn=reduce_fn,
                                       reduce_fn_kwargs=reduce_fn_kwargs,
                                       num_categories=num_categories)
  
  _, initial_params = RepresentationModel.init_by_shape(key,
                                                     input_specs=[((1, 1), jnp.float32)],
                                                     encoder_fn=encoder_fn,
                                                     encoder_fn_kwargs=encoder_fn_kwargs,
                                                     reduce_fn=reduce_fn,
                                                     reduce_fn_kwargs=reduce_fn_kwargs,
                                                     num_categories=num_categories)
  
  model = nn.Model(module, initial_params)
  
  return model
