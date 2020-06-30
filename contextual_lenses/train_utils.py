"""Train utils

General tools for instantiating and training models.
"""


import flax
from flax import nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils

import jax
from jax import random
import jax.nn
import jax.numpy as jnp

import tensorflow as tf

import numpy as np

import functools



# Data batching.
def create_data_iterator(df, input_col, output_col, batch_size, epochs=1, buffer_size=None, seed=0, drop_remainder=False):

  if buffer_size is None:
    buffer_size = len(df)

  inputs = list(df[input_col].values)
  inputs = tf.data.Dataset.from_tensor_slices(inputs)

  outputs = df[output_col].values
  outputs = tf.data.Dataset.from_tensor_slices(outputs)

  batches = tf.data.Dataset.zip((inputs, outputs)).shuffle(buffer_size=buffer_size, seed=seed)
  batches = batches.repeat(epochs).batch(batch_size=batch_size, drop_remainder=drop_remainder).as_numpy_iterator()

  return batches


def create_optimizer(model, learning_rate, weight_decay):
  """Instantiates Adam optimizer."""

  optimizer_def = optim.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
  optimizer = optimizer_def.create(model)
  
  return optimizer


@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(optimizer, X, Y, loss_fn, loss_fn_kwargs):
  """Trains model (optimizer.target) using specified loss function."""

  def compute_loss_fn(model, X, Y, loss_fn, loss_fn_kwargs):
    Y_hat = model(X)
    loss = loss_fn(Y, Y_hat, **loss_fn_kwargs)
    return loss
  
  grad_fn = jax.value_and_grad(compute_loss_fn)
  _, grad = grad_fn(optimizer.target, X, Y, loss_fn, loss_fn_kwargs)
  optimizer = optimizer.apply_gradient(grad)
  
  return optimizer


def get_p_train_step():
    """Wraps train_step with jax.pmap."""
    
    p_train_step = jax.pmap(train_step, axis_name='batch', static_broadcasted_argnums=(3, 4))
    
    return p_train_step


def train(model, train_data, loss_fn, loss_fn_kwargs, learning_rate=1e-4, weight_decay=0.1,
          restore_dir=None, save_dir=None, use_pmap=False):
  """Instantiates optimizer, applies train_step/p_train_step over training data.""" 
  
  optimizer = create_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)

  if restore_dir is not None:
    optimizer = checkpoints.restore_checkpoint(ckpt_dir=restore_dir, target=optimizer)

  if use_pmap:
    p_train_step = get_p_train_step()
    optimizer = optimizer.replicate()

    for batch in iter(train_data):
      X, Y = batch
      X, Y = common_utils.shard(X), common_utils.shard(Y)
      optimizer = p_train_step(optimizer, X, Y, loss_fn, loss_fn_kwargs)

    optimizer = optimizer.unreplicate()
  
  else: 
    for batch in iter(train_data):
      X, Y = batch
      optimizer = train_step(optimizer, X, Y, loss_fn, loss_fn_kwargs)
  
  if save_dir is not None:
    checkpoints.save_checkpoint(ckpt_dir=save_dir, target=optimizer, step=optimizer.state.step)

  return optimizer


class RepresentationModel(nn.Module):

  def apply(self, x, encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs,
            num_categories, output_features, embed=False):
    """Computes padding mask, encodes indices using embeddings, 
       applies lensing operation, predicts scalar value.
    """

    padding_mask = jnp.expand_dims(jnp.where(x < num_categories-1, 1, 0), axis=2)

    x = encoder_fn(x, num_categories=num_categories, **encoder_fn_kwargs)

    rep = reduce_fn(x, padding_mask=padding_mask, **reduce_fn_kwargs)

    if embed:
      return rep
    
    out = nn.Dense(rep,
                   output_features,
                   kernel_init=nn.initializers.xavier_uniform(),
                   bias_init=nn.initializers.normal(stddev=1e-6)) 
    
    return out


def create_representation_model(encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs,
                                num_categories, output_features, embed=False, key=random.PRNGKey(0)):
  """Instantiates a RepresentationModel object."""

  module = RepresentationModel.partial(encoder_fn=encoder_fn,
                                       encoder_fn_kwargs=encoder_fn_kwargs, 
                                       reduce_fn=reduce_fn,
                                       reduce_fn_kwargs=reduce_fn_kwargs,
                                       num_categories=num_categories,
                                       output_features=output_features,
                                       embed=embed)
  
  _, initial_params = RepresentationModel.init_by_shape(key,
                                                        input_specs=[((1, 1), jnp.float32)],
                                                        encoder_fn=encoder_fn,
                                                        encoder_fn_kwargs=encoder_fn_kwargs,
                                                        reduce_fn=reduce_fn,
                                                        reduce_fn_kwargs=reduce_fn_kwargs,
                                                        num_categories=num_categories,
                                                        output_features=output_features,
                                                        embed=embed)
  
  model = nn.Model(module, initial_params)
  
  return model
