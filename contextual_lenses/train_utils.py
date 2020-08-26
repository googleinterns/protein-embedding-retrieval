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
from jax.config import config
config.enable_omnistaging()

import tensorflow as tf

import numpy as np

import functools

from google_research.protein_lm import domains, models


# Data batching.
def create_data_iterator(df,
                         input_col,
                         output_col,
                         batch_size,
                         epochs=1,
                         buffer_size=None,
                         seed=0,
                         drop_remainder=False,
                         add_outputs=True,
                         as_numpy=True):
    """Creates iterator of batches of (inputs) or (inputs, outputs)."""

    if buffer_size is None:
        buffer_size = len(df)

    inputs = list(df[input_col].values)
    inputs = tf.data.Dataset.from_tensor_slices(inputs)

    outputs = df[output_col].values
    outputs = tf.data.Dataset.from_tensor_slices(outputs)

    if add_outputs:
        batches = tf.data.Dataset.zip(
            (inputs, outputs)).shuffle(buffer_size=buffer_size,
                                       seed=seed,
                                       reshuffle_each_iteration=True)
    else:
        batches = inputs.shuffle(buffer_size=buffer_size,
                                 seed=seed,
                                 reshuffle_each_iteration=True)

    batches = batches.repeat(epochs).batch(batch_size=batch_size,
                                           drop_remainder=drop_remainder)

    if as_numpy:
        batches = batches.as_numpy_iterator()

    return batches


def path_inclusion_filter_fn(path, param, layer):
    """Returns whether or not layer name is contained in path."""

    return layer in path


def create_optimizer(model, learning_rate, weight_decay, layers=None):
    """Instantiates Adam multi-optimizer."""

    if layers is None:
        assert (
            type(learning_rate) == type(weight_decay) == float
        ), 'Specify float values for moded learning rate and weight decay!'
        optimizer_def = optim.Adam(learning_rate=learning_rate,
                                   weight_decay=weight_decay)
        optimizer = optimizer_def.create(model)

    else:
        assert (
            len(learning_rate) == len(weight_decay) == len(layers)
        ), 'Number of specified learning rates, weight decays, and layers must be equal!'
        optimizers = []
        for lr, wd, layer in zip(learning_rate, weight_decay, layers):
            if lr > 0:
                opt = optim.Adam(learning_rate=lr, weight_decay=wd)
                filter_fn = functools.partial(path_inclusion_filter_fn,
                                              layer=layer)
                traversal = optim.ModelParamTraversal(filter_fn)
                traversal_opt = (traversal, opt)
                optimizers.append(traversal_opt)
        optimizer_def = optim.MultiOptimizer(*optimizers)
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

    p_train_step = jax.pmap(train_step,
                            axis_name='batch',
                            static_broadcasted_argnums=(3, 4))

    return p_train_step


def train(model,
          train_data,
          loss_fn,
          loss_fn_kwargs,
          learning_rate=1e-4,
          weight_decay=0.1,
          layers=None,
          restore_dir=None,
          save_dir=None,
          use_pmap=False):
    """Instantiates optimizer, applies train_step/p_train_step over training data."""

    optimizer = create_optimizer(model,
                                 learning_rate=learning_rate,
                                 weight_decay=weight_decay,
                                 layers=layers)

    if restore_dir is not None:
        optimizer = checkpoints.restore_checkpoint(ckpt_dir=restore_dir,
                                                   target=optimizer)

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
        state = optimizer.state
        if type(state) == list:
            step = [sub_state.step for sub_state in state]
        else:
            step = state.step
        checkpoints.save_checkpoint(ckpt_dir=save_dir,
                                    target=optimizer,
                                    step=step)

    return optimizer


def load_params(params,
                encoder_fn_params=None,
                reduce_fn_params=None,
                predict_fn_params=None):
    """Updates randomly initialized parameters using loaded parameters."""

    loaded_params = copy.deepcopy(params)
    fn_names = list(loaded_params.keys())

    num_learnable_layers = len([
        params_dict for params_dict in
        [encoder_fn_params, reduce_fn_params, predict_fn_params]
        if params_dict is not None
    ])
    if encoder_fn_params is not None:
        encoder_fn_ind = '_0'
        if reduce_fn_params is not None:
            reduce_fn_ind = '_1'
            predict_fn_ind = '_2'
        else:
            predict_fn_ind = '_1'
    else:
        if reduce_fn_params is not None:
            reduce_fn_ind = '_0'
            predict_fn_ind = '_1'
        else:
            predict_fn_ind = '_0'

    assert (len(loaded_params.keys()) >= num_learnable_layers
            ), 'Model encoder and lens architecture incorrectly specified!'

    encoder_fn_name = None
    if encoder_fn_params is not None:
        for fn_name in fn_names:
            if encoder_fn_ind in fn_name:
                if encoder_fn_name is not None:
                    raise ValueError(
                        'Multiple instances of encoder_fn detected. %s' %
                        fn_name)
                encoder_fn_name = fn_name
        loaded_params[encoder_fn_name] = encoder_fn_params

    reduce_fn_name = None
    if reduce_fn_params is not None:
        for fn_name in fn_names:
            if reduce_fn_ind in fn_name:
                if reduce_fn_name is not None:
                    raise ValueError(
                        'Multiple instances of reduce_fn detected. %s' %
                        fn_name)
                reduce_fn_name = fn_name
        loaded_params[reduce_fn_name] = reduce_fn_params

    predict_fn_name = None
    if predict_fn_params is not None:
        for fn_name in fn_names:
            if predict_fn_ind in fn_name:
                if predict_fn_name is not None:
                    raise ValueError(
                        'Multiple instances of predict_fn detected. %s' %
                        fn_name)
                predict_fn_name = fn_name
        loaded_params[predict_fn_name] = predict_fn_params

    return loaded_params


class RepresentationModel(nn.Module):
    def apply(self,
              x,
              encoder_fn,
              encoder_fn_kwargs,
              reduce_fn,
              reduce_fn_kwargs,
              num_categories,
              output_features,
              output='prediction',
              use_transformer=False,
              padding_mask=None):
        """Computes padding mask, encodes indices using embeddings, 
       applies lensing operation, predicts scalar value.
    """

        outputs = dict()

        if padding_mask is None:
            padding_mask = jnp.expand_dims(jnp.where(x < num_categories - 1, 1,
                                                     0),
                                           axis=2)

        if not use_transformer:
            x = encoder_fn(x,
                           num_categories=num_categories,
                           **encoder_fn_kwargs)
        else:
            x = encoder_fn(x)

        rep = reduce_fn(x, padding_mask=padding_mask, **reduce_fn_kwargs)

        outputs['embedding'] = rep

        out = nn.Dense(rep,
                       output_features,
                       kernel_init=nn.initializers.xavier_uniform(),
                       bias_init=nn.initializers.normal(stddev=1e-6))

        outputs['prediction'] = out

        return outputs[output]


def create_representation_model(encoder_fn,
                                encoder_fn_kwargs,
                                reduce_fn,
                                reduce_fn_kwargs,
                                num_categories,
                                output_features,
                                output='prediction',
                                key=random.PRNGKey(0),
                                encoder_fn_params=None,
                                reduce_fn_params=None,
                                predict_fn_params=None):
    """Instantiates a RepresentationModel object."""

    module = RepresentationModel.partial(encoder_fn=encoder_fn,
                                         encoder_fn_kwargs=encoder_fn_kwargs,
                                         reduce_fn=reduce_fn,
                                         reduce_fn_kwargs=reduce_fn_kwargs,
                                         num_categories=num_categories,
                                         output_features=output_features,
                                         output=output,
                                         use_transformer=False)

    _, initial_params = RepresentationModel.init_by_shape(
        key,
        input_specs=[((1, 1), jnp.float32)],
        encoder_fn=encoder_fn,
        encoder_fn_kwargs=encoder_fn_kwargs,
        reduce_fn=reduce_fn,
        reduce_fn_kwargs=reduce_fn_kwargs,
        num_categories=num_categories,
        output_features=output_features,
        output=output,
        use_transformer=False)

    loaded_params = load_params(initial_params, encoder_fn_params,
                                reduce_fn_params, predict_fn_params)

    model = nn.Model(module, loaded_params)

    return model


def create_transformer_representation_model(transformer_kwargs,
                                            reduce_fn,
                                            reduce_fn_kwargs,
                                            num_categories,
                                            output_features,
                                            bidirectional=False,
                                            output='prediction',
                                            key=random.PRNGKey(0),
                                            encoder_fn_params=None,
                                            reduce_fn_params=None,
                                            predict_fn_params=None):
    """Instantiates a RepresentationModel object with Transformer encoder."""

    if not bidirectional:
        transformer = models.FlaxLM(**transformer_kwargs)
    else:
        transformer = models.FlaxBERT(**transformer_kwargs)
    transformer_optimizer = transformer._optimizer
    transformer_model = models.jax_utils.unreplicate(
        transformer_optimizer.target)
    transformer_encoder = transformer_model.module.partial(
        output_head='output_emb')

    module = RepresentationModel.partial(encoder_fn=transformer_encoder,
                                         encoder_fn_kwargs={},
                                         reduce_fn=reduce_fn,
                                         reduce_fn_kwargs=reduce_fn_kwargs,
                                         num_categories=num_categories,
                                         output_features=output_features,
                                         output=output,
                                         use_transformer=True)

    _, initial_params = RepresentationModel.init_by_shape(
        key,
        input_specs=[((1, 1), jnp.float32)],
        encoder_fn=transformer_encoder,
        encoder_fn_kwargs={},
        reduce_fn=reduce_fn,
        reduce_fn_kwargs=reduce_fn_kwargs,
        num_categories=num_categories,
        output_features=output_features,
        output=output,
        use_transformer=True)

    loaded_params = load_params(initial_params, encoder_fn_params,
                                reduce_fn_params, predict_fn_params)

    model = nn.Model(module, loaded_params)

    return model


def architecture_to_layers(encoder_fn_name, reduce_fn_name):

    layers = []

    no_trainable_encoder = False
    if encoder_fn_name is None or encoder_fn_name == 'transformer':
        layers.append('Transformer_0')
    elif encoder_fn_name == 'one_hot':
        no_trainable_encoder = True
    elif encoder_fn_name == 'cnn_one_hot':
        layers.append('CNN_0')
    else:
        raise ValueError('Incorrect encoder name specified.')

    no_trainable_lens = False
    if reduce_fn_name == 'mean_pool' or reduce_fn_name == 'max_pool':
        no_trainable_lens = True
    elif reduce_fn_name == 'linear_mean_pool' or reduce_fn_name == 'linear_max_pool':
        if no_trainable_encoder:
            layers.append('Dense_0')
        else:
            layers.append('Dense_1')
    elif reduce_fn_name == 'gated_conv':
        if no_trainable_encoder:
            layers.append('GatedConv_0')
        else:
            layers.append('GatedConv_1')
    else:
        raise ValueError('Incorrect lens name specified.')

    if no_trainable_encoder:
        if no_trainable_lens:
            layers.append('Dense_0')
        else:
            layers.append('Dense_1')
    else:
        if no_trainable_lens:
            layers.append('Dense_1')
        else:
            layers.append('Dense_2')

    trainable_encoder = not no_trainable_encoder

    return layers, trainable_encoder
