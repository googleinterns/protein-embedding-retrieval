'''Utils for loading FlaxLM or FlaxBERT models or parameters.'''


import os
import tensorflow as tf
import gin
from protein_lm import data, models


def load_transformer_model(ckpt_dir, model_cls, domain=None):
  """Loads a model from directory."""
  
  if domain is None:
    domain = data.protein_domain
  config_path = os.path.join(ckpt_dir, 'config.gin')
  with gin.config_scope('load_model'):
    with tf.io.gfile.GFile(config_path) as f:
      gin.parse_config(f, skip_unknown=True)
    model = model_cls(domain=domain)

    model.load_checkpoint(ckpt_dir)
  return model


def load_transformer_params(ckpt_dir, model_cls, domain=None):
    """Returns parameters of a loaded model."""
    
    model = load_transformer_model(ckpt_dir, model_cls, domain=domain)
    params = models.jax_utils.unreplicate(model._optimizer.target).params
    
    return params
