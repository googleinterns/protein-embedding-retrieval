"""Nearest neighbors classification sampling sweep

Measures performance of nearest neighbors protein family classification 
with different numbers of training samples
using transformer contextual lens embeddings trained to classify protein families.
"""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap, restore_transformer_dir, bidirectional, static_encoder = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

import flax
from flax import nn
from flax.training import checkpoints

import jax
import jax.numpy as jnp

import numpy as np

import pandas as pd

from protein_lm import domains, models

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import create_optimizer, train, \
create_transformer_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import cross_entropy_loss

from pfam_utils import pfam_nearest_neighbors_classification

from load_transformer import load_transformer_params, load_transformer_encoder


# Load family IDs
family_ids = open('pfam_family_ids.txt', 'r').readlines()
num_families = len(family_ids)


# Optimizer target initialization.
transformer_kwargs = {
    'emb_dim': 256,
    'num_heads': 4,
    'num_layers': 2,
    'qkv_dim': 256,
    'mlp_dim': 1024,
  }

if bidirectional:
  model_type = models.FlaxBERT
else:
  model_type = models.FlaxLM

if restore_transformer_dir is not None:
  pretrained_transformer_params = load_transformer_params(restore_transformer_dir, model_type)
  transformer_encoder = load_transformer_encoder(restore_transformer_dir, model_type)
else:
  pretrained_transformer_params = None
  transformer = model_type(**transformer_kwargs)
  transformer_encoder = models.jax_utils.unreplicate(transformer._optimizer.target)

reduce_fn = linear_max_pool
reduce_fn_kwargs = {
    'rep_size': 256
}
loss_fn_kwargs = {
  'num_classes': num_families
}
model = create_transformer_representation_model(transformer_kwargs=transformer_kwargs,
                                                reduce_fn=reduce_fn,
                                                reduce_fn_kwargs=reduce_fn_kwargs,
                                                num_categories=27,
                                                output_features=num_families,
                                                output='embedding',
                                                bidirectional=bidirectional,
                                                encoder_fn_params=pretrained_transformer_params)


learning_rate = [0.0, 1e-3, 1e-3]
weight_decay = [0.0, 0.0, 0.0]
layers = ['Transformer_0', 'Dense_1', 'Dense_2']

optimizer = create_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay, layers=layers)


sweep_data = []

named_family_accessions = {}
named_family_accessions[0] = ('1-100', ['PF%05d' % i for i in range(1, 101)])
named_family_accessions[1] = ('101-200', ['PF%05d' % i for i in range(101, 201)])
named_family_accessions[2] = ('1-200', ['PF%05d' % i for i in range(1, 201)])

train_samples_sweep = [i for i in range(1, 6)] + [5*i for i in range(2, 6)] + [25*i for i in range(2, 5)] + [None]
for train_samples in train_samples_sweep:
    lens_train = 0
    for key in named_family_accessions.keys():
        families, family_accessions = named_family_accessions[i]
        results = pfam_nearest_neighbors_classification(encoder=optimizer.target, 
											            train_family_accessions=family_accessions, 
											            test_family_accessions=family_accessions,
                                                        train_samples=train_samples)[0]
        accuracy = results['1-nn accuracy']
        datum = {
                 'families': families,
                 'train_samples': train_samples,
                 'lens_train': lens_train,
                 'accuracy': accuracy
                 }
        print(datum)
        sweep_data.append(datum)
        

# Restore optimizer from checkpoint. 
if restore_dir is not None:
  loaded_optimizer = checkpoints.restore_checkpoint(ckpt_dir=restore_dir, target=optimizer)
else:
   print('Specify restore_dir!')

train_samples_sweep = [i for i in range(1, 6)] + [5*i for i in range(2, 6)] + [25*i for i in range(2, 5)] + [None]
for train_samples in train_samples_sweep:
    lens_train = 1
    for key in named_family_accessions.keys():
        families, family_accessions = named_family_accessions[i]
        results = pfam_nearest_neighbors_classification(encoder=loaded_optimizer.target, 
											            train_family_accessions=family_accessions, 
											            test_family_accessions=family_accessions,
                                                        train_samples=train_samples)[0]
        accuracy = results['1-nn accuracy']
        datum = {
                 'families': families,
                 'train_samples': train_samples,
                 'lens_train': lens_train,
                 'accuracy': accuracy
                 }        
        print(datum)
        sweep_data.append(datum)


sweep_df = pd.DataFrame(sweep_data)
if restore_transformer_dir is not None:
  sweep_df.to_csv('transformer_frozen_pretrain_samples_sweep.csv')
else:
  sweep_df.to_csv('transformer_frozen_no_pretrain_samples_sweep.csv')

        
