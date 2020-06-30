"""Nearest neighbors classification

Uses nearest neighborss to classify protein families 
using contextual lens embeddings trained to classify protein families.
"""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

import flax
from flax import nn
from flax.training import checkpoints

import jax
import jax.numpy as jnp

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import create_optimizer, train, \
create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import cross_entropy_loss

from pfam_utils import pfam_nearest_neighbors_classification


# Load family IDs
family_ids = open('pfam_family_ids.txt', 'r').readlines()
num_families = len(family_ids)

# Optimizer target initialization.
encoder_fn = cnn_one_hot_encoder
encoder_fn_kwargs = {
    'n_layers': 1,
    'n_features': [1024],
    'n_kernel_sizes': [12]
}
reduce_fn = max_pool
reduce_fn_kwargs = {
    
}
loss_fn_kwargs = {
  'num_classes': num_families
}
cnn_max_pool_model = create_representation_model(encoder_fn=encoder_fn,
                                                 encoder_fn_kwargs=encoder_fn_kwargs,
                                                 reduce_fn=reduce_fn,
                                                 reduce_fn_kwargs=reduce_fn_kwargs,
                                                 num_categories=26,
                                                 output_features=num_families,
                                                 embed=True)

optimizer = create_optimizer(cnn_max_pool_model, learning_rate=0., weight_decay=0.)


# Restore optimizer from checkpoint. 
if restore_dir is not None:
  optimizer = checkpoints.restore_checkpoint(ckpt_dir=restore_dir, target=optimizer)


# Lens training: PF00001 - PF00100, NN training: PF00001 - PF00100
train_family_accessions = []
test_family_accessions = []
for i in range(1, 101):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)

results = pfam_nearest_neighbors_classification(encoder=optimizer.target, 
											    train_family_accessions=train_family_accessions, 
											    test_family_accessions=test_family_accessions)[0]

print("Lens training: PF00001 - PF00100, NN training: PF00001 - PF00100")
print(results)
print()


# Lens training: PF00001 - PF00100, NN training: PF00101 - PF00200
train_family_accessions = []
test_family_accessions = []
for i in range(101, 201):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)

results = pfam_nearest_neighbors_classification(encoder=optimizer.target, 
											    train_family_accessions=train_family_accessions, 
											    test_family_accessions=test_family_accessions)[0]

print("Lens training: PF00001 - PF00100, NN training: PF00101 - PF00200")
print(results)
print()


# Lens training: PF00001 - PF00100, NN training: PF00001 - PF00200
train_family_accessions = []
test_family_accessions = []
for i in range(1, 201):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)

results = pfam_nearest_neighbors_classification(encoder=optimizer.target, 
											    train_family_accessions=train_family_accessions, 
											    test_family_accessions=test_family_accessions)[0]

print("Lens training: PF00001 - PF00100, NN training: PF00001 - PF00200")
print(results)
print()


# Lens training: PF00001 - PF00100, NN training: PF00001 - PF00400
train_family_accessions = []
test_family_accessions = []
for i in range(1, 401):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)

results = pfam_nearest_neighbors_classification(encoder=optimizer.target, 
											    train_family_accessions=train_family_accessions, 
											    test_family_accessions=test_family_accessions)[0]

print("Lens training: PF00001 - PF00100, NN training: PF00001 - PF00400")
print(results)
print()
