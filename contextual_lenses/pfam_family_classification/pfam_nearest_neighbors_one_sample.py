"""Nearest neighbors classification

Uses nearest neighbors with a single sample to classify protein families 
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

from pfam_utils import train_and_test_knn_pfam_families


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


family_ranges = [(1, 100), (101, 200). (1, 200), (1, 400)]
for family_range in family_ranges:
  start, end = family_range
  results = train_and_test_knn_pfam_families(encoder=optimizer.target, start=start, end=end, train_samples=1)
  print('Lens training: None, k-NN training: ' + 'PF%05d' % start + ' - ' + 'PF%05d' % end)
  print(results)
  print()


# Restore optimizer from checkpoint. 
if restore_dir is not None:
  optimizer = checkpoints.restore_checkpoint(ckpt_dir=restore_dir, target=optimizer)

  family_ranges = [(1, 100), (101, 200). (1, 200), (1, 400)]
  for family_range in family_ranges:
    start, end = family_range
    results = train_and_test_knn_pfam_families(encoder=optimizer.target, start=start, end=end, train_samples=1)
    print('Lens training: ' + 'PF%05d' % 1 + ' - ' + 'PF%05d' % 100 + ', k-NN training: ' + 'PF%05d' % start + ' - ' + 'PF%05d' % end)
    print(results)
    print()
else:
  print('Please specify restore_dir for loading optimizer.')

  