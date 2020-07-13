"""Transformer contextual lenses on TAPE fluoresence dataset."""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap, restore_transformer_dir, bidirectional = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

import flax.nn as nn

from protein_lm import domains, models

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import train, create_transformer_representation_model

from loss_fns import mse_loss

from tape_gfp_utils import create_gfp_batches, gfp_evaluate

from load_transformer import load_transformer_params


# Transformer + MaxPool lens.
epochs = 100
train_batches, train_fluorescences = create_gfp_batches(batch_size=256, epochs=epochs, drop_remainder=True)
lr = 1e-3
wd = 0.

transformer_kwargs = {
    'emb_dim': 256,
    'num_heads': 4,
    'num_layers': 2,
    'qkv_dim': 256,
    'mlp_dim': 1024,
  }

if restore_transformer_dir is not None:
  pretrained_transformer_params = load_transformer_params(restore_transformer_dir, models.FlaxLM)
else:
  pretrained_transformer_params = None

reduce_fn = linear_max_pool
reduce_fn_kwargs = {
    'rep_size': 256
}
loss_fn_kwargs = {
    
}
transformer_max_pool_model = create_transformer_representation_model(transformer_kwargs=transformer_kwargs,
                                                                     reduce_fn=reduce_fn,
                                                                     reduce_fn_kwargs=reduce_fn_kwargs,
                                                                     num_categories=27,
                                                                     output_features=1,
                                                                     bidirectional=bidirectional,
                                                                     encoder_fn_params=pretrained_transformer_params)

transformer_max_pool_optimizer = train(model=transformer_max_pool_model,
                                       train_data=train_batches,
                                       loss_fn=mse_loss,
                                       loss_fn_kwargs=loss_fn_kwargs,
                                       learning_rate=lr,
                                       weight_decay=wd,
                                       restore_dir=restore_dir,
                                       save_dir=save_dir,
                                       use_pmap=use_pmap)

transformer_max_pool_results, transformer_max_pool_pred_fluorescences = \
  gfp_evaluate(predict_fn=transformer_max_pool_optimizer.target,
               title='Transformer + LinearMaxPool',
               batch_size=256)

print(transformer_max_pool_results)

