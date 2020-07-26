"""Transformer contextual lenses on Pfam family classification dataset (first 100 families)."""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap, restore_transformer_dir, bidirectional = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

from protein_lm import domains, models

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import train, create_transformer_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import cross_entropy_loss

from pfam_utils import create_pfam_batches, pfam_evaluate, pfam_num_categories

from load_transformer import load_transformer_params


# Global variables.
family_ids = open('pfam_family_ids.txt', 'r').readlines()
num_families = len(family_ids)

train_family_accessions = []
test_family_accessions = []
for i in range(1, 101):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)


# Transformer + LinearMaxPool.
epochs = 100
train_batches, train_indexes = create_pfam_batches(family_accessions=train_family_accessions, batch_size=512, 
                                                   epochs=epochs, drop_remainder=True)

transformer_kwargs = {
    'emb_dim': 256,
    'num_heads': 4,
    'num_layers': 2,
    'qkv_dim': 256,
    'mlp_dim': 1024,
  }


learning_rate = [0.0, 1e-3, 1e-3]
weight_decay = [0.0, 0.0, 0.0]
layers = ['Transformer_0', 'Dense_1', 'Dense_2']

if restore_transformer_dir is not None:
  pretrained_transformer_params = load_transformer_params(restore_transformer_dir, models.FlaxLM)
else:
  pretrained_transformer_params = None

reduce_fn = linear_max_pool
reduce_fn_kwargs = {
    'rep_size': 256
}
loss_fn_kwargs = {
  'num_classes': num_families
}
title = 'Transformer + LinearMaxPool'

model = create_transformer_representation_model(transformer_kwargs=transformer_kwargs,
                                                reduce_fn=reduce_fn,
                                                reduce_fn_kwargs=reduce_fn_kwargs,
                                                num_categories=pfam_num_categories,
                                                output_features=num_families,
                                                output='prediction',
                                                bidirectional=bidirectional,
                                                encoder_fn_params=pretrained_transformer_params)

optimizer = train(model=model,
                  train_data=train_batches,
                  loss_fn=cross_entropy_loss,
                  loss_fn_kwargs=loss_fn_kwargs,
                  learning_rate=learning_rate,
                  weight_decay=weight_decay,
                  layers=layers,
                  restore_dir=restore_dir,
                  save_dir=save_dir,
                  use_pmap=use_pmap)

results, preds = pfam_evaluate(predict_fn=optimizer.target,
                               test_family_accessions=test_family_accessions,
                               title=title,
                               loss_fn_kwargs=loss_fn_kwargs,
                               batch_size=512)

print(results)
