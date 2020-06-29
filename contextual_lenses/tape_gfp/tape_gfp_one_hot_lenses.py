"""Contextual lenses on TAPE fluoresence dataset."""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

import flax.nn as nn

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import train, create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import mse_loss

from tape_gfp_utils import create_train_batches, evaluate


# One-hot + positional embeddings + CNN + GatedConv lens.
epochs = 50
train_batches = create_train_batches(batch_size=256, epochs=epochs, drop_remainder=True)
lr = 1e-3
wd = 0.
encoder_fn = cnn_one_hot_pos_emb_encoder
encoder_fn_kwargs = {
    'n_layers': 1,
    'n_features': [512],
    'n_kernel_sizes': [12],
    'max_len': 512,
    'posemb_init': nn.initializers.normal(stddev=1e-6)
}
reduce_fn = gated_conv
reduce_fn_kwargs = {
    'rep_size': 256,
    'm_layers': 3,
    'm_features': [[512, 512], [512, 512]],
    'm_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
    'conv_rep_size': 256
}
loss_fn_kwargs = {

}
cnn_pos_emb_gated_conv_model = create_representation_model(encoder_fn=encoder_fn,
                                                           encoder_fn_kwargs=encoder_fn_kwargs,
                                                           reduce_fn=reduce_fn,
                                                           reduce_fn_kwargs=reduce_fn_kwargs,
                                                           num_categories=21,
                                                           output_features=1)

cnn_pos_emb_gated_conv_optimizer = train(model=cnn_pos_emb_gated_conv_model,
                                         train_data=train_batches,
                                         loss_fn=mse_loss,
                                         loss_fn_kwargs=loss_fn_kwargs,
                                         learning_rate=lr,
                                         weight_decay=wd,
                                         restore_dir=restore_dir,
                                         save_dir=save_dir,
                                         use_pmap=use_pmap)

cnn_pos_emb_gated_conv_results, cnn_pos_emb_gated_conv_pred_fluorescences = \
  evaluate(predict_fn=cnn_pos_emb_gated_conv_optimizer.target,
           title='OneHot + PosEmb + CNN + GatedConv',
           batch_size=256)

print(cnn_pos_emb_gated_conv_results)
