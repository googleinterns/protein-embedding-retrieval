"""Contextual lenses on Pfam family classification dataset (first 100 families)."""


# Parse command line arguments.
from parser import parse_args
tpu_name, save_dir, restore_dir, use_pmap = parse_args()

# Connect TPU to VM instance.
from cloud_utils.tpu_init import connect_tpu
connect_tpu(tpu_name=tpu_name)

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import train, create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder

from loss_fns import cross_entropy_loss

from pfam_utils import create_pfam_batches, pfam_evaluate


# Global variables.
family_ids = open('pfam_family_ids.txt', 'r').readlines()
num_families = len(family_ids)

train_family_accessions = []
test_family_accessions = []
for i in range(1, 101):
  family_name = 'PF%05d' % i
  train_family_accessions.append(family_name)
  test_family_accessions.append(family_name)


# CNN + max pool.
epochs = 100
train_batches, train_indexes = create_pfam_batches(family_accessions=train_family_accessions, batch_size=512, 
                                                   epochs=epochs, drop_remainder=True)
lr = 1e-3
wd = 0.
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
                                                 output_features=num_families)

cnn_max_pool_optimizer = train(model=cnn_max_pool_model,
                               train_data=train_batches,
                               loss_fn=cross_entropy_loss,
                               loss_fn_kwargs=loss_fn_kwargs,
                               learning_rate=lr,
                               weight_decay=wd,
                               restore_dir=restore_dir,
                               save_dir=save_dir,
                               use_pmap=use_pmap)

cnn_max_pool_results, cnn_max_pool_preds = pfam_evaluate(predict_fn=cnn_max_pool_optimizer.target,
                                                         test_family_accessions=test_family_accessions,
                                                         title='CNN + Max Pool',
                                                         loss_fn_kwargs=loss_fn_kwargs,
                                                         batch_size=512)

print(cnn_max_pool_results)
