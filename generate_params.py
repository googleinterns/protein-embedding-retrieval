"""Generate JSON files with hyperparameter combos for caliban."""

import json

import itertools

import numpy as np


params_combinations = []
index_to_params = {}

count = 0


# *** FIRST BATCH *** 
reduce_fn_name = 'linear_max_pool'
reduce_fn_kwargs_paths = ['linear_pool_256', 'linear_pool_1024']
epochs = 50
batch_size = 64
lr_1 = 0.0
lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
wd_1 = 0.0
wds = [0.0, 0.1]
train_families = 1000
lens_train_samples = 50


# Transformer experiments
encoder_fn_name = 'transformer'
encoder_fn_kwargs_path = 'medium_transformer_kwargs'
use_transformer = True 
use_bert = True 
for params in itertools.product(reduce_fn_kwargs_paths, lrs, lrs, wds, wds):
	reduce_fn_kwargs_path, lr_2, lr_3, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'encoder_fn_name': encoder_fn_name,
					'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'batch_size': batch_size,
					'lens_lr': lr_2, 
					'predictor_lr': lr_3,
					'lens_wd':  wd_2,
					'predictor_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'use_transformer': use_transformer,
					'use_bert': use_bert,
					'index': index
				 }
	# params_combinations.append(param_dict)
	# index_to_params[index] = param_dict
	count += 1


# Pretrained transformer experiments
encoder_fn_name = 'transformer'
encoder_fn_kwargs_path = 'medium_transformer_kwargs'
use_transformer = True 
use_bert = True 
restore_transformer_dir = 'gs://sequin-public/transformer_models/medium_trembl_bert/'
for params in itertools.product(reduce_fn_kwargs_paths, lrs, lrs, wds, wds):
	reduce_fn_kwargs_path, lr_2, lr_3, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'encoder_fn_name': encoder_fn_name,
					'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'batch_size': batch_size,
					'lens_lr': lr_2, 
					'predictor_lr': lr_3,
					'lens_wd':  wd_2,
					'predictor_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'use_transformer': use_transformer,
					'use_bert': use_bert,
					'restore_transformer_dir': restore_transformer_dir,
					'index': index
				 }
	# params_combinations.append(param_dict)
	# index_to_params[index] = param_dict
	count += 1


# CNN experiments
batch_size = 256
encoder_fn_name = 'cnn_one_hot'
encoder_fn_kwargs_path = 'cnn_kwargs'
use_transformer = False 
use_bert = False 
for params in itertools.product(reduce_fn_kwargs_paths, lrs, lrs, lrs, wds, wds, wds):
	reduce_fn_kwargs_path, lr_1, lr_2, lr_3, wd_1, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'encoder_fn_name': encoder_fn_name,
					'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'batch_size': batch_size,
					'encoder_lr': lr_1,
					'lens_lr': lr_2,
					'predictor_lr': lr_3,
					'encoder_wd': wd_1,
					'lens_wd': wd_2,
					'encoder_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'use_transformer': use_transformer,
					'use_bert': use_bert,
					'index': index
				 }
	# params_combinations.append(param_dict)
	# index_to_params[index] = param_dict
	count += 1


# *** SECOND BATCH ***
reduce_fn_name = 'linear_max_pool'
reduce_fn_kwargs_path = 'linear_pool_1024'
epochs = 10
measurements = 2
batch_size = 64
lr_1 = 0.0
lrs = [1e-3, 5e-4, 1e-4, 5e-5]
wd_1 = 0.0
wds = [0.0, 0.05, 0.1, 0.2]
train_families = 10000
lens_train_samples = 50


# Transformer experiments
encoder_fn_name = 'transformer'
encoder_fn_kwargs_path = 'medium_transformer_kwargs'
use_transformer = True 
use_bert = True 
for params in itertools.product(lrs, lrs, wds, wds):
	lr_2, lr_3, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'encoder_fn_name': encoder_fn_name,
					'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'measurements': measurements,
					'batch_size': batch_size,
					'lens_lr': lr_2, 
					'predictor_lr': lr_3,
					'lens_wd':  wd_2,
					'predictor_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'use_transformer': use_transformer,
					'use_bert': use_bert,
					'index': index
				 }
	# params_combinations.append(param_dict)
	# index_to_params[index] = param_dict
	count += 1


# Pretrained transformer experiments
encoder_fn_name = 'transformer'
encoder_fn_kwargs_path = 'medium_transformer_kwargs'
restore_transformer_dir = 'gs://sequin-public/transformer_models/medium_trembl_bert/'
use_transformer = True 
use_bert = True 
for params in itertools.product(lrs, lrs, wds, wds):
	lr_2, lr_3, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'encoder_fn_name': encoder_fn_name,
					'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'measurements': measurements,
					'batch_size': batch_size,
					'lens_lr': lr_2, 
					'predictor_lr': lr_3,
					'lens_wd':  wd_2,
					'predictor_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'use_transformer': use_transformer,
					'use_bert': use_bert,
					'restore_transformer_dir': restore_transformer_dir,
					'index': index
				 }
	# params_combinations.append(param_dict)
	# index_to_params[index] = param_dict
	count += 1


# CNN experiments
reduce_fn_name = 'linear_max_pool'
reduce_fn_kwargs_path = 'linear_pool_1024'
epochs = 10
measurements = 2
batch_size = 512
lrs = [1e-3, 1e-4, 1e-5]
wds = [0.0, 0.1, 0.2]
train_families = 10000
lens_train_samples = 50
for params in itertools.product(lrs, lrs, lrs, wds, wds, wds):
	lr_1, lr_2, lr_3, wd_1, wd_2, wd_3 = params
	index = '%08d' % count
	param_dict = {
					'reduce_fn_name': reduce_fn_name,
					'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
					'epochs': epochs,
					'measurements': measurements,
					'lens_batch_size': batch_size,
					'knn_batch_size': batch_size,
					'encoder_lr': lr_1,
					'lens_lr': lr_2, 
					'predictor_lr': lr_3,
					'encoder_wd': wd_1,
					'lens_wd':  wd_2,
					'predictor_wd': wd_3,
					'train_families': train_families,
					'lens_train_samples': lens_train_samples,
					'index': index
				 }
	params_combinations.append(param_dict)
	index_to_params[index] = param_dict
	count += 1


np.random.seed(0)
params_combinations = list(np.random.permutation(np.array(params_combinations)))
with open('params_combinations.json', 'w') as f:
	json.dump(params_combinations, f)

with open('index_to_params.json', 'w') as f:
	json.dump(index_to_params, f)

print(count)


