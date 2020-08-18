"""Lens training + nearest neighbors classification pipeline."""


import os

import sys
sys.path.insert(1, 'google_research/')

import flax
from flax import nn
from flax.training import checkpoints

import jax
import jax.numpy as jnp
from jax.config import config
config.enable_omnistaging()

import numpy as np

import pandas as pd

import json

import copy

import time

from pkg_resources import resource_filename

from fs_gcsfs import GCSFS

from google_research.protein_lm import domains, models

from contextual_lenses.contextual_lenses import reduce_fn_name_to_fn

from contextual_lenses.train_utils import create_optimizer, train, \
create_representation_model, create_transformer_representation_model, \
architecture_to_layers

from contextual_lenses.encoders import encoder_fn_name_to_fn

from contextual_lenses.loss_fns import cross_entropy_loss

from contextual_lenses.pfam_utils import family_ids, pfam_num_categories, \
pfam_evaluate, create_pfam_batches, pfam_nearest_neighbors_classification

from contextual_lenses.load_transformer import load_transformer_params

from absl import app, flags


# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('encoder_fn_name', 'cnn_one_hot', 'Name of encoder_fn to use. None if using Transformer.')
flags.DEFINE_string('encoder_fn_kwargs_path', 'cnn_kwargs', 'Path to encoder_fn_kwargs.')
flags.DEFINE_string('reduce_fn_name', 'linear_max_pool', 'Name of reduce_fn to use.')
flags.DEFINE_string('reduce_fn_kwargs_path', 'linear_pool_1024', 'Path to reduce_fn_kwargs.')

flags.DEFINE_integer('epochs', 10, 'Number of epochs for lens training.')
flags.DEFINE_integer('measurements', 1, 'Number of times to interrupt lens training loop to take measurements (1 = no interruption).')
flags.DEFINE_integer('lens_batch_size', 64, 'Batch size for lens training.')
flags.DEFINE_integer('knn_batch_size', 64, 'Batch size for KNN vector computation.')

flags.DEFINE_float('encoder_lr', 0.0, 'Encoder learning rate.')
flags.DEFINE_float('lens_lr', 1e-3, 'Lens learning rate.')
flags.DEFINE_float('predictor_lr', 1e-3, 'Predictor learning rate.')

flags.DEFINE_float('encoder_wd', 0.0, 'Encoder weight decay.')
flags.DEFINE_float('lens_wd', 0.0, 'Lens weight decay.')
flags.DEFINE_float('predictor_wd', 0.0, 'Predictor weight decay.')

flags.DEFINE_integer('train_families', 1000, 'Number of famlies used to train lens.')
flags.DEFINE_integer('lens_train_samples', 50, 'Number of samples used to train lens.')
flags.DEFINE_integer('first_test_family', 15000, 'First family to test on.')
flags.DEFINE_integer('last_test_family', 16000, 'Last family to test on.')


flags.DEFINE_boolean('use_transformer', False, 'Whether or not to use transformer encoder')
flags.DEFINE_boolean('use_bert', False, 'Whether or not to use bidirectional transformer.')
flags.DEFINE_string('restore_transformer_dir', None, 'Directory to load pretrained transformer from.')

flags.DEFINE_string('gcs_bucket', 'sequin-public', 'GCS bucket to save to and load from.')
flags.DEFINE_string('save_dir', 'second_batch', 'Directory in GCS bucket to save to.')
flags.DEFINE_string('index', '00000000', 'Index used to save experiment results.')

flags.DEFINE_integer('sleep_time', 600, 'Max number of seconds to sleep for before job starts, used to balance cloud quotas.')


# Train lens and measure performance of lens and nearest neighbors classifier.
def main(_):

	if FLAGS.sleep_time > 0:
		time.sleep(np.random.uniform(0, FLAGS.sleep_time))

	if FLAGS.use_transformer:
		assert(FLAGS.encoder_fn_name=='transformer'), 'encoder_fn_name must be \'transformer\' if \'use_transformer\' is True'

	assert(FLAGS.epochs % FLAGS.measurements == 0), 'Number of measurements must divide number of epochs!'
	measurement_epochs = FLAGS.epochs // FLAGS.measurements

	datum = {
				'index': FLAGS.index,
				'encoder_fn_name': FLAGS.encoder_fn_name,
				'encoder_fn_kwargs_path': FLAGS.encoder_fn_kwargs_path,
				'reduce_fn_name': FLAGS.reduce_fn_name,
				'reduce_fn_kwargs_path': FLAGS.reduce_fn_kwargs_path,
				'epochs': FLAGS.epochs,
				'measurements': FLAGS.measurements,
				'lens_batch_size': FLAGS.lens_batch_size,
				'knn_batch_size': FLAGS.knn_batch_size,
				'encoder_lr': FLAGS.encoder_lr,
				'lens_lr': FLAGS.lens_lr,
				'predictor_lr': FLAGS.predictor_lr,
				'encoder_wd': FLAGS.encoder_wd,
				'lens_wd': FLAGS.lens_wd,
				'predictor_wd': FLAGS.predictor_wd,
				'train_families': FLAGS.train_families,
				'use_transformer': FLAGS.use_transformer,
				'use_bert': FLAGS.use_bert,
				'restore_transformer_dir': FLAGS.restore_transformer_dir
			}

	knn_train_samples_ = [1, 5, 10, 50]

	gcsfs = GCSFS(FLAGS.gcs_bucket)

	num_families = len(family_ids)
	loss_fn_kwargs = {
	  	'num_classes': num_families
	}

	lens_knn_train_family_accessions = []
	for _ in range(1, FLAGS.train_families+1):
  		family_name = 'PF%05d' % _
  		lens_knn_train_family_accessions.append(family_name)
	
	knn_test_family_accessions = []
	for _ in range(FLAGS.first_test_family+1, FLAGS.last_test_family+1):
		family_name = 'PF%05d' % _
		knn_test_family_accessions.append(family_name)

	def create_model(use_transformer, use_bert, restore_transformer_dir, encoder_fn, encoder_fn_kwargs,
					 reduce_fn, reduce_fn_kwargs, layers, output='prediction', 
					 encoder_fn_params=None, reduce_fn_params=None, predict_fn_params=None):
		
		if use_transformer:

			if use_bert:
				model_cls = models.FlaxBERT
			else:
				model_cls = models.FlaxLM

			if encoder_fn_params is not None:
				pretrained_transformer_params = encoder_fn_params
			else:
				if FLAGS.restore_transformer_dir is not None:
					pretrained_transformer_params = load_transformer_params(FLAGS.restore_transformer_dir, model_cls)
				else:
					pretrained_transformer_params = None

			model = create_transformer_representation_model(transformer_kwargs=encoder_fn_kwargs,
		                                                    reduce_fn=reduce_fn,
		                                                    reduce_fn_kwargs=reduce_fn_kwargs,
		                                                    num_categories=pfam_num_categories,
		                                                    output_features=num_families,
		                                                    output=output,
		                                                    bidirectional=FLAGS.use_bert,
		                                                    encoder_fn_params=pretrained_transformer_params,
		                                                    reduce_fn_params=reduce_fn_params,
		                                                    predict_fn_params=predict_fn_params)

		else:
			model = create_representation_model(encoder_fn=encoder_fn,
			                                    encoder_fn_kwargs=encoder_fn_kwargs,
			                                    reduce_fn=reduce_fn,
			                                    reduce_fn_kwargs=reduce_fn_kwargs,
			                                    num_categories=pfam_num_categories,
			                                    output_features=num_families,
			                                  	output=output,
			                                  	encoder_fn_params=encoder_fn_params,
		                                        reduce_fn_params=reduce_fn_params,
		                                        predict_fn_params=predict_fn_params)

		return model

	encoder_fn = encoder_fn_name_to_fn(FLAGS.encoder_fn_name)
	encoder_fn_kwargs = json.load(open(resource_filename('contextual_lenses.resources', os.path.join('encoder_fn_kwargs_resources', FLAGS.encoder_fn_kwargs_path + '.json'))))

	reduce_fn = reduce_fn_name_to_fn(FLAGS.reduce_fn_name)
	reduce_fn_kwargs = json.load(open(resource_filename('contextual_lenses.resources', os.path.join('reduce_fn_kwargs_resources', FLAGS.reduce_fn_kwargs_path + '.json'))))

	layers, trainable_encoder = architecture_to_layers(FLAGS.encoder_fn_name, FLAGS.reduce_fn_name)

	embedding_model = create_model(use_transformer=FLAGS.use_transformer, use_bert=FLAGS.use_bert, 
								   restore_transformer_dir=FLAGS.restore_transformer_dir, encoder_fn=encoder_fn, 
								   encoder_fn_kwargs=encoder_fn_kwargs, reduce_fn=reduce_fn, reduce_fn_kwargs=reduce_fn_kwargs,
								   layers=layers, output='embedding')

	embedding_optimizer = create_optimizer(embedding_model,learning_rate=[FLAGS.encoder_lr, FLAGS.lens_lr, FLAGS.predictor_lr], 
										   weight_decay=[FLAGS.encoder_wd, FLAGS.lens_wd, FLAGS.predictor_wd], layers=layers)

	train_knn_results_untrained_lens = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
					                                                         train_family_accessions=lens_knn_train_family_accessions, 
					                                                         test_family_accessions=lens_knn_train_family_accessions,
					                                                         batch_size=FLAGS.knn_batch_size,
					                                                         train_samples=1,
					                                                         shuffle_seed=1,
					                                                         sample_random_state=1)[0]
	train_knn_accuracy_untrained_lens = train_knn_results_untrained_lens['1-nn accuracy']
	datum['train_knn_accuracy_untrained_lens_1_knn_train_samples'] = train_knn_accuracy_untrained_lens

	for knn_train_samples in knn_train_samples_:

		test_knn_results_untrained_lens = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
					                                                            train_family_accessions=knn_test_family_accessions, 
					                                                            test_family_accessions=knn_test_family_accessions,
					                                                            batch_size=FLAGS.knn_batch_size,
					                                                            train_samples=knn_train_samples,
					                                                            shuffle_seed=1,
					                                                            sample_random_state=1)[0]
		test_knn_accuracy_untrained_lens = test_knn_results_untrained_lens['1-nn accuracy']
		datum['test_knn_accuracy_untrained_lens_' + str(knn_train_samples) + '_knn_train_samples'] = test_knn_accuracy_untrained_lens

	encoder_fn_params = None
	reduce_fn_params = None
	predict_fn_params = None
	model = create_model(use_transformer=FLAGS.use_transformer, use_bert=FLAGS.use_bert, restore_transformer_dir=FLAGS.restore_transformer_dir,
							 encoder_fn=encoder_fn, encoder_fn_kwargs=encoder_fn_kwargs, reduce_fn=reduce_fn, reduce_fn_kwargs=reduce_fn_kwargs, layers=layers, 
							 output='prediction', encoder_fn_params=encoder_fn_params, reduce_fn_params=reduce_fn_params, predict_fn_params=predict_fn_params)
	
	optimizer = create_optimizer(model=model, 
								 learning_rate=[FLAGS.encoder_lr, FLAGS.lens_lr, FLAGS.predictor_lr], 
								 weight_decay=[FLAGS.encoder_wd, FLAGS.lens_wd, FLAGS.predictor_wd],
								 layers=layers)

	for i in range(FLAGS.measurements):

		train_batches, train_indexes = create_pfam_batches(family_accessions=lens_knn_train_family_accessions,
														   batch_size=FLAGS.lens_batch_size,
														   samples=FLAGS.lens_train_samples,
														   epochs=measurement_epochs, 
														   drop_remainder=True)

		optimizer = train(model=optimizer.target,
	                      train_data=train_batches,
	                      loss_fn=cross_entropy_loss,
	                      loss_fn_kwargs=loss_fn_kwargs,
	                      learning_rate=[FLAGS.encoder_lr, FLAGS.lens_lr, FLAGS.predictor_lr],
	                      weight_decay=[FLAGS.encoder_wd, FLAGS.lens_wd, FLAGS.predictor_wd],
	                      layers=layers)

		trained_params = copy.deepcopy(optimizer.target.params)

		results, preds = pfam_evaluate(predict_fn=optimizer.target,
	                                   test_family_accessions=lens_knn_train_family_accessions,
	                                   title=None,
	                                   loss_fn_kwargs=loss_fn_kwargs,
	                                   batch_size=FLAGS.lens_batch_size)

		lens_accuracy = results['accuracy']
		datum['lens_accuracy' + '_measurement_' + str(i)] = lens_accuracy

		lens_cross_entropy = float(results['cross_entropy'])
		datum['lens_cross_entropy' + '_measurement_' + str(i)] = lens_cross_entropy

		assert(embedding_optimizer.target.params.keys()==trained_params.keys()), 'Optimizer parameters do not match!'
		for layer in embedding_optimizer.target.params.keys():
			embedding_optimizer.target.params[layer] = trained_params[layer]

		train_knn_results_trained_lens = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
					                                                           train_family_accessions=lens_knn_train_family_accessions, 
					                                                           test_family_accessions=lens_knn_train_family_accessions,
					                                                           batch_size=FLAGS.knn_batch_size,
					                                                           train_samples=1,
					                                                           shuffle_seed=1,
					                                                           sample_random_state=1)[0]
		train_knn_accuracy_trained_lens = train_knn_results_trained_lens['1-nn accuracy']
		datum['train_knn_accuracy_trained_lens_1_knn_train_samples' + '_measurement_' + str(i)] = train_knn_accuracy_trained_lens

		for knn_train_samples in knn_train_samples_:

			test_knn_results_trained_lens = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
					                                                              train_family_accessions=knn_test_family_accessions, 
					                                                              test_family_accessions=knn_test_family_accessions,
					                                                              batch_size=FLAGS.knn_batch_size,
					                                                              train_samples=knn_train_samples,
					                                                              shuffle_seed=1,
					                                                              sample_random_state=1)[0]
			test_knn_accuracy_trained_lens = test_knn_results_trained_lens['1-nn accuracy']
			datum['test_knn_accuracy_trained_lens_' + str(knn_train_samples) + '_knn_train_samples' + '_measurement_' + str(i)] = test_knn_accuracy_trained_lens

	print(datum)
	df = pd.DataFrame([datum])

	with gcsfs.open(os.path.join(FLAGS.save_dir, FLAGS.index + '.csv'), 'w') as gcs_file:
		df.to_csv(gcs_file, index=False)


if __name__ == '__main__':
	app.run(main)
