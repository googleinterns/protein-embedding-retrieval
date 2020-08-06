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

from pkg_resources import resource_filename

from fs_gcsfs import GCSFS

from google_research.protein_lm import domains, models

from contextual_lenses.contextual_lenses import reduce_fn_name_to_fn

from contextual_lenses.train_utils import create_optimizer, train, \
create_representation_model, create_transformer_representation_model, \
architecture_to_layers

from contextual_lenses.encoders import encoder_fn_name_to_fn

from contextual_lenses.loss_fns import cross_entropy_loss

from contextual_lenses.pfam_family_classification.pfam_utils import family_ids, \
pfam_num_categories, create_pfam_batches, pfam_nearest_neighbors_classification

from contextual_lenses.load_transformer import load_transformer_params

from absl import app, flags


# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string('encoder_fn_name', None, 'Name of encoder_fn to use. None if using Transformer.')
flags.DEFINE_string('encoder_fn_kwargs_path', 'medium_transformer_kwargs.json', 'Path to encoder_fn_kwargs.')
flags.DEFINE_string('reduce_fn_name', 'linear_max_pool', 'Name of reduce_fn to use.')
flags.DEFINE_string('reduce_fn_kwargs_path', 'linear_pool_1024.json', 'Path to reduce_fn_kwargs.')

flags.DEFINE_integer('epochs', 1, 'Number of epochs for lens training.') # 10
flags.DEFINE_list('learning_rate', [0.0, 1e-3, 1e-3], 'Learning rates for encoder, lens, and predictor.')
flags.DEFINE_list('weight_decay', [0.0, 0.0, 0.0], 'Weight decays for encoder, lens, and predictor.')
flags.DEFINE_integer('lens_train_families', 100, 'Number of famlies used to train lens.') # 1000/10000

flags.DEFINE_string('restore_transformer_dir', None, 'Directory to load pretrained transformer from.')
flags.DEFINE_boolean('use_transformer', True, 'Whether or not to use transformer encoder')
flags.DEFINE_boolean('use_bert', True, 'Whether or not to use bidirectional transformer.')

flags.DEFINE_integer('knn_train_samples', 5, 'Number of samples used to train nearest neighbors algorithm.')


# Train lens and measure performance of lens and nearest neighbors classifier.
def main(_):

	gcsfs = GCSFS('sequin-public')

	with gcsfs.open('test_running.txt', 'w') as f:
		f.write('RUNNING!')

	print('LOSS_FN_KWARGS')
	num_families = len(family_ids)
	loss_fn_kwargs = {
	  	'num_classes': num_families
	}

	train_family_accessions = []
	for _ in range(1, FLAGS.lens_train_families+1):
  		family_name = 'PF%05d' % _
  		train_family_accessions.append(family_name)
	
	test_family_accessions = []
	for _ in range(15001, 15101): # 16001
		family_name = 'PF%05d' % _
		test_family_accessions.append(family_name)
	
	train_batches, train_indexes = create_pfam_batches(family_accessions=train_family_accessions,
													   batch_size=128,
													   epochs=FLAGS.epochs, 
													   drop_remainder=True)

	encoder_fn = encoder_fn_name_to_fn(FLAGS.encoder_fn_name)
	encoder_fn_kwargs = json.load(open(resource_filename('contextual_lenses.resources', os.path.join('encoder_fn_kwargs_resources', FLAGS.encoder_fn_kwargs_path))))

	reduce_fn = reduce_fn_name_to_fn(FLAGS.reduce_fn_name)
	reduce_fn_kwargs = json.load(open(resource_filename('contextual_lenses.resources', os.path.join('reduce_fn_kwargs_resources', FLAGS.reduce_fn_kwargs_path))))

	with gcsfs.open('test_load.txt', 'w') as f:
		f.write('LOADED ARGUMENTS!')

	if FLAGS.use_transformer:

		if FLAGS.use_bert:
			model_cls = models.FlaxBERT
		else:
			model_cls = models.FlaxLM

		if FLAGS.restore_transformer_dir is not None:
			pretrained_transformer_params = load_transformer_params(restore_transformer_dir, model_cls)
		else:
			pretrained_transformer_params = None

		model = create_transformer_representation_model(transformer_kwargs=encoder_fn_kwargs,
	                                                    reduce_fn=reduce_fn,
	                                                    reduce_fn_kwargs=reduce_fn_kwargs,
	                                                    num_categories=pfam_num_categories,
	                                                    output_features=num_families,
	                                                    output='prediction',
	                                                    bidirectional=FLAGS.use_bert,
	                                                    encoder_fn_params=pretrained_transformer_params)

		embedding_model = create_transformer_representation_model(transformer_kwargs=encoder_fn_kwargs,
	                                                    		  reduce_fn=reduce_fn,
				                                                  reduce_fn_kwargs=reduce_fn_kwargs,
				                                                  num_categories=pfam_num_categories,
				                                                  output_features=num_families,
				                                                  output='embedding',
				                                                  bidirectional=FLAGS.use_bert,
				                                                  encoder_fn_params=pretrained_transformer_params)
	else:
		model = create_representation_model(encoder_fn=encoder_fn,
		                                    encoder_fn_kwargs=encoder_fn_kwargs,
		                                    reduce_fn=reduce_fn,
		                                    reduce_fn_kwargs=reduce_fn_kwargs,
		                                    num_categories=pfam_num_categories,
		                                    output_features=num_families,
		                                  	output='prediction')

		embedding_model = create_representation_model(encoder_fn=encoder_fn,
				                                      encoder_fn_kwargs=encoder_fn_kwargs,
				                                      reduce_fn=reduce_fn,
				                                      reduce_fn_kwargs=reduce_fn_kwargs,
				                                      num_categories=pfam_num_categories,
				                                      output_features=num_families,
				                                  	  output='embedding')

	layers = architecture_to_layers(FLAGS.encoder_fn_name, FLAGS.reduce_fn_name)
	
	with gcsfs.open('test_model.txt', 'w') as f:
		f.write('CREATED MODEL!')

	optimizer = train(model=model,
                      train_data=train_batches,
                      loss_fn=cross_entropy_loss,
                      loss_fn_kwargs=loss_fn_kwargs,
                      learning_rate=FLAGS.learning_rate,
                      weight_decay=FLAGS.weight_decay,
                      layers=layers)

	with gcsfs.open('test_learning.txt', 'w') as f:
		f.write('MODEL TRAINED!')

	title = str(FLAGS.encoder_fn_name) + '-' + FLAGS.encoder_fn_kwargs_path + '-' + FLAGS.reduce_fn_name + '-' + \
			FLAGS.reduce_fn_kwargs_path + '-' + str(FLAGS.epochs) + ' epochs' + '-' + \
			'learning_rate: ' + str(FLAGS.learning_rate) + '-' + 'weight_decay: ' + str(FLAGS.weight_decay) + \
			str(FLAGS.lens_train_families) + ' lens_train_families' + '-' + 'transformer: ' + str(FLAGS.use_transformer)
	
	if FLAGS.use_transformer:
		title = title + '-' + 'bert: ' + str(FLAGS.use_bert) + '-' + 'pretrained_dir: ' + str(FLAGS.restore_transformer_dir)
	
	title = title + '-' + str(FLAGS.knn_train_samples) + ' knn_train_samples'

	results, preds = pfam_evaluate(predict_fn=optimizer.target,
                                   test_family_accessions=test_family_accessions,
                                   title=title,
                                   loss_fn_kwargs=loss_fn_kwargs,
                                   batch_size=128)
	
	with gcsfs.open('test_eval.txt', 'w') as f:
		f.write('MODEL EVALUATED!')

	lens_accuracy = results['accuracy']
	lens_cross_entropy = results['cross_entropy']

	embedding_optimizer = create_optimizer(embedding_model, 
										   learning_rate=[0.0, 0.0, 0.0], 
										   weight_decay=[0.0, 0.0, 0.0], 
										   layers=layers)

	train_knn_results = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
                                                              train_family_accessions=train_family_accessions, 
                                                              test_family_accessions=train_family_accessions,
                                                              batch_size=128,
                                                              train_samples=FLAGS.knn_train_samples)[0]
	train_knn_accuracy = train_knn_results['1-nn accuracy']
	
	with gcsfs.open('test_train_knn.txt', 'w') as f:
		f.write('TRAIN KNN!')

	test_knn_results = pfam_nearest_neighbors_classification(encoder=embedding_optimizer.target, 
                                                             train_family_accessions=test_family_accessions, 
                                                             test_family_accessions=test_family_accessions,
                                                             batch_size=128,
                                                             train_samples=FLAGS.knn_train_samples)[0]
	test_knn_accuracy = test_knn_results['1-nn accuracy']
	
	with gcsfs.open('test_test_knn.txt', 'w') as f:
		f.write('TEST KNN!')

	datum = {
				'encoder_fn_name': FLAGS.encoder_fn_name,
				'encoder_fn_kwargs_path': FLAGS.encoder_fn_kwargs_path,
				'reduce_fn_name': FLAGS.reduce_fn_name,
				'reduce_fn_kwargs_path': FLAGS.reduce_fn_kwargs_path,
				'epochs': FLAGS.epochs,
				'learning_rate': FLAGS.learning_rate,
				'weight_decay': FLAGS.weight_decay,
				'lens_train_families': FLAGS.lens_train_families,
				'restore_transformer_dir': FLAGS.restore_transformer_dir,
				'use_transformer': FLAGS.use_transformer,
				'use_bert': FLAGS.use_bert,
				'knn_train_samples': FLAGS.knn_train_samples,
				'lens_cross_entropy': lens_cross_entropy,
				'lens_accuracy': lens_accuracy,
				'train_knn_accuracy': train_knn_accuracy,
				'test_knn_accuracy': test_knn_accuracy
			}

	df = pd.DataFrame([datum])
    
	with gcsfs.open(os.path.join('sweep_data', title + '.csv'), 'w') as gcs_file:
		df.to_pickle(gcs_file)

	with gcsfs.open('test_saving.txt', 'w') as f:
		f.write('DATUM SAVED!')


if __name__ == '__main__':
	app.run(main)

