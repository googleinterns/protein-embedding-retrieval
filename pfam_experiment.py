"""Lens training + nearest neighbors classification pipeline."""

import os

import sys
sys.path.insert(1, 'google_research/')

import flax
from flax import nn
from flax.training import checkpoints

import jax
from jax import random
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

from contextual_lenses.pfam_utils import get_family_ids, PFAM_NUM_CATEGORIES, \
pfam_evaluate, create_pfam_batches, pfam_nearest_neighbors_classification

from contextual_lenses.load_transformer import load_transformer_params

from absl import app, flags

# Define flags.
FLAGS = flags.FLAGS

flags.DEFINE_string('encoder_fn_name', 'cnn_one_hot',
                    'Name of encoder_fn to use. None if using Transformer.')
flags.DEFINE_string('encoder_fn_kwargs_path', 'cnn_kwargs',
                    'Path to encoder_fn_kwargs.')
flags.DEFINE_string('reduce_fn_name', 'linear_max_pool',
                    'Name of reduce_fn to use.')
flags.DEFINE_string('reduce_fn_kwargs_path', 'linear_pool_1024',
                    'Path to reduce_fn_kwargs.')

flags.DEFINE_integer('epochs', 10, 'Number of epochs for lens training.')
flags.DEFINE_integer(
    'measurements', 1,
    'Number of times to interrupt lens training loop to take measurements (1 = no interruption).'
)
flags.DEFINE_integer('lens_batch_size', 64, 'Batch size for lens training.')
flags.DEFINE_integer('knn_batch_size', 64,
                     'Batch size for KNN vector computation.')

flags.DEFINE_float('encoder_lr', 0.0, 'Encoder learning rate.')
flags.DEFINE_float('lens_lr', 1e-5, 'Lens learning rate.')
flags.DEFINE_float('predictor_lr', 1e-3, 'Predictor learning rate.')

flags.DEFINE_float('encoder_wd', 0.0, 'Encoder weight decay.')
flags.DEFINE_float('lens_wd', 0.0, 'Lens weight decay.')
flags.DEFINE_float('predictor_wd', 0.0, 'Predictor weight decay.')

flags.DEFINE_integer('train_families', 10000,
                     'Number of famlies used to train lens.')
flags.DEFINE_integer('lens_train_samples', 50,
                     'Number of samples used to train lens.')
flags.DEFINE_integer('first_test_family', 15001, 'First family to test on.')
flags.DEFINE_integer('last_test_family', 16000, 'Last family to test on.')

flags.DEFINE_integer('lens_shuffle_seed', 0,
                     'Random seed used for lens training data batching.')
flags.DEFINE_integer('lens_sample_random_state', 0,
                     'Random state used for lens training data sampling.')
flags.DEFINE_integer('knn_shuffle_seed', 1,
                     'Random seed used for KNN data batching.')
flags.DEFINE_integer('knn_sample_random_state', 1,
                     'Random state used for KNN data sampling.')
flags.DEFINE_integer('model_random_key', 0,
                     'Random key used for model instantiation.')

flags.DEFINE_boolean('use_transformer', False,
                     'Whether or not to use transformer encoder')
flags.DEFINE_boolean('use_bert', False,
                     'Whether or not to use bidirectional transformer.')
flags.DEFINE_string('restore_transformer_dir', None,
                    'Directory to load pretrained transformer from.')

flags.DEFINE_string('gcs_bucket', 'sequin-public',
                    'GCS bucket to save to and load from.')
flags.DEFINE_string('data_partitions_dirpath', 'random_split/',
                    'Location of Pfam data in GCS bucket.')

flags.DEFINE_string('results_save_dir', '',
                    'Directory in GCS bucket to save to.')

flags.DEFINE_boolean('load_model', False,
                     'Whether or not to load a trained model.')
flags.DEFINE_string('load_model_dir', '',
                    'Directory in GCS bucket to load trained optimizer from.')
flags.DEFINE_integer(
    'load_model_step', 0,
    'Number of steps optimizer to be loaded has been trained for.')
flags.DEFINE_boolean('save_model', False,
                     'Whether or not to save trained model.')
flags.DEFINE_string('save_model_dir', '',
                    'Directory in GCS bucket to save trained optimizer to.')

flags.DEFINE_string('label', '', 'Label used to save experiment results.')


def get_model_kwargs(encoder_fn_name, encoder_fn_kwargs_path, reduce_fn_name,
                     reduce_fn_kwargs_path):
    """Determines model components using string names."""

    encoder_fn = encoder_fn_name_to_fn(encoder_fn_name)
    encoder_fn_kwargs = json.load(
        open(
            resource_filename(
                'contextual_lenses.resources',
                os.path.join('encoder_fn_kwargs_resources',
                             encoder_fn_kwargs_path + '.json'))))

    reduce_fn = reduce_fn_name_to_fn(reduce_fn_name)
    reduce_fn_kwargs = json.load(
        open(
            resource_filename(
                'contextual_lenses.resources',
                os.path.join('reduce_fn_kwargs_resources',
                             reduce_fn_kwargs_path + '.json'))))

    layers, trainable_encoder = architecture_to_layers(encoder_fn_name,
                                                       reduce_fn_name)

    return encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs, layers


def create_model(use_transformer,
                 use_bert,
                 restore_transformer_dir,
                 encoder_fn,
                 encoder_fn_kwargs,
                 reduce_fn,
                 reduce_fn_kwargs,
                 layers,
                 output='prediction',
                 encoder_fn_params=None,
                 reduce_fn_params=None,
                 predict_fn_params=None,
                 random_key=0):
    """Creates representation model (encoder --> lens --> predictor) architecture."""

    family_ids = get_family_ids()
    num_families = len(family_ids)

    if use_transformer:

        if use_bert:
            model_cls = models.FlaxBERT
        else:
            model_cls = models.FlaxLM

        if encoder_fn_params is not None:
            pretrained_transformer_params = encoder_fn_params
        else:
            if restore_transformer_dir is not None:
                pretrained_transformer_params = load_transformer_params(
                    restore_transformer_dir, model_cls)
            else:
                pretrained_transformer_params = None

        model = create_transformer_representation_model(
            transformer_kwargs=encoder_fn_kwargs,
            reduce_fn=reduce_fn,
            reduce_fn_kwargs=reduce_fn_kwargs,
            num_categories=PFAM_NUM_CATEGORIES,
            output_features=num_families,
            bidirectional=use_bert,
            output=output,
            key=random.PRNGKey(random_key),
            encoder_fn_params=pretrained_transformer_params,
            reduce_fn_params=reduce_fn_params,
            predict_fn_params=predict_fn_params)

    else:
        model = create_representation_model(
            encoder_fn=encoder_fn,
            encoder_fn_kwargs=encoder_fn_kwargs,
            reduce_fn=reduce_fn,
            reduce_fn_kwargs=reduce_fn_kwargs,
            num_categories=PFAM_NUM_CATEGORIES,
            output_features=num_families,
            output=output,
            key=random.PRNGKey(random_key),
            encoder_fn_params=encoder_fn_params,
            reduce_fn_params=reduce_fn_params,
            predict_fn_params=predict_fn_params)

    return model


def set_model_parameters(model, params):
    """Updates a model's parameters using a parameters dictionary."""

    params = copy.deepcopy(params)

    assert (
        model.params.keys() == params.keys()), 'Model parameters do not match!'

    for layer in model.params.keys():
        model.params[layer] = params[layer]

    return model


def measure_nearest_neighbor_performance(accuracy_label, encoder,
                                         family_accessions, batch_size,
                                         train_samples, shuffle_seed,
                                         sample_random_state):
    """Measures nearest neighbor classification performance and updates datum."""

    results = pfam_nearest_neighbors_classification(
        encoder=encoder,
        family_accessions=family_accessions,
        batch_size=batch_size,
        train_samples=train_samples,
        shuffle_seed=shuffle_seed,
        sample_random_state=sample_random_state,
        data_partitions_dirpath=FLAGS.data_partitions_dirpath,
        gcs_bucket=FLAGS.gcs_bucket)[0]

    accuracy = results['1-nn accuracy']

    accuracy_dict = {accuracy_label: accuracy}

    return accuracy_dict


# Train lens and measure performance of lens and nearest neighbors classifier.
def main(_):

    if FLAGS.use_transformer:
        assert (
            FLAGS.encoder_fn_name == 'transformer'
        ), 'encoder_fn_name must be transformer if use_transformer is True!'

    assert (FLAGS.epochs % FLAGS.measurements == 0
            ), 'Number of measurements must divide number of epochs!'
    measurement_epochs = FLAGS.epochs // FLAGS.measurements

    assert FLAGS.results_save_dir != '', 'Specify results_save_dir!'

    assert FLAGS.label != '', 'Specify label!'

    if FLAGS.load_model:
        assert FLAGS.load_model_dir != '', 'Specify load_model_dir!'
        assert FLAGS.load_model_step > 0, 'Loaded model must have been trained for more than 0 steps.'

    if FLAGS.save_model:
        assert FLAGS.save_model_dir != '', 'Specify save_model_dir!'

    datum = {
        'label': FLAGS.label,
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
        'lens_train_samples': FLAGS.lens_train_samples,
        'first_test_family': FLAGS.first_test_family,
        'last_test_family': FLAGS.last_test_family,
        'lens_shuffle_seed': FLAGS.lens_shuffle_seed,
        'lens_sample_random_state': FLAGS.lens_sample_random_state,
        'knn_shuffle_seed': FLAGS.knn_shuffle_seed,
        'knn_sample_random_state': FLAGS.knn_sample_random_state,
        'model_random_key': FLAGS.model_random_key,
        'use_transformer': FLAGS.use_transformer,
        'use_bert': FLAGS.use_bert,
        'restore_transformer_dir': FLAGS.restore_transformer_dir,
        'gcs_bucket': FLAGS.gcs_bucket,
        'data_partitions_dirpath': FLAGS.data_partitions_dirpath,
        'results_save_dir': FLAGS.results_save_dir,
        'load_model': FLAGS.load_model,
        'load_model_dir': FLAGS.load_model_dir,
        'load_model_step': FLAGS.load_model_step,
        'save_model': FLAGS.save_model,
        'save_model_dir': FLAGS.save_model_dir
    }

    gcsfs = GCSFS(FLAGS.gcs_bucket)

    print(datum)
    df = pd.DataFrame([datum])
    with gcsfs.open(os.path.join(FLAGS.results_save_dir, FLAGS.label + '.csv'),
                    'w') as gcs_file:
        df.to_csv(gcs_file, index=False)

    knn_train_samples_ = [1, 5, 10, 50]

    family_ids = get_family_ids()
    num_families = len(family_ids)
    loss_fn_kwargs = {'num_classes': num_families}

    lens_knn_train_family_accessions = []
    for _ in range(1, FLAGS.train_families + 1):
        family_name = 'PF%05d' % _
        lens_knn_train_family_accessions.append(family_name)

    knn_test_family_accessions = []
    for _ in range(FLAGS.first_test_family, FLAGS.last_test_family + 1):
        family_name = 'PF%05d' % _
        knn_test_family_accessions.append(family_name)

    encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs, layers = get_model_kwargs(
        encoder_fn_name=FLAGS.encoder_fn_name,
        encoder_fn_kwargs_path=FLAGS.encoder_fn_kwargs_path,
        reduce_fn_name=FLAGS.reduce_fn_name,
        reduce_fn_kwargs_path=FLAGS.reduce_fn_kwargs_path)

    embedding_model = create_model(
        use_transformer=FLAGS.use_transformer,
        use_bert=FLAGS.use_bert,
        restore_transformer_dir=FLAGS.restore_transformer_dir,
        encoder_fn=encoder_fn,
        encoder_fn_kwargs=encoder_fn_kwargs,
        reduce_fn=reduce_fn,
        reduce_fn_kwargs=reduce_fn_kwargs,
        layers=layers,
        output='embedding',
        random_key=FLAGS.model_random_key)

    datum.update(
        measure_nearest_neighbor_performance(
            accuracy_label=
            'train_knn_accuracy_untrained_lens_1_knn_train_samples',
            encoder=embedding_model,
            family_accessions=lens_knn_train_family_accessions,
            batch_size=FLAGS.knn_batch_size,
            train_samples=1,
            shuffle_seed=FLAGS.knn_shuffle_seed,
            sample_random_state=FLAGS.knn_sample_random_state))

    for knn_train_samples in knn_train_samples_:

        datum.update(
            measure_nearest_neighbor_performance(
                accuracy_label='test_knn_accuracy_untrained_lens_' +
                str(knn_train_samples) + '_knn_train_samples',
                encoder=embedding_model,
                family_accessions=knn_test_family_accessions,
                batch_size=FLAGS.knn_batch_size,
                train_samples=knn_train_samples,
                shuffle_seed=FLAGS.knn_shuffle_seed,
                sample_random_state=FLAGS.knn_sample_random_state))

    encoder_fn_params = None
    reduce_fn_params = None
    predict_fn_params = None

    model = create_model(use_transformer=FLAGS.use_transformer,
                         use_bert=FLAGS.use_bert,
                         restore_transformer_dir=FLAGS.restore_transformer_dir,
                         encoder_fn=encoder_fn,
                         encoder_fn_kwargs=encoder_fn_kwargs,
                         reduce_fn=reduce_fn,
                         reduce_fn_kwargs=reduce_fn_kwargs,
                         layers=layers,
                         output='prediction',
                         encoder_fn_params=encoder_fn_params,
                         reduce_fn_params=reduce_fn_params,
                         predict_fn_params=predict_fn_params,
                         random_key=FLAGS.model_random_key)

    optimizer = create_optimizer(
        model=model,
        learning_rate=[FLAGS.encoder_lr, FLAGS.lens_lr, FLAGS.predictor_lr],
        weight_decay=[FLAGS.encoder_wd, FLAGS.lens_wd, FLAGS.predictor_wd],
        layers=layers)

    if FLAGS.load_model:
        optimizer = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(
            'gs://' + FLAGS.gcs_bucket, FLAGS.load_model_dir),
                                                   target=optimizer,
                                                   step=FLAGS.load_model_step)

        trained_params = optimizer.target.params
        embedding_model = set_model_parameters(model=embedding_model,
                                               params=trained_params)

    if FLAGS.save_model:
        checkpoints.save_checkpoint(ckpt_dir=os.path.join(
            'gs://' + FLAGS.gcs_bucket, FLAGS.save_model_dir),
                                    target=optimizer,
                                    step=FLAGS.load_model_step)

    for i in range(FLAGS.measurements):

        train_batches, train_indexes = create_pfam_batches(
            family_accessions=lens_knn_train_family_accessions,
            batch_size=FLAGS.lens_batch_size,
            samples=FLAGS.lens_train_samples,
            epochs=measurement_epochs,
            drop_remainder=True,
            shuffle_seed=FLAGS.lens_shuffle_seed + i,
            sample_random_state=FLAGS.lens_sample_random_state)

        optimizer = train(
            model=optimizer.target,
            train_data=train_batches,
            loss_fn=cross_entropy_loss,
            loss_fn_kwargs=loss_fn_kwargs,
            learning_rate=[
                FLAGS.encoder_lr, FLAGS.lens_lr, FLAGS.predictor_lr
            ],
            weight_decay=[FLAGS.encoder_wd, FLAGS.lens_wd, FLAGS.predictor_wd],
            layers=layers)

        results, preds = pfam_evaluate(
            predict_fn=optimizer.target,
            test_family_accessions=lens_knn_train_family_accessions,
            title=None,
            loss_fn_kwargs=loss_fn_kwargs,
            batch_size=FLAGS.lens_batch_size,
            data_partitions_dirpath=FLAGS.data_partitions_dirpath,
            gcs_bucket=FLAGS.gcs_bucket)

        lens_accuracy = results['accuracy']
        datum['lens_accuracy' + '_measurement_' + str(i)] = lens_accuracy

        lens_cross_entropy = float(results['cross_entropy'])
        datum['lens_cross_entropy' + '_measurement_' +
              str(i)] = lens_cross_entropy

        trained_params = optimizer.target.params
        embedding_model = set_model_parameters(model=embedding_model,
                                               params=trained_params)

        datum.update(
            measure_nearest_neighbor_performance(
                accuracy_label=
                'train_knn_accuracy_trained_lens_1_knn_train_samples' +
                '_measurement_' + str(i),
                encoder=embedding_model,
                family_accessions=lens_knn_train_family_accessions,
                batch_size=FLAGS.knn_batch_size,
                train_samples=1,
                shuffle_seed=FLAGS.knn_shuffle_seed,
                sample_random_state=FLAGS.knn_sample_random_state))

        for knn_train_samples in knn_train_samples_:

            datum.update(
                measure_nearest_neighbor_performance(
                    accuracy_label='test_knn_accuracy_trained_lens_' +
                    str(knn_train_samples) + '_knn_train_samples' +
                    '_measurement_' + str(i),
                    encoder=embedding_model,
                    family_accessions=knn_test_family_accessions,
                    batch_size=FLAGS.knn_batch_size,
                    train_samples=knn_train_samples,
                    shuffle_seed=FLAGS.knn_shuffle_seed,
                    sample_random_state=FLAGS.knn_sample_random_state))

    print(datum)
    df = pd.DataFrame([datum])
    with gcsfs.open(os.path.join(FLAGS.results_save_dir, FLAGS.label + '.csv'),
                    'w') as gcs_file:
        df.to_csv(gcs_file, index=False)

    if FLAGS.save_model:
        checkpoints.save_checkpoint(ckpt_dir=os.path.join(
            'gs://' + FLAGS.gcs_bucket, FLAGS.save_model_dir),
                                    target=optimizer,
                                    step=FLAGS.load_model_step + FLAGS.epochs)


if __name__ == '__main__':
    app.run(main)
