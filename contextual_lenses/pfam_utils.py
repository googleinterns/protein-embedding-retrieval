"""Utils for Pfam family classification experiments."""

import os

import jax
import jax.numpy as jnp

import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats

import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier as knn

from pkg_resources import resource_filename

from fs_gcsfs import GCSFS

from google_research.protein_lm import domains

from contextual_lenses.train_utils import create_data_iterator

from contextual_lenses.loss_fns import cross_entropy_loss


# Data preprocessing.
# Original code source: https://www.kaggle.com/drewbryant/starter-pfam-seed-random-split.
def read_all_shards(partition, data_dir, bucket_name):
    """Combines different CSVs into a single dataframe."""

    shards = []
    gcsfs = GCSFS(bucket_name)
    for fn in gcsfs.listdir(os.path.join(data_dir, partition)):
        with gcsfs.open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
            
    return pd.concat(shards)


def mod_family_accession(family_accession):
    """Reduces family accession to everything prior to '.'."""

    return family_accession[:family_accession.index('.')]


# Pfam protein_lm domain.
PFAM_PROTEIN_DOMAIN = domains.VariableLengthDiscreteDomain(
    vocab=domains.ProteinVocab(include_anomalous_amino_acids=True,
                               include_bos=True,
                               include_eos=True,
                               include_pad=True,
                               include_mask=True),
    length=512)


# Number of categories for one-hot encoding.
PFAM_NUM_CATEGORIES = 27


def residues_to_one_hot_inds(seq):
    """Converts amino acid residues to one hot indices."""

    one_hot_inds = PFAM_PROTEIN_DOMAIN.encode([seq])[0]

    return one_hot_inds


def get_family_ids():
    """Pfam family ids."""

    family_ids = open(
        resource_filename('contextual_lenses.resources', 'pfam_family_ids.txt'),
        'r').readlines()

    return family_ids

def get_family_id_to_index():
    """Dictionary mapping family id to index."""
   
    family_ids = open(
        resource_filename('contextual_lenses.resources', 'pfam_family_ids.txt'),
        'r').readlines()
    family_id_to_index = {}
    for i, family_id in enumerate(family_ids):
        family_id_to_index[family_id.replace('\n', '')] = i

    return family_id_to_index


def create_pfam_df(family_accessions,
                   test=False,
                   samples=None,
                   random_state=0,
                   data_partitions_dirpath='random_split/',
                   gcs_bucket='neuralblast_public'):
    """Processes Pfam data into a featurized dataframe with samples many entries per family."""

    family_id_to_index = get_family_id_to_index()
    
    if test:
        pfam_df = read_all_shards(partition='test',
                                  data_dir=data_partitions_dirpath,
                                  bucket_name=gcs_bucket)
    else:
        pfam_df = read_all_shards(partition='train',
                                  data_dir=data_partitions_dirpath,
                                  bucket_name=gcs_bucket)

    pfam_df['mod_family_accession'] = pfam_df.family_accession.apply(
        lambda x: mod_family_accession(x))
    pfam_df = pfam_df[pfam_df.mod_family_accession.isin(family_accessions)]
    pfam_df['index'] = pfam_df.family_id.apply(lambda x: family_id_to_index[x])

    pfam_df['one_hot_inds'] = pfam_df.sequence.apply(
        lambda x: residues_to_one_hot_inds(x[:512]))

    if samples is not None:
        pfam_df = pfam_df.sample(frac=1,
                                 replace=False,
                                 random_state=random_state)
        pfam_df = pfam_df.groupby('mod_family_accession').head(
            samples).reset_index()

    return pfam_df


def create_pfam_seq_batches(family_accessions,
                            batch_size,
                            test=False,
                            samples=None,
                            epochs=1,
                            drop_remainder=False,
                            buffer_size=None,
                            shuffle_seed=0,
                            sample_random_state=0,
                            data_partitions_dirpath='random_split/',
                            gcs_bucket='neuralblast_public',
                            as_numpy=False):
    """Creates iterable object of Pfam sequences."""

    pfam_df = create_pfam_df(family_accessions,
                             test=test,
                             samples=samples,
                             random_state=sample_random_state,
                             data_partitions_dirpath=data_partitions_dirpath,
                             gcs_bucket=gcs_bucket)

    pfam_batches = create_data_iterator(df=pfam_df,
                                        input_col='one_hot_inds',
                                        output_col='index',
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        buffer_size=buffer_size,
                                        seed=shuffle_seed,
                                        drop_remainder=drop_remainder,
                                        add_outputs=False,
                                        as_numpy=as_numpy)

    return pfam_batches


def create_pfam_batches(family_accessions,
                        batch_size,
                        test=False,
                        samples=None,
                        epochs=1,
                        drop_remainder=False,
                        buffer_size=None,
                        shuffle_seed=0,
                        sample_random_state=0,
                        data_partitions_dirpath='random_split/',
                        gcs_bucket='neuralblast_public',
                        as_numpy=True):
    """Creates iterable object of Pfam data batches."""

    pfam_df = create_pfam_df(family_accessions,
                             test=test,
                             samples=samples,
                             random_state=sample_random_state,
                             data_partitions_dirpath=data_partitions_dirpath,
                             gcs_bucket=gcs_bucket)

    pfam_indexes = pfam_df['index'].values

    pfam_batches = create_data_iterator(df=pfam_df,
                                        input_col='one_hot_inds',
                                        output_col='index',
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        buffer_size=buffer_size,
                                        seed=shuffle_seed,
                                        drop_remainder=drop_remainder,
                                        as_numpy=as_numpy)

    return pfam_batches, pfam_indexes


# Model evaluation.
def pfam_evaluate(predict_fn,
                  test_family_accessions,
                  title,
                  loss_fn_kwargs,
                  batch_size=512,
                  data_partitions_dirpath='random_split/',
                  gcs_bucket='neuralblast_public'):
    """Computes predicted family ids and measures performance in cross entropy and accuracy."""

    test_batches, test_indexes = create_pfam_batches(
        family_accessions=test_family_accessions,
        batch_size=batch_size,
        test=True,
        buffer_size=1,
        gcs_bucket=gcs_bucket,
        data_partitions_dirpath=data_partitions_dirpath)

    pred_indexes = []
    cross_entropy = 0.

    for batch in iter(test_batches):

        X, Y = batch

        Y_hat = predict_fn(X)

        cross_entropy += cross_entropy_loss(Y, Y_hat, **loss_fn_kwargs)

        preds = jnp.argmax(Y_hat, axis=1)
        for pred in preds:
            pred_indexes.append(pred)

    pred_indexes = np.array(pred_indexes)

    acc = metrics.accuracy_score(test_indexes, pred_indexes)

    results = {
        'title': title,
        'cross_entropy': cross_entropy,
        'accuracy': acc,
    }

    return results, pred_indexes


def compute_embeddings(encoder, data_batches):
    """Computes sequence embeddings according to a specified encoder."""

    vectors = []
    for batch in iter(data_batches):
        X, Y = batch
        X_embedded = encoder(X)
        for vec in np.array(X_embedded):
            vectors.append(vec)
    vectors = np.array(vectors)

    return vectors


def pfam_nearest_neighbors_classification(
        encoder,
        family_accessions,
        batch_size=512,
        n_neighbors=1,
        train_samples=None,
        test_samples=None,
        shuffle_seed=0,
        sample_random_state=0,
        data_partitions_dirpath='random_split/',
        gcs_bucket='neuralblast_public'):
    """Nearest neighbors classification on Pfam families using specified encoder."""

    train_batches, train_indexes = create_pfam_batches(
        family_accessions=family_accessions,
        batch_size=batch_size,
        samples=train_samples,
        buffer_size=1,
        shuffle_seed=shuffle_seed,
        sample_random_state=sample_random_state,
        data_partitions_dirpath=data_partitions_dirpath,
        gcs_bucket=gcs_bucket)
    test_batches, test_indexes = create_pfam_batches(
        family_accessions=family_accessions,
        batch_size=batch_size,
        test=True,
        samples=test_samples,
        buffer_size=1,
        shuffle_seed=shuffle_seed,
        sample_random_state=sample_random_state,
        data_partitions_dirpath=data_partitions_dirpath,
        gcs_bucket=gcs_bucket)

    train_vectors = compute_embeddings(encoder, train_batches)
    test_vectors = compute_embeddings(encoder, test_batches)

    knn_classifier = knn(n_neighbors=n_neighbors)
    knn_classifier.fit(train_vectors, train_indexes)
    knn_predictions = knn_classifier.predict(test_vectors)

    knn_accuracy = metrics.accuracy_score(test_indexes, knn_predictions)

    results = {
        str(n_neighbors) + "-nn accuracy": knn_accuracy,
        'train_samples': train_samples,
        'test_samples': test_samples
    }

    return results, knn_predictions, knn_classifier
