"""Generate JSON files with hyperparameter combos for caliban."""

import json

import itertools

import os

import numpy as np

from frozendict import frozendict


def create_params(encoder_lrs,
                  lens_lrs,
                  predictor_lrs,
                  encoder_wds,
                  lens_wds,
                  predictor_wds,
                  reduce_fn_kwargs_paths,
                  lens_train_samples,
                  train_families,
                  epochs,
                  measurements,
                  encoder_fn_name,
                  encoder_fn_kwargs_path,
                  reduce_fn_name,
                  lens_batch_size=64,
                  knn_batch_size=64,
                  use_transformer=False,
                  use_bert=False,
                  restore_transformer_dir=None,
                  random_keys=[0],
                  first_test_family=15000,
                  last_test_family=16000,
                  gcs_bucket='sequin-public',
                  data_partitions_dirpath='random_split/',
                  results_save_dir='pfam_experiment_data',
                  lens_shuffle_seed=0,
                  lens_sample_random_state=0,
                  knn_shuffle_seed=1,
                  knn_sample_random_state=1,
                  load_model=False,
                  load_model_dir='',
                  load_model_step=0,
                  save_model=False,
                  save_model_dir=''):
    """Generates parameters from lists of parameters."""

    params = []

    for encoder_lr, lens_lr, predictor_lr, encoder_wd, lens_wd, predictor_wd, reduce_fn_kwargs_path, lens_train_samples, random_key in \
     itertools.product(encoder_lrs, lens_lrs, predictor_lrs, encoder_wds, lens_wds, predictor_wds, reduce_fn_kwargs_paths, lens_train_samples, random_keys):

        param_dict = {
            'encoder_fn_name': encoder_fn_name,
            'encoder_fn_kwargs_path': encoder_fn_kwargs_path,
            'reduce_fn_name': reduce_fn_name,
            'reduce_fn_kwargs_path': reduce_fn_kwargs_path,
            'epochs': epochs,
            'measurements': measurements,
            'lens_batch_size': lens_batch_size,
            'knn_batch_size': knn_batch_size,
            'encoder_lr': encoder_lr,
            'lens_lr': lens_lr,
            'predictor_lr': predictor_lr,
            'encoder_wd': encoder_wd,
            'lens_wd': lens_wd,
            'predictor_wd': predictor_wd,
            'train_families': train_families,
            'lens_train_samples': lens_train_samples,
            'first_test_family': first_test_family,
            'last_test_family': last_test_family,
            'lens_shuffle_seed': lens_shuffle_seed,
            'lens_sample_random_state': lens_sample_random_state,
            'knn_shuffle_seed': knn_shuffle_seed,
            'knn_sample_random_state': knn_sample_random_state,
            'random_key': random_key,
            'use_transformer': use_transformer,
            'use_bert': use_bert,
            'gcs_bucket': gcs_bucket,
            'data_partitions_dirpath': data_partitions_dirpath,
            'results_save_dir': results_save_dir,
            'load_model': load_model,
            'load_model_dir': load_model_dir,
            'load_model_step': load_model_step,
            'save_model': save_model,
            'save_model_dir': save_model_dir
        }

        if restore_transformer_dir is not None:
            param_dict['restore_transformer_dir'] = restore_transformer_dir

        params.append(param_dict)

    return params


# Generate parameters from different sets of parameter combinations.
def main():

    params = []

    '''
    # 1000 train families

    # Medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        predictor_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        encoder_wds=[0.0],
        lens_wds=[0.0, 0.1],
        predictor_wds=[0.0, 0.1],
        reduce_fn_kwargs_paths=['linear_pool_256', 'linear_pool_1024'],
        lens_train_samples=[50],
        train_families=1000,
        epochs=50,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=None,
        gcs_bucket='sequin-public')

    # Pretrained medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        predictor_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        encoder_wds=[0.0],
        lens_wds=[0.0, 0.1],
        predictor_wds=[0.0, 0.1],
        reduce_fn_kwargs_paths=['linear_pool_256', 'linear_pool_1024'],
        lens_train_samples=[50],
        train_families=1000,
        epochs=50,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/medium_trembl_bert/',
        gcs_bucket='sequin-public')

    # 1-layer CNN
    params += create_params(
        encoder_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        lens_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        predictor_lrs=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        encoder_wds=[0.0, 0.1],
        lens_wds=[0.0, 0.1],
        predictor_wds=[0.0, 0.1],
        reduce_fn_kwargs_paths=['linear_pool_256', 'linear_pool_1024'],
        lens_train_samples=[50],
        train_families=1000,
        epochs=50,
        measurements=1,
        encoder_fn_name='cnn_one_hot',
        encoder_fn_kwargs_path='1-layer_cnn_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=256,
        knn_batch_size=256,
        use_transformer=False,
        use_bert=False,
        restore_transformer_dir=None,
        gcs_bucket='sequin-public')


    # 10000 train families

    # Medium transformer
    params += create_params(encoder_lrs=[0.0],
                            lens_lrs=[1e-3, 5e-4, 1e-4, 5e-5],
                            predictor_lrs=[1e-3, 5e-4, 1e-4, 5e-5],
                            encoder_wds=[0.0],
                            lens_wds=[0.0, 0.05, 0.1, 0.2],
                            predictor_wds=[0.0, 0.05, 0.1, 0.2],
                            reduce_fn_kwargs_paths=['linear_pool_1024'],
                            lens_train_samples=[50],
                            train_families=10000,
                            epochs=10,
                            measurements=2,
                            encoder_fn_name='transformer',
                            encoder_fn_kwargs_path='medium_transformer_kwargs',
                            reduce_fn_name='linear_max_pool',
                            lens_batch_size=64,
                            knn_batch_size=64,
                            use_transformer=True,
                            use_bert=True,
                            restore_transformer_dir=None,
                            gcs_bucket='sequin-public')

    # Pretrained medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3, 5e-4, 1e-4, 5e-5],
        predictor_lrs=[1e-3, 5e-4, 1e-4, 5e-5],
        encoder_wds=[0.0],
        lens_wds=[0.0, 0.05, 0.1, 0.2],
        predictor_wds=[0.0, 0.05, 0.1, 0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/medium_trembl_bert/',
        gcs_bucket='sequin-public')

    # 2-layer CNN
    params += create_params(encoder_lrs=[1e-3, 1e-4, 1e-5],
                            lens_lrs=[1e-3, 1e-4, 1e-5],
                            predictor_lrs=[1e-3, 1e-4, 1e-5],
                            encoder_wds=[0.0, 0.1, 0.2],
                            lens_wds=[0.0, 0.1, 0.2],
                            predictor_wds=[0.0, 0.1, 0.2],
                            reduce_fn_kwargs_paths=['linear_pool_1024'],
                            lens_train_samples=[50],
                            train_families=10000,
                            epochs=10,
                            measurements=2,
                            encoder_fn_name='cnn_one_hot',
                            encoder_fn_kwargs_path='2-layer_cnn_kwargs',
                            reduce_fn_name='linear_max_pool',
                            lens_batch_size=512,
                            knn_batch_size=512,
                            use_transformer=False,
                            use_bert=False,
                            restore_transformer_dir=None,
                            gcs_bucket='sequin-public')

    # Medium transformer
    params += create_params(encoder_lrs=[0.0],
                            lens_lrs=[1e-4, 5e-5, 1e-5],
                            predictor_lrs=[1e-3, 5e-4, 1e-4],
                            encoder_wds=[0.0],
                            lens_wds=[0.05, 0.1, 0.2],
                            predictor_wds=[0.0, 0.05, 0.1],
                            reduce_fn_kwargs_paths=['linear_pool_1024'],
                            lens_train_samples=[50],
                            train_families=10000,
                            epochs=10,
                            measurements=2,
                            encoder_fn_name='transformer',
                            encoder_fn_kwargs_path='medium_transformer_kwargs',
                            reduce_fn_name='linear_max_pool',
                            lens_batch_size=64,
                            knn_batch_size=64,
                            use_transformer=True,
                            use_bert=True,
                            restore_transformer_dir=None,
                            gcs_bucket='sequin-public')

    # Pretrained medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4, 5e-5, 1e-5],
        predictor_lrs=[1e-3, 5e-4, 1e-4],
        encoder_wds=[0.0],
        lens_wds=[0.05, 0.1, 0.2],
        predictor_wds=[0.15, 0.2, 0.25],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/medium_trembl_bert/',
        gcs_bucket='sequin-public')

    # 2-layer CNN
    params += create_params(encoder_lrs=[1e-3, 5e-4, 1e-4],
                            lens_lrs=[1e-4, 5e-5, 1e-5],
                            predictor_lrs=[1e-3, 5e-4, 1e-4],
                            encoder_wds=[0.05, 0.1, 0.2],
                            lens_wds=[0.05, 0.1, 0.2],
                            predictor_wds=[0.0, 0.05, 0.1],
                            reduce_fn_kwargs_paths=['linear_pool_1024'],
                            lens_train_samples=[50],
                            train_families=10000,
                            epochs=10,
                            measurements=2,
                            encoder_fn_name='cnn_one_hot',
                            encoder_fn_kwargs_path='2-layer_cnn_kwargs',
                            reduce_fn_name='linear_max_pool',
                            lens_batch_size=512,
                            knn_batch_size=512,
                            use_transformer=False,
                            use_bert=False,
                            restore_transformer_dir=None,
                            gcs_bucket='sequin-public')

    # 2-layer CNN
    params += create_params(encoder_lrs=[1e-3, 5e-4, 1e-4, 5e-5],
                            lens_lrs=[5e-5, 1e-5, 5e-6],
                            predictor_lrs=[1e-4, 5e-5, 1e-5, 5e-6],
                            encoder_wds=[0.2, 0.3],
                            lens_wds=[0.05, 0.1],
                            predictor_wds=[0.0, 0.05, 0.1],
                            reduce_fn_kwargs_paths=['linear_pool_1024'],
                            lens_train_samples=[50],
                            train_families=10000,
                            epochs=10,
                            measurements=2,
                            encoder_fn_name='cnn_one_hot',
                            encoder_fn_kwargs_path='2-layer_cnn_kwargs',
                            reduce_fn_name='linear_max_pool',
                            lens_batch_size=512,
                            knn_batch_size=512,
                            use_transformer=False,
                            use_bert=False,
                            restore_transformer_dir=None,
                            gcs_bucket='sequin-public')
    '''

    '''
    # Medium transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-5],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.05],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-5],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.05],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/medium_trembl_bert/',
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[5e-5],
        predictor_lrs=[5e-4],
        encoder_wds=[0.0],
        lens_wds=[0.2],
        predictor_wds=[0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained medium transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[5e-5],
        predictor_lrs=[5e-4],
        encoder_wds=[0.0],
        lens_wds=[0.2],
        predictor_wds=[0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=2,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='medium_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/medium_trembl_bert/',
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Small transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3, 1e-4, 1e-5, 1e-6],
        predictor_lrs=[1e-3, 1e-4, 1e-5, 1e-6],
        encoder_wds=[0.0],
        lens_wds=[0.0, 0.1, 0.2],
        predictor_wds=[0.0, 0.1, 0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public')

    # Pretrained small transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3, 1e-4, 1e-5, 1e-6],
        predictor_lrs=[1e-3, 1e-4, 1e-5, 1e-6],
        encoder_wds=[0.0],
        lens_wds=[0.0, 0.1, 0.2],
        predictor_wds=[0.0, 0.1, 0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/small_trembl_bert/',
        gcs_bucket='sequin-public')

    # Small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.1],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.1],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/small_trembl_bert/',
        random_keys=range(10))

    # Small transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.1],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-3],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.1],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/small_trembl_bert/',
        random_keys=range(10))

    # Small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.0],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.0],
        predictor_wds=[0.0],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/small_trembl_bert/',
        random_keys=range(10))

    # Small transformer
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.2],
        predictor_wds=[0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        random_keys=range(10))

    # Pretrained small transformer 
    params += create_params(
        encoder_lrs=[0.0],
        lens_lrs=[1e-4],
        predictor_lrs=[1e-3],
        encoder_wds=[0.0],
        lens_wds=[0.2],
        predictor_wds=[0.2],
        reduce_fn_kwargs_paths=['linear_pool_1024'],
        lens_train_samples=[50],
        train_families=10000,
        epochs=10,
        measurements=1,
        encoder_fn_name='transformer',
        encoder_fn_kwargs_path='small_transformer_kwargs',
        reduce_fn_name='linear_max_pool',
        lens_batch_size=64,
        knn_batch_size=64,
        use_transformer=True,
        use_bert=True,
        gcs_bucket='sequin-public',
        restore_transformer_dir=
        'gs://sequin-public/transformer_models/small_trembl_bert/',
        random_keys=range(10))
    '''

    for i in range(5):
        # Large transformer 
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-5],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.05],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_0_' + str(i)))

        # Pretrained large transformer
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-5],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.05],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            restore_transformer_dir=
            'gs://sequin-public/transformer_models/large_bert/',
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_1_' + str(i)))

        # Large transformer
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[5e-5],
            predictor_lrs=[5e-4],
            encoder_wds=[0.0],
            lens_wds=[0.2],
            predictor_wds=[0.2],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_2_' + str(i)))

        # Pretrained large transformer
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[5e-5],
            predictor_lrs=[5e-4],
            encoder_wds=[0.0],
            lens_wds=[0.2],
            predictor_wds=[0.2],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            restore_transformer_dir=
            'gs://sequin-public/transformer_models/large_bert/',
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_3_' + str(i)))

        # Large transformer 
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-4],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.1],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_4_' + str(i)))

        # Pretrained large transformer 
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-4],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.1],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            restore_transformer_dir=
            'gs://sequin-public/transformer_models/large_bert/',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_5_' + str(i)))

        # Large transformer
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-3],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.1],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_6_' + str(i)))

        # Pretrained large transformer 
        params += create_params(
            encoder_lrs=[0.0],
            lens_lrs=[1e-3],
            predictor_lrs=[1e-3],
            encoder_wds=[0.0],
            lens_wds=[0.1],
            predictor_wds=[0.0],
            reduce_fn_kwargs_paths=['linear_pool_1024'],
            lens_train_samples=[50],
            train_families=10000,
            epochs=10,
            measurements=1,
            encoder_fn_name='transformer',
            encoder_fn_kwargs_path='large_transformer_kwargs',
            reduce_fn_name='linear_max_pool',
            lens_batch_size=8,
            knn_batch_size=64,
            use_transformer=True,
            use_bert=True,
            gcs_bucket='sequin-public',
            restore_transformer_dir=
            'gs://sequin-public/transformer_models/large_bert/',
            random_keys=[i],
            save_model=True,
            save_model_dir=os.path.join('pfam_experiment_optimizers', 'large_7_' + str(i)))


    frozen_param_dict_to_label = {}
    label = 0 + 408
    for param_dict in params:
        if frozendict(param_dict) not in frozen_param_dict_to_label.keys():
            frozen_param_dict_to_label[frozendict(param_dict)] = label
            label += 1

    unique_params = []
    for frozen_param_dict in frozen_param_dict_to_label.keys():
        param_dict = dict(frozen_param_dict)
        param_dict.update(
            {'label': frozen_param_dict_to_label[frozen_param_dict]})
        unique_params.append(param_dict)
    unique_params = sorted(unique_params, key=lambda x: x['label'])

    def transform_label(param_dict):
        param_dict['label'] = '%08d' % param_dict['label']
        return param_dict

    unique_params = [
        transform_label(param_dict) for param_dict in unique_params
    ]

    with open('params_combinations.json', 'w') as f:
        json.dump(unique_params, f)

    label_to_params = {}
    for param_dict in unique_params:
        label_to_params[param_dict['label']] = param_dict

    with open('label_to_params.json', 'w') as f:
        json.dump(label_to_params, f)


if __name__ == '__main__':
    main()
