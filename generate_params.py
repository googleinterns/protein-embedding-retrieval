"""Generate JSON files with hyperparameter combos for caliban."""

import json

import itertools

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
                  save_dir='pfam_experiment_data',
                  lens_shuffle_seed=0,
                  lens_sample_random_state=0,
                  knn_shuffle_seed=1,
                  knn_sample_random_state=1):
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
            'random_key': random_key,
            'use_transformer': use_transformer,
            'use_bert': use_bert,
            'restore_transformer_dir': restore_transformer_dir,
            'lens_shuffle_seed': lens_shuffle_seed,
            'lens_sample_random_state': lens_sample_random_state,
            'knn_shuffle_seed': knn_shuffle_seed,
            'knn_sample_random_state': knn_sample_random_state,
            'gcs_bucket': gcs_bucket,
            'data_partitions_dirpath': data_partitions_dirpath,
            'save_dir': save_dir
        }
        params.append(param_dict)

    return params


# Generate parameters from different sets of parameter combinations.
def main():

    params = []

    # 1000 train families
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

    # NEW
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


    frozen_param_dict_to_index = {}
    index = 0
    for param_dict in params:
        if frozendict(param_dict) not in frozen_param_dict_to_index.keys():
            frozen_param_dict_to_index[frozendict(param_dict)] = index
            index += 1

    unique_params = []
    for frozen_param_dict in frozen_param_dict_to_index.keys():
        param_dict = dict(frozen_param_dict)
        param_dict.update(
            {'index': frozen_param_dict_to_index[frozen_param_dict]})
        unique_params.append(param_dict)
    unique_params = sorted(unique_params, key=lambda x: x['index'])

    def transform_index(param_dict):
        param_dict['index'] = '%08d' % param_dict['index']
        return param_dict

    unique_params = [
        transform_index(param_dict) for param_dict in unique_params
    ]

    with open('params_combinations.json', 'w') as f:
        json.dump(unique_params, f)

    index_to_params = {}
    for param_dict in unique_params:
        index_to_params[param_dict['index']] = param_dict

    with open('index_to_params.json', 'w') as f:
        json.dump(index_to_params, f)


if __name__ == '__main__':
    main()
