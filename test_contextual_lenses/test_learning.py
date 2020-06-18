"""End-to-end learning tests for contextual_lenses.py."""


import jax.numpy as jnp

import numpy as np

from absl.testing import parameterized
from absl.testing import absltest

from contextual_lenses import mean_pool, max_pool, \
linear_max_pool, linear_mean_pool, gated_conv 

from train_utils import create_optimizer, train_step, \
create_representation_model

from encoders import one_hot_encoder, cnn_one_hot_encoder, \
one_hot_pos_emb_encoder, cnn_one_hot_pos_emb_encoder


def generate_random_sequences(bs=3, seq_len=12, num_categories=21):
  """Generates bs many random sequences of length seq_len from num_categories indices."""

  np.random.seed(0)
  input_data = jnp.array(np.random.randint(0, num_categories-1, size=(bs, seq_len)))
  
  np.random.seed(0)
  output_data = jnp.array(np.random.normal(size=(bs)))

  return input_data, output_data


def compute_train_loss(model, input_data, output_data, learning_rate=1e-3, epochs=100):
  """Fits model to training data and returns train loss."""
  # print(input_data)
  optimizer = create_optimizer(model, learning_rate=learning_rate, weight_decay=0.)

  for epoch in range(epochs):
    optimizer = train_step(optimizer, input_data, output_data)

  preds = jnp.squeeze(optimizer.target(input_data), axis=1)

  train_loss = jnp.mean((preds-output_data)**2)

  return train_loss


"""Test cases."""

"""One-hot."""

"""Test-case bugged due to optimization issues."""
# test1 = {
#           'testcase_name': 'max_pool',
#           'encoder_fn': one_hot_encoder,
#           'encoder_fn_kwargs': {
# 
#           },
#           'reduce_fn': max_pool,
#           'reduce_fn_kwargs': {
# 
#           },
#           'learning_rate': 1e-2,
#           'epochs': 100000,
#           'loss_threshold': 1e-4
#       }

"""Test-case bugged due to optimization issues."""
# test2 = {
#           'testcase_name': 'mean_pool',
#           'encoder_fn': one_hot_encoder,
#           'encoder_fn_kwargs': {
# 
#           },
#           'reduce_fn': mean_pool,
#           'reduce_fn_kwargs': {
# 
#           },
#           'learning_rate': 1e-4,
#           'epochs': 100000,
#           'loss_threshold': 1e-4
#       }

test3 = {
          'testcase_name': 'linear_max_pool',
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs': {

          },
          'reduce_fn': linear_max_pool,
          'reduce_fn_kwargs': {
              'rep_size': 512
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }

test4 = {
          'testcase_name': 'linear_mean_pool',
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs': {

          },
          'reduce_fn': linear_mean_pool,
          'reduce_fn_kwargs': {
              'rep_size': 2048
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test5 = {
          'testcase_name': 'gated_conv',
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs': {

          },
          'reduce_fn': gated_conv,
          'reduce_fn_kwargs': {
              'rep_size': 256,
              'M_layers': 3,
              'M_features': [[512, 512], [512, 512]],
              'M_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
              'zero_rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


"""CNN + one-hot."""
test6 = {
          'testcase_name': 'cnn_max_pool',
          'encoder_fn': cnn_one_hot_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [256],
              'N_kernel_sizes': [3]
          },
          'reduce_fn': max_pool,
          'reduce_fn_kwargs': {

          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test7 = {
          'testcase_name': 'cnn_mean_pool',
          'encoder_fn': cnn_one_hot_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [512],
              'N_kernel_sizes': [5]
          },
          'reduce_fn': mean_pool,
          'reduce_fn_kwargs': {

          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test8 = {
          'testcase_name': 'cnn_linear_max_pool',
          'encoder_fn': cnn_one_hot_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [32],
              'N_kernel_sizes': [3]
          },
          'reduce_fn': linear_max_pool,
          'reduce_fn_kwargs': {
              'rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test9 = {
          'testcase_name': 'cnn_linear_mean_pool',
          'encoder_fn': cnn_one_hot_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [512],
              'N_kernel_sizes': [5]
          },
          'reduce_fn': linear_mean_pool,
          'reduce_fn_kwargs': {
              'rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test10 = {
          'testcase_name': 'cnn_gated_conv',
          'encoder_fn': cnn_one_hot_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [32],
              'N_kernel_sizes': [3]
          },
          'reduce_fn': gated_conv,
          'reduce_fn_kwargs': {
              'rep_size': 256,
              'M_layers': 3,
              'M_features': [[512, 512], [512, 512]],
              'M_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
              'zero_rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


"""One-hot + positional embeddings."""

"""Test-case bugged due to optimization issues."""
# test11 = {
#           'testcase_name': 'pos_emb_max_pool',
#           'encoder_fn': one_hot_pos_emb_encoder,
#           'encoder_fn_kwargs': {
#               'max_len': 512
#           },
#           'reduce_fn': max_pool,
#           'reduce_fn_kwargs': {
# 
#           },
#           'learning_rate': 1e-2,
#           'epochs': 100000,
#           'loss_threshold': 1e-4
#       }

"""Test-case bugged due to optimization issues."""
# test12 = {
#           'testcase_name': 'pos_emb_mean_pool',
#           'encoder_fn': one_hot_pos_emb_encoder,
#           'encoder_fn_kwargs': {
#               'max_len': 512
#           },
#           'reduce_fn': mean_pool,
#           'reduce_fn_kwargs': {
# 
#           },
#           'learning_rate': 1e-4,
#           'epochs': 100000,
#           'loss_threshold': 1e-4
#       }

test13 = {
          'testcase_name': 'pos_emb_linear_max_pool',
          'encoder_fn': one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'max_len': 512
          },
          'reduce_fn': linear_max_pool,
          'reduce_fn_kwargs': {
              'rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }

test14 = {
          'testcase_name': 'pos_emb_linear_mean_pool',
          'encoder_fn': one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'max_len': 512
          },
          'reduce_fn': linear_mean_pool,
          'reduce_fn_kwargs': {
              'rep_size': 4096
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test15 = {
          'testcase_name': 'pos_emb_gated_conv',
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs': {
              'max_len': 512
          },
          'reduce_fn': gated_conv,
          'reduce_fn_kwargs': {
              'rep_size': 256,
              'M_layers': 3,
              'M_features': [[512, 512], [512, 512]],
              'M_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
              'zero_rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


"""CNN + one-hot + positional embeddings."""
test16 = {
          'testcase_name': 'cnn_pos_emb_max_pool',
          'encoder_fn': cnn_one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [256],
              'N_kernel_sizes': [3],
              'max_len': 512
          },
          'reduce_fn': max_pool,
          'reduce_fn_kwargs': {

          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test17 = {
          'testcase_name': 'cnn_pos_emb_mean_pool',
          'encoder_fn': cnn_one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [512],
              'N_kernel_sizes': [5],
              'max_len': 512
          },
          'reduce_fn': mean_pool,
          'reduce_fn_kwargs': {

          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test18 = {
          'testcase_name': 'cnn_pos_emb_linear_max_pool',
          'encoder_fn': cnn_one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [32],
              'N_kernel_sizes': [3],
              'max_len': 512
          },
          'reduce_fn': linear_max_pool,
          'reduce_fn_kwargs': {
              'rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test19 = {
          'testcase_name': 'cnn_pos_emb_linear_mean_pool',
          'encoder_fn': cnn_one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [512],
              'N_kernel_sizes': [5],
              'max_len': 512
          },
          'reduce_fn': linear_mean_pool,
          'reduce_fn_kwargs': {
              'rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


test20 = {
          'testcase_name': 'cnn_pos_emb_gated_conv',
          'encoder_fn': cnn_one_hot_pos_emb_encoder,
          'encoder_fn_kwargs': {
              'N_layers': 1,
              'N_features': [32],
              'N_kernel_sizes': [3],
              'max_len': 512
          },
          'reduce_fn': gated_conv,
          'reduce_fn_kwargs': {
              'rep_size': 256,
              'M_layers': 3,
              'M_features': [[512, 512], [512, 512]],
              'M_kernel_sizes': [[12, 12], [10, 10], [8, 8]],
              'zero_rep_size': 256
          },
          'learning_rate': 1e-3,
          'epochs': 100,
          'loss_threshold': 1e-4
      }


tests = (test3,
         test4,
         test5,
         test6,
         test7,
         test8,
         test9,
         test10,
         test13,
         test14,
         test15,
         test16,
         test17,
         test18,
         test19,
         test20)


class TestLearning(parameterized.TestCase):
  @parameterized.named_parameters(
      *tests
  )
  def test_learning(self, encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs,
                    learning_rate=1e-3, epochs=1000, loss_threshold=1e-4):
    
    input_data, output_data = generate_random_sequences(bs=3, seq_len=12, num_categories=21)

    model = create_representation_model(encoder_fn=encoder_fn,
                                        encoder_fn_kwargs=encoder_fn_kwargs,
                                        reduce_fn=reduce_fn,
                                        reduce_fn_kwargs=reduce_fn_kwargs,
                                        num_categories=21)
  
    train_loss = compute_train_loss(model, input_data, output_data)

    self.assertTrue(train_loss < loss_threshold)


if __name__ == '__main__':
  absltest.main()
