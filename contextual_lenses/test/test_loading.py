"""Tests for saving and restoring checkpoints."""


import os

from absl.testing import parameterized
from absl.testing import absltest

from flax.training import checkpoints

from contextual_lenses import mean_pool, max_pool

from train_utils import create_optimizer, create_representation_model

from encoders import one_hot_encoder


# Test cases:
test1 = {
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs' : {

          },
          'reduce_fn': mean_pool,
          'reduce_fn_kwargs': {

          }
      }

test2 = {
          'encoder_fn': one_hot_encoder,
          'encoder_fn_kwargs' : {

          },
          'reduce_fn': max_pool,
          'reduce_fn_kwargs': {

          }
      }

tests = (test1, test2)


class TestLoading(parameterized.TestCase):
  """Abstract method for testing saving and restoring optimizer checkpoints."""

  @parameterized.parameters(
      *tests
  )
  def test_loading(self, encoder_fn, encoder_fn_kwargs, reduce_fn, reduce_fn_kwargs):

  	model = create_representation_model(encoder_fn=encoder_fn,
  										encoder_fn_kwargs=encoder_fn_kwargs,
  										reduce_fn=reduce_fn,
  										reduce_fn_kwargs=reduce_fn_kwargs,
  										num_categories=21,
  										output_features=1)

  	optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.)

  	params = optimizer.target.params
  	step = optimizer.state.step

  	os.mkdir('tmp/')

  	checkpoints.save_checkpoint(ckpt_dir='tmp/', target=optimizer, step=optimizer.state.step)

  	loaded_optimizer = checkpoints.restore_checkpoint(ckpt_dir='tmp/', target=optimizer)

  	loaded_params = loaded_optimizer.target.params
  	loaded_step = loaded_optimizer.state.step

  	os.system('rm -rf tmp/')

  	self.assertTrue(step==loaded_step)

  	self.assertTrue(list(params.keys())==list(loaded_params.keys()))
  	for key in params.keys():
  		sub_params = params[key]
  		loaded_sub_params = loaded_params[key]
  		self.assertTrue(list(sub_params.keys())==list(loaded_sub_params.keys()))
  		for sub_key in sub_params.keys():
  			self.assertTrue((sub_params[sub_key]==loaded_sub_params[sub_key]).all())


if __name__ == '__main__':
  absltest.main()

