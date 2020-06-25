"""Tests for contextual_lenses.py."""


import jax
import jax.numpy as jnp

import numpy as np

from absl.testing import parameterized
from absl.testing import absltest

from contextual_lenses import mean_pool, max_pool


# Test cases:
test1 = {
          'x': jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[0, -1, 2, -3], [4, -5, 6, -7], [8, -9, 10, -11]]]),
          'padding_mask' : None,
          'mean_pool_rep': jnp.array([[4, 5, 6, 7],
                                      [4, -5, 6, -7]]),
          'max_pool_rep': jnp.array([[8, 9, 10, 11],
                                     [8, -1, 10, -3]])
      }

test2 = {
          'x': jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[0, -1, 2, -3], [4, -5, 6, -7], [8, -9, 10, -11]]]),
          'padding_mask' : jnp.array([[[1], [1], [1]]
                                      [1], [1], [1]]),
          'mean_pool_rep': jnp.array([[4, 5, 6, 7],
                                      [4, -5, 6, -7]]),
          'max_pool_rep': jnp.array([[8, 9, 10, 11],
                                     [8, -1, 10, -3]])
      }

test3 = {
          'x': jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[0, -1, 2, -3], [4, -5, 6, -7], [8, -9, 10, -11]]]),
          'padding_mask' : jnp.array([[[1], [1], [1]], 
                                      [[1], [1], [0]]]),
          'mean_pool_rep': jnp.array([[4, 5, 6, 7],
                                      [2, -3, 4, -5]]),
          'max_pool_rep': jnp.array([[8, 9, 10, 11],
                                     [4, -1, 6, -3]])
      }

test4 = {
          'x': jnp.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                          [[0, -1, 2, -3], [4, -5, 6, -7], [8, -9, 10, -11]]]),
          'padding_mask' : jnp.array([[[1], [0], [1]], 
                                      [[0], [1], [0]]]),
          'mean_pool_rep': jnp.array([[4, 5, 6, 7],
                                      [4, -5, 6, -7]]),
          'max_pool_rep': jnp.array([[8, 9, 10, 11],
                                     [4, -5, 6, -7]])
      }

test5 = {
          'x': jnp.array([[[5, 2, -5], [1, -3, 4], [-3, -8, 1]],
                          [[-2, -7, 4], [9, -4, 5], [-1, 2, -3]]]),
          'padding_mask' : None,
          'mean_pool_rep': jnp.array([[1, -3, 0],
                                      [2, -3, 2]]),
          'max_pool_rep': jnp.array([[5, 2, 4],
                                     [9, 2, 5]])
      }

test6 = {
          'x': jnp.array([[[5, 2, -5], [1, -3, 4], [-3, -8, 1]],
                          [[-2, -7, 4], [9, -4, 5], [-1, 2, -3]]]),
          'padding_mask' : jnp.array([[[1], [1], [1]], 
                                      [[1], [1], [1]]]),
          'mean_pool_rep': jnp.array([[1, -3, 0],
                                      [2, -3, 2]]),
          'max_pool_rep':  jnp.array([[5, 2, 4],
                                     [9, 2, 5]])
      }

test7 = {
          'x': jnp.array([[[6, 2, -5], [1, -2, 4], [-3, -8, 0]],
                          [[-2, -7, 4], [10, -4, 5], [-1, 3, -3]]]),
          'padding_mask' : jnp.array([[[0], [1], [1]], 
                                      [[1], [0], [0]]]),
          'mean_pool_rep': jnp.array([[-1, -5, 2],
                                      [-2, -7, 4]]),
          'max_pool_rep': jnp.array([[1, -2, 4],
                                     [-2, -7, 4]])
      }

test8 = {
          'x': jnp.array([[[6, 2, -5], [1, -2, 4], [-3, -8, 0]],
                          [[-2, -7, 4], [10, -4, 5], [-1, 3, -3]]]),
          'padding_mask' : jnp.array([[[1], [0], [1]], 
                                      [[1], [1], [0]]]),
          'mean_pool_rep': jnp.array([[3/2, -3, -5/2],
                                      [4, -11/2, 9/2]]),
          'max_pool_rep': jnp.array([[6, 2, 0],
                                     [10, -4, 5]])
      }

tests = (test1,
         test2,
         test3,
         test4,
         test5,
         test6,
         test7,
         test8)


class TestPooling(parameterized.TestCase):
  """Abstract method for testing mean and max pool reduce functions."""
  @parameterized.parameters(
      *tests
  )
  def test_mean_pool(self, x, padding_mask, mean_pool_rep, **unused_kwargs):
    self.assertTrue(jnp.array_equal(mean_pool(x, padding_mask), mean_pool_rep))

  @parameterized.parameters(
      *tests
  )
  def test_max_pool(self, x, padding_mask, max_pool_rep, **unused_kwargs):
    self.assertTrue(jnp.array_equal(max_pool(x, padding_mask), max_pool_rep))
  
  @parameterized.parameters(
      *tests
  )
  def test_max_pool_greater_than_mean_pool(self, x, padding_mask, **unused_kwargs):
    self.assertTrue((max_pool(x, padding_mask=None) >= mean_pool(x, padding_mask=None)).all())


if __name__ == '__main__':
  absltest.main()
