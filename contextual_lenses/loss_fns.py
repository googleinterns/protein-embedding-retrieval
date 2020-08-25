"""Loss functions

Jax loss functions for computing gradient updates.
"""

import jax
import jax.numpy as jnp



def mse_loss(Y, Y_hat):
    """Squeezes predictions and returns MSE loss."""

    if len(Y_hat.shape) > 1:
        Y_hat = jnp.squeeze(Y_hat, axis=1)

    loss = jnp.mean(jnp.square(Y - Y_hat))

    return loss


def cross_entropy_loss(Y, Y_hat, num_classes):
    """Applies log-softmax to predictions and one-hot encodes true values
       to compute and return cross-entropy loss.
    """

    Y_hat = jax.nn.log_softmax(Y_hat)

    Y = jax.nn.one_hot(Y, num_classes=num_classes)

    loss = -jnp.sum(Y * Y_hat)

    return loss
