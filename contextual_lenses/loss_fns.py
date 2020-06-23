"""Loss functions

Jax loss functions for computing gradient updates."""


import jax
import jax.numpy as jnp


def mse_loss(Y, Y_hat, **loss_fn_kwargs):
    
    if len(Y_hat.shape) > 1:
        Y_hat = jnp.squeeze(Y_hat, axis=1)
    
    loss = jnp.mean(jnp.square(Y-Y_hat))
    
    return loss
    

def cross_entropy_loss(Y, Y_hat, **loss_fn_kwargs):
    
    num_categories = list(loss_fn_kwargs.values())[0]
    
    Y = jax.nn.one_hot(Y, num_classes=num_categories)
    
    loss = jnp.sum(Y * Y_lhat)
    
    return loss
