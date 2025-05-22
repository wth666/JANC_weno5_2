import jax
import jax.numpy as jnp

##parallel settings##
num_devices = jax.local_device_count()
devices = jax.devices()

def split_and_distribute_data(grid):
    grid_list = jnp.array(jnp.split(grid,num_devices,axis=1))
    return grid_list