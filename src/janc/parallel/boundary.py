import jax
import jax.numpy as jnp
from .boundary_padding import pad
from ..boundary import boundary

##parallel settings##
num_devices = jax.local_device_count()
devices = jax.devices()

boundary_func = boundary.boundary_func


def boundary_conditions(U,aux,theta=None):
    device_idx = jax.lax.axis_index('x')
    U_periodic_pad,aux_periodic_pad = pad(U,aux)
    U_with_lb,aux_with_lb = jax.lax.cond(device_idx==0,lambda:boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad,theta),lambda:(U_periodic_pad,aux_periodic_pad))
    U_with_rb,aux_with_rb = jax.lax.cond(device_idx==(num_devices-1),lambda:boundary_func['right_boundary'](U_with_lb,aux_with_lb,theta),lambda:(U_with_lb,aux_with_lb))
    U_with_bb,aux_with_bb = boundary_func['bottom_boundary'](U_with_rb,aux_with_rb,theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary_func['up_boundary'](U_with_bb,aux_with_bb,theta)
    return U_with_ghost_cell,aux_with_ghost_cell
