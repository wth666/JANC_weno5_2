import jax
import jax.numpy as jnp
from .boundary_padding import pad, replace_lb, replace_rb, replace_ub, replace_bb
from . import slip_wall
from . import neumann
from . import pressure_outlet


boundary_func = {'left_boundary':None,
                'right_boundary':None,
                'up_boundary':None,
                'bottom_boundary':None}


def set_boundary(boundary_config:dict):
    global boundary_func
    
    
    if callable(boundary_config['left_boundary']):
        def left_boundary(padded_U,padded_aux,theta=None):
            U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
            U_lb,aux_lb = boundary_config['left_boundary'](U_lb,aux_lb,theta)
            U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
            return U_with_lb,aux_with_lb
    else:
        assert (boundary_config['left_boundary'] == 'slip_wall') or\
               (boundary_config['left_boundary'] == 'periodic') or\
                   (boundary_config['left_boundary'] == 'neumann') or\
               (boundary_config['left_boundary'] == 'pressure_outlet'),\
                'the bc type is not supported, please try custom  bc.'
                
        if boundary_config['left_boundary'] == 'slip_wall':
            def left_boundary(padded_U,padded_aux,theta=None):
                U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
                U_lb,aux_lb = slip_wall.left(U_lb,aux_lb)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
        elif boundary_config['left_boundary'] == 'periodic':
            def left_boundary(padded_U,padded_aux,theta=None):
                return padded_U,padded_aux
        elif boundary_config['left_boundary'] == 'neumann':
            def left_boundary(padded_U,padded_aux,theta=None):
                U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
                U_lb,aux_lb = neumann.left(U_lb,aux_lb)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
        elif boundary_config['left_boundary'] == 'pressure_outlet':
            def left_boundary(padded_U,padded_aux,theta=None):
                U_lb,aux_lb = padded_U[:,3:6,3:-3],padded_aux[:,3:6,3:-3]
                U_lb,aux_lb = pressure_outlet.left(U_lb,aux_lb,theta)
                U_with_lb,aux_with_lb = replace_lb(U_lb,aux_lb,padded_U,padded_aux)
                return U_with_lb,aux_with_lb
            
    boundary_func['left_boundary'] = left_boundary
    
    
    if callable(boundary_config['right_boundary']):
        def right_boundary(padded_U,padded_aux,theta=None):
            U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
            U_rb,aux_rb = boundary_config['right_boundary'](U_rb,aux_rb,theta)
            U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
            return U_with_rb,aux_with_rb
    else:
        assert (boundary_config['right_boundary'] == 'slip_wall') or\
               (boundary_config['right_boundary'] == 'periodic') or\
                   (boundary_config['right_boundary'] == 'neumann') or\
               (boundary_config['right_boundary'] == 'pressure_outlet'),\
                'the bc type is not supported, please try custom  bc.'
                
        if boundary_config['right_boundary'] == 'slip_wall':
            def right_boundary(padded_U,padded_aux,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
                U_rb,aux_rb = slip_wall.right(U_rb,aux_rb)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
        elif boundary_config['right_boundary'] == 'periodic':
            def right_boundary(padded_U,padded_aux,theta=None):
                return padded_U,padded_aux
        elif boundary_config['right_boundary'] == 'neumann':
            def right_boundary(padded_U,padded_aux,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
                U_rb,aux_rb = neumann.right(U_rb,aux_rb)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
        elif boundary_config['right_boundary'] == 'pressure_outlet':
            def right_boundary(padded_U,padded_aux,theta=None):
                U_rb,aux_rb = padded_U[:,-6:-3,3:-3],padded_aux[:,-6:-3,3:-3]
                U_rb,aux_rb = pressure_outlet.right(U_rb,aux_rb,theta)
                U_with_rb,aux_with_rb = replace_rb(U_rb,aux_rb,padded_U,padded_aux)
                return U_with_rb,aux_with_rb
            
    boundary_func['right_boundary'] = right_boundary    
    
    
    
    if callable(boundary_config['bottom_boundary']):
        def bottom_boundary(padded_U,padded_aux,theta=None):
            U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
            U_bb,aux_bb = boundary_config['bottom_boundary'](U_bb,aux_bb,theta)
            U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
            return U_with_bb,aux_with_bb
    else:
        assert (boundary_config['bottom_boundary'] == 'slip_wall') or\
               (boundary_config['bottom_boundary'] == 'periodic') or\
                   (boundary_config['bottom_boundary'] == 'neumann') or\
               (boundary_config['bottom_boundary'] == 'pressure_outlet'),\
                'the bc type is not supported, please try custom  bc.'
                
        if boundary_config['bottom_boundary'] == 'slip_wall':
            def bottom_boundary(padded_U,padded_aux,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
                U_bb,aux_bb = slip_wall.bottom(U_bb,aux_bb)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        elif boundary_config['bottom_boundary'] == 'periodic':
            def bottom_boundary(padded_U,padded_aux,theta=None):
                return padded_U,padded_aux
        elif boundary_config['bottom_boundary'] == 'neumann':
            def bottom_boundary(padded_U,padded_aux,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
                U_bb,aux_bb = neumann.bottom(U_bb,aux_bb)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
        elif boundary_config['bottom_boundary'] == 'pressure_outlet':
            def bottom_boundary(padded_U,padded_aux,theta=None):
                U_bb,aux_bb = padded_U[:,3:-3,3:6],padded_aux[:,3:-3,3:6]
                U_bb,aux_bb = pressure_outlet.bottom(U_bb,aux_bb,theta)
                U_with_bb,aux_with_bb = replace_bb(U_bb,aux_bb,padded_U,padded_aux)
                return U_with_bb,aux_with_bb
            
    boundary_func['bottom_boundary'] = bottom_boundary
    
    
    
    if callable(boundary_config['up_boundary']):
        def up_boundary(padded_U,padded_aux,theta=None):
            U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
            U_ub,aux_ub = boundary_config['up_boundary'](U_ub,aux_ub,theta)
            U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
            return U_with_ub,aux_with_ub
    else:
        assert (boundary_config['up_boundary'] == 'slip_wall') or\
               (boundary_config['up_boundary'] == 'periodic') or\
                   (boundary_config['up_boundary'] == 'neumann') or\
               (boundary_config['up_boundary'] == 'pressure_outlet'),\
                'the bc type is not supported, please try custom  bc.'
                
        if boundary_config['up_boundary'] == 'slip_wall':
            def up_boundary(padded_U,padded_aux,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
                U_ub,aux_ub = slip_wall.up(U_ub,aux_ub)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        elif boundary_config['up_boundary'] == 'periodic':
            def up_boundary(padded_U,padded_aux,theta=None):
                return padded_U,padded_aux
        elif boundary_config['up_boundary'] == 'neumann':
            def up_boundary(padded_U,padded_aux,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
                U_ub,aux_ub = neumann.up(U_ub,aux_ub)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
        elif boundary_config['up_boundary'] == 'pressure_outlet':
            def up_boundary(padded_U,padded_aux,theta=None):
                U_ub,aux_ub = padded_U[:,3:-3,-6:-3],padded_aux[:,3:-3,-6:-3]
                U_ub,aux_ub = pressure_outlet.up(U_ub,aux_ub,theta)
                U_with_ub,aux_with_ub = replace_ub(U_ub,aux_ub,padded_U,padded_aux)
                return U_with_ub,aux_with_ub
            
    boundary_func['up_boundary'] = up_boundary


def boundary_conditions(U, aux, theta=None):
    U_periodic_pad,aux_periodic_pad = pad(U,aux)
    U_with_lb,aux_with_lb = boundary_func['left_boundary'](U_periodic_pad,aux_periodic_pad, theta)
    U_with_rb,aux_with_rb = boundary_func['right_boundary'](U_with_lb,aux_with_lb,theta)
    U_with_bb,aux_with_bb = boundary_func['bottom_boundary'](U_with_rb,aux_with_rb,theta)
    U_with_ghost_cell,aux_with_ghost_cell = boundary_func['up_boundary'](U_with_bb,aux_with_bb,theta)
    return U_with_ghost_cell,aux_with_ghost_cell

