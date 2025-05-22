import jax.numpy as jnp
from jax import jit,vmap,pmap
from ..solver import aux_func
from .flux import weno5
from .flux import weno5_w
from ..thermodynamics import thermo
from ..thermodynamics import chemical
from ..boundary import boundary
from ..parallel import boundary as parallel_boundary
from functools import partial


from jaxamr import amr

def CFL(field,dx,dy,cfl=0.20):
    U, aux = field[0:-2],field[-2:]
    _,u,v,_,_,a = aux_func.U_to_prim(U,aux)
    cx = jnp.max(abs(u) + a)
    cy = jnp.max(abs(v) + a)
    dt = jnp.minimum(cfl*dx/cx,cfl*dy/cy)
    return dt

def set_solver(thermo_set, boundary_set, source_set = None, nondim_set = None, solver_mode='base'):
    thermo.set_thermo(thermo_set,nondim_set)
    boundary.set_boundary(boundary_set)
    aux_func.set_source_terms(source_set)
    if thermo.thermo_settings['is_detailed_chemistry']:
        chem_solver_type = 'implicit'
    else:
        chem_solver_type = 'explicit'
    
    if solver_mode == 'amr':
        
        @partial(vmap,in_axes=(0, 0, None, None, None))
        def rhs(U, aux, dx, dy, theta=None):
            aux = aux_func.update_aux(U, aux)
            physical_rhs = weno5(U,aux,dx,dy) + aux_func.source_terms(U[:,3:-3,3:-3], aux[:,3:-3,3:-3], theta)
            return jnp.pad(physical_rhs,pad_width=((0,0),(3,3),(3,3)))
    
    else:
        if solver_mode == 'parallel':
            def rhs(U,aux,dx,dy,theta=None):
                aux = aux_func.update_aux(U, aux)
                U_with_ghost,aux_with_ghost = parallel_boundary.boundary_conditions(U,aux,theta)
                physical_rhs = weno5(U_with_ghost,aux_with_ghost,dx,dy) + aux_func.source_terms(U, aux, theta)
                return physical_rhs
        else:
            def rhs(U,aux,dx,dy,theta=None):
                aux = aux_func.update_aux(U, aux)
                U_with_ghost,aux_with_ghost = boundary.boundary_conditions(U,aux,theta)
                physical_rhs = weno5_w(U_with_ghost,aux_with_ghost,dx,dy) + aux_func.source_terms(U, aux, theta)
                return physical_rhs
    
    if solver_mode == 'amr':
        def advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta=None):

            num = 3

            ghost_blk_data = amr.get_ghost_block_data(ref_blk_data, ref_blk_info)
            U,aux = ghost_blk_data[:,0:-2],ghost_blk_data[:,-2:]
            U1 = U + dt * rhs(U, aux, dx, dy, theta)
            blk_data1 = jnp.concatenate([U1,aux],axis=1)
            blk_data1 = amr.update_external_boundary(level, blk_data, blk_data1[..., num:-num, num:-num], ref_blk_info)


            ghost_blk_data1 = amr.get_ghost_block_data(blk_data1, ref_blk_info)
            U1 = ghost_blk_data1[:,0:-2]
            U2 = 3/4*U + 1/4*(U1 + dt * rhs(U1, aux, dx, dy, theta))
            blk_data2 = jnp.concatenate([U2,aux],axis=1)
            blk_data2 = amr.update_external_boundary(level, blk_data, blk_data2[..., num:-num, num:-num], ref_blk_info)

            ghost_blk_data2 = amr.get_ghost_block_data(blk_data2, ref_blk_info)
            U2 = ghost_blk_data2[:,0:-2]
            U3 = 1/3*U + 2/3*(U2 + dt * rhs(U2, aux, dx, dy, theta))
            blk_data3 = jnp.concatenate([U3,aux],axis=1)
            blk_data3 = amr.update_external_boundary(level, blk_data, blk_data3[..., num:-num, num:-num], ref_blk_info)
            
            return blk_data3
    
    else:
        def advance_flux(field,dx,dy,dt,theta=None):
            
            U, aux = field[0:-2],field[-2:]
            U1 = U + dt * rhs(U,aux,dx,dy,theta)
            U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,aux,dx,dy,theta))
            U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,aux,dx,dy,theta))
            field = jnp.concatenate([U3,aux],axis=0)
            
            return field
    
    def advance_source_term(field,dt):
        U, aux = field[0:-2],field[-2:]
        aux = aux_func.update_aux(U, aux)
        _,T = aux_func.aux_to_thermo(U,aux)
        rho = U[0:1]
        Y = U[4:]/rho
        drhoY = chemical.solve_implicit_rate(T,rho,Y,dt)

        p1 = U[0:4,:,:]
        p2 = U[4:,:,:] + drhoY
        U_new = jnp.concatenate([p1,p2],axis=0)
        return jnp.concatenate([U_new,aux],axis=0)

    
    if chem_solver_type == 'implicit':
        if solver_mode == 'amr':
            @partial(jit,static_argnames='level')
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info,theta=None):
                field_adv = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info,theta)
                field = vmap(advance_source_term,in_axes=(0, None))(field_adv,dt)
                return field
        
        else:
            if solver_mode == 'parallel':
                @partial(pmap,axis_name='x')    
                @jit    
                def advance_one_step(field,dx,dy,dt,theta=None):
                    field_adv = advance_flux(field,dx,dy,dt,theta)
                    field = advance_source_term(field_adv,dt)
                    return field
            else:
                @jit    
                def advance_one_step(field,dx,dy,dt,theta=None):
                    field_adv = advance_flux(field,dx,dy,dt,theta)
                    field = advance_source_term(field_adv,dt)
                    return field
    else:
        if solver_mode == 'amr':
            @partial(jit,static_argnames='level')
            def advance_one_step(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info,theta=None):
                field = advance_flux(level, blk_data, dx, dy, dt, ref_blk_data, ref_blk_info, theta)
                return field
        
        else:
            if solver_mode == 'parallel':
                @partial(pmap,axis_name='x')    
                @jit    
                def advance_one_step(field,dx,dy,dt,theta=None):
                    field = advance_flux(field,dx,dy,dt,theta)
                    return field
            else:
                @jit    
                def advance_one_step(field,dx,dy,dt,theta=None):
                    field = advance_flux(field,dx,dy,dt,theta)
                    return field
    
    print('solver is initialized successfully!')
    
    return advance_one_step,rhs
        

    


