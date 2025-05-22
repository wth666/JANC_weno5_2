import jax
import jax.numpy as jnp
from ..thermodynamics import thermo

def pressure_outlet(state_out,gamma_out,T_out,normal_vel,Pb):
    rho_out = state_out[0:1,:,:]
    u_out = state_out[1:2,:,:]/rho_out
    v_out = state_out[2:3,:,:]/rho_out
    Y_out = state_out[4:,:,:]/rho_out
    R_out = thermo.get_R(Y_out)
    p_out = rho_out*(R_out*T_out)
    a_out = jnp.sqrt(gamma_out*p_out/rho_out)
    mask = (normal_vel/a_out < 1)
    rho_cor_out = jax.lax.select(mask, Pb / (p_out / rho_out),rho_out)
    p_cor_out = jax.lax.select(mask, Pb*jnp.ones_like(p_out),p_out)
    T_cor_out = jax.lax.select(mask, p_cor_out/(rho_cor_out*R_out),T_out)
    
    #p_cor_out = p_out
    #T_cor_out = T_out
    #rho_cor_out = rho_out
    
    _, gamma_out, h_out, _, _ = thermo.get_thermo(T_cor_out,Y_out)
    U_bd = jnp.concatenate([rho_cor_out, rho_cor_out * u_out, rho_cor_out * v_out,
                      rho_cor_out*h_out - p_cor_out + 0.5 * rho_cor_out * (u_out ** 2 + v_out ** 2),
                      rho_cor_out * Y_out], axis=0)
    aux_bd = jnp.concatenate([gamma_out,T_cor_out], axis=0)
    return U_bd, aux_bd

def left(U_bd, aux_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,0:1,:]
    gamma_out = aux_bd[0:1,0:1,:]
    T_out = aux_bd[1:2,0:1,:]
    normal_vel = -state_out[1:2,:,:]/state_out[0:1,:,:]
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,-1:,:]
    gamma_out = aux_bd[0:1,-1:,:]
    T_out = aux_bd[1:2,-1:,:]
    normal_vel = state_out[1:2,:,:]/state_out[0:1,:,:]
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel, Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,0:1,:],U_bd[:,0:1,:],U_bd[:,0:1,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,0:1,:],aux_bd[:,0:1,:],aux_bd[:,0:1,:]],axis=1)
    return U_bd_ghost, aux_bd_ghost

def bottom(U_bd, aux_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,:,0:1]
    gamma_out = aux_bd[0:1,:,0:1]
    T_out = aux_bd[1:2,:,0:1]
    normal_vel = -state_out[2:3,:,:]/state_out[0:1,:,:]
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    return U_bd_ghost, aux_bd_ghost

def up(U_bd, aux_bd, theta):
    Pb = theta['Pb']
    state_out = U_bd[:,:,-1:]
    gamma_out = aux_bd[0:1,:,-1:]
    T_out = aux_bd[1:2,:,-1:]
    normal_vel = state_out[2:3,:,:]/state_out[0:1,:,:]
    U_bd,aux_bd = pressure_outlet(state_out, gamma_out, T_out, normal_vel,Pb)  
    U_bd_ghost = jnp.concatenate([U_bd[:,:,0:1],U_bd[:,:,0:1],U_bd[:,:,0:1]],axis=2)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,0:1],aux_bd[:,:,0:1],aux_bd[:,:,0:1]],axis=2)
    return U_bd_ghost, aux_bd_ghost

