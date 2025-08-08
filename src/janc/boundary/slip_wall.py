import jax.numpy as jnp

def left(U_bd, aux_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,2:3,:],U_bd[:,1:2,:],U_bd[:,0:1,:]],axis=1)
    
    U_bd_ghost = U_bd_ghost.at[3,:,0].set(U_bd_ghost[3,:,0] - 0.5*U_bd_ghost[0,:,0]*(U_bd_ghost[2,:,0]/U_bd_ghost[0,:,0])**2)
    U_bd_ghost = U_bd_ghost.at[3,:,-1].set(U_bd_ghost[3,:,-1] - 0.5*U_bd_ghost[0,:,-1]*(U_bd_ghost[2,:,-1]/U_bd_ghost[0,:,-1])**2)
    U_bd_ghost = U_bd_ghost.at[2,:,0].set(0)
    U_bd_ghost = U_bd_ghost.at[2,:,-1].set(0)
              
    aux_bd_ghost = jnp.concatenate([aux_bd[:,2:3,:],aux_bd[:,1:2,:],aux_bd[:,0:1,:]],axis=1)
    U_bd_ghost = U_bd_ghost.at[1].multiply(-1)
    return U_bd_ghost, aux_bd_ghost

def right(U_bd, aux_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,-1:,:],U_bd[:,-2:-1,:],U_bd[:,-3:-2,:]],axis=1)
    aux_bd_ghost = jnp.concatenate([aux_bd[:,-1:,:],aux_bd[:,-2:-1,:],aux_bd[:,-3:-2,:]],axis=1)
    U_bd_ghost = U_bd_ghost.at[1].multiply(-1)
    return U_bd_ghost, aux_bd_ghost

def bottom(U_bd, aux_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,2:3],U_bd[:,:,1:2],U_bd[:,:,0:1]],axis=2)

    U_bd_ghost = U_bd_ghost.at[3,0,:].set(U_bd_ghost[3,0,:] - 0.5*U_bd_ghost[0,0,:]*(U_bd_ghost[1,0,:]/U_bd_ghost[0,0,:])**2)
    U_bd_ghost = U_bd_ghost.at[1,0,:].set(0)

    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,2:3],aux_bd[:,:,1:2],aux_bd[:,:,0:1]],axis=2)
    U_bd_ghost = U_bd_ghost.at[2].multiply(-1)
    return U_bd_ghost, aux_bd_ghost

def up(U_bd, aux_bd):
    U_bd_ghost = jnp.concatenate([U_bd[:,:,-1:],U_bd[:,:,-2:-1],U_bd[:,:,-3:-2]],axis=2)

    U_bd_ghost = U_bd_ghost.at[3,0,:].set(U_bd_ghost[3,0,:] - 0.5*U_bd_ghost[0,0,:]*(U_bd_ghost[1,0,:]/U_bd_ghost[0,0,:])**2)
    U_bd_ghost = U_bd_ghost.at[1,0,:].set(0)
    
    aux_bd_ghost = jnp.concatenate([aux_bd[:,:,-1:],aux_bd[:,:,-2:-1],aux_bd[:,:,-3:-2]],axis=2)
    U_bd_ghost = U_bd_ghost.at[2].multiply(-1)

    return U_bd_ghost, aux_bd_ghost


