import jax.numpy as jnp

Rg = 8.314463
P0 = 1.0
T0 = 1.0
R0 = 1.0
x0 = 1.0
rho0 = P0/(R0*T0)
M0 = Rg/R0
e0 = P0/rho0
u0 = jnp.sqrt(P0/rho0)
t0 = x0/u0

def set_nondim(nondim_config):
    global P0,T0,R0,x0,rho0,M0,e0,u0,t0
    if nondim_config is not None:
        P0 = nondim_config['P0']
        T0 = nondim_config['T0']
        R0 = nondim_config['R0']
        x0 = nondim_config['x0']
        rho0 = P0/(R0*T0)
        M0 = Rg/R0
        e0 = P0/rho0
        u0 = jnp.sqrt(P0/rho0)
        t0 = x0/u0