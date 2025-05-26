import jax.numpy as jnp
from jax import jit
from ..solver import aux_func

p = 2
eps = 1e-6
C1 = 1 / 10
C2 = 3 / 5
C3 = 3 / 10


@jit
def splitFlux_LF(ixy, U, aux):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1

    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    theta = zx*um + zy*vm
    Hplus = 0.5 * (F + theta * U)
    Hminus = 0.5 * (F - theta * U)
    return Hplus, Hminus

@jit
def splitFlux_LF_w(ixy, U, aux):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1

    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p)], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p)], axis=0)
    um = jnp.nanmax(abs(u) + a)
    vm = jnp.nanmax(abs(v) + a)
    theta = zx*um + zy*vm
    #theta = zx * (jnp.abs(u) + a) + zy * (jnp.abs(v) + a)
    Hplus = 0.5 * (F + theta * U[0:4,:,:])
    Hminus = 0.5 * (F - theta * U[0:4,:,:])
    return Hplus, Hminus

@jit
def WENO_plus_x(f):
    fj = f[:,2:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3]
    fjp2 = f[:,4:-1,3:-3]
    fjm1 = f[:,1:-4,3:-3]
    fjm2 = f[:,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,1:,:] - fj_halfp[:,0:-1,:]
    return dfj

@jit
def WENO_plus_y(f):

    fj = f[:,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-2]
    fjp2 = f[:,3:-3,4:-1]
    fjm1 = f[:,3:-3,1:-4]
    fjm2 = f[:,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,:,1:] - fj_halfp[:,:,0:-1]

    return dfj

@jit
def WENO_minus_x(f):

    fj = f[:,3:-2,3:-3]
    fjp1 = f[:,4:-1,3:-3]
    fjp2 = f[:,5:,3:-3]
    fjm1 = f[:,2:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,1:,:] - fj_halfm[:,0:-1,:])

    return dfj

@jit
def WENO_minus_y(f):

    fj = f[:,3:-3,3:-2]
    fjp1 = f[:,3:-3,4:-1]
    fjp2 = f[:,3:-3,5:]
    fjm1 = f[:,3:-3,2:-3]
    fjm2 = f[:,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,:,1:] - fj_halfm[:,:,0:-1])

    return dfj







@jit
def WENO_plus_x_w(f):

    fj = f[:,2:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3]
    fjp2 = f[:,4:-1,3:-3]
    fjm1 = f[:,1:-4,3:-3]
    fjm2 = f[:,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3

    return fj_halfp[:,1:,:],fj_halfp[:,0:-1,:]

@jit
def WENO_plus_y_w(f):

    fj = f[:,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-2]
    fjp2 = f[:,3:-3,4:-1]
    fjm1 = f[:,3:-3,1:-4]
    fjm2 = f[:,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3

    return fj_halfp[:,:,1:],fj_halfp[:,:,0:-1]

@jit
def WENO_minus_x_w(f):

    fj = f[:,3:-2,3:-3]
    fjp1 = f[:,4:-1,3:-3]
    fjp2 = f[:,5:,3:-3]
    fjm1 = f[:,2:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3

    return fj_halfm[:,1:,:],fj_halfm[:,0:-1,:]

@jit
def WENO_minus_y_w(f):

    fj = f[:,3:-3,3:-2]
    fjp1 = f[:,3:-3,4:-1]
    fjp2 = f[:,3:-3,5:]
    fjm1 = f[:,3:-3,2:-3]
    fjm2 = f[:,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3

    return fj_halfm[:,:,1:],fj_halfm[:,:,0:-1]


@jit
def weno5(U,aux,dx,dy):
    Fplus, Fminus = splitFlux_LF(1, U, aux)
    Gplus, Gminus = splitFlux_LF(2, U, aux)

    dFp = WENO_plus_x(Fplus)
    dFm = WENO_minus_x(Fminus)

    dGp = WENO_plus_y(Gplus)
    dGm = WENO_minus_y(Gminus)

    dF = dFp + dFm
    dG = dGp + dGm

    netflux = dF/dx + dG/dy

    return -netflux

@jit
def weno5_w(U,aux,dx,dy):
    Fplus, Fminus = splitFlux_LF_w(1, U, aux)
    Gplus, Gminus = splitFlux_LF_w(2, U, aux)

    Y = U[4:]/U[0:1]
    
    fj1,fj2 = WENO_plus_x_w(Fplus)
    Y1, Y2 = WENO_plus_x_w(Y)
    dFp = jnp.concatenate([fj1 - fj2,Y1*fj1[0:1]-Y2*fj2[0:1]],axis=0)
    fj1, fj2 = WENO_minus_x_w(Fminus)
    Y1, Y2 = WENO_minus_x_w(Y)
    dFm = jnp.concatenate([fj1 - fj2,Y1*fj1[0:1]-Y2*fj2[0:1]],axis=0)

    fj1, fj2 = WENO_plus_y_w(Gplus)
    Y1, Y2 = WENO_plus_y_w(Y)
    dGp = jnp.concatenate([fj1 - fj2,Y1*fj1[0:1]-Y2*fj2[0:1]],axis=0)
    fj1, fj2 = WENO_minus_y_w(Gminus)
    Y1, Y2 = WENO_minus_y_w(Y)
    dGm = jnp.concatenate([fj1 - fj2,Y1*fj1[0:1]-Y2*fj2[0:1]],axis=0) 

    dF = dFp + dFm
    dG = dGp + dGm

    netflux = dF/dx + dG/dy

    return -netflux

@jit
def weno5_amr(field,dx,dy):
    
    U,aux_old = field[0:-2],field[-2:]
    aux = aux_func.update_aux(U,aux_old)
    
    Fplus, Fminus = splitFlux_LF(1, U, aux)
    Gplus, Gminus = splitFlux_LF(2, U, aux)

    dFp = WENO_plus_x(Fplus)
    dFm = WENO_minus_x(Fminus)

    dGp = WENO_plus_y(Gplus)
    dGm = WENO_minus_y(Gminus)

    dF = dFp + dFm
    dG = dGp + dGm

    netflux = dF/dx + dG/dy

    return -netflux









@jit
def WENO_L_x(f):
    fj = f[:,2:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3]
    fjp2 = f[:,4:-1,3:-3]
    fjm1 = f[:,1:-4,3:-3]
    fjm2 = f[:,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3

    return fj_halfp

@jit
def WENO_L_y(f):

    fj = f[:,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-2]
    fjp2 = f[:,3:-3,4:-1]
    fjm1 = f[:,3:-3,1:-4]
    fjm2 = f[:,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3

    return fj_halfp

@jit
def WENO_R_x(f):

    fj = f[:,3:-2,3:-3]
    fjp1 = f[:,4:-1,3:-3]
    fjp2 = f[:,5:,3:-3]
    fjm1 = f[:,2:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3

    return fj_halfm

@jit
def WENO_R_y(f):

    fj = f[:,3:-3,3:-2]
    fjp1 = f[:,3:-3,4:-1]
    fjp2 = f[:,3:-3,5:]
    fjm1 = f[:,3:-3,2:-3]
    fjm2 = f[:,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3

    return fj_halfm



@jit
def HLLC_flux(Ul, Ur, aux_l, aux_r, ixy):
    # 提取左右状态
    rhoL, uL, vL, YL, pL, aL = aux_func.U_to_prim(Ul, aux_l)
    rhoR, uR, vR, YR, pR, aR = aux_func.U_to_prim(Ur, aux_r)

    EL = Ul[3:4,:,:] / rhoL
    ER = Ur[3:4,:,:] / rhoR

    # 用 ixy 控制提取的方向变量，避免 if
    u_nL = jnp.where(ixy == 1, uL, vL)
    u_nR = jnp.where(ixy == 1, uR, vR)
    tangL = jnp.where(ixy == 1, vL, uL)
    tangR = jnp.where(ixy == 1, vR, uR)

    # 波速估计
    SL = jnp.minimum(u_nL - aL, u_nR - aR)
    SR = jnp.maximum(u_nL + aL, u_nR + aR)

    # 中间波速 S*
    S_star = (pR - pL + rhoL * u_nL * (SL - u_nL) - rhoR * u_nR * (SR - u_nR)) / (
        rhoL * (SL - u_nL) - rhoR * (SR - u_nR) + 1e-6
    )

    # 左右通量
    FL = flux(Ul, aux_l, ixy)
    FR = flux(Ur, aux_r, ixy)

    # 定义星区状态函数
    def U_star(U, rho, u_n, tang, S, S_star, p, side):
        factor = rho * (S - u_n) / (S - S_star + 1e-6)
        E = U[3:4,:,:] / rho
        Y = U[4:,:,:] / rho

        # 构造星区守恒变量
        Ust = jnp.stack([
            factor,                              # rho*
            factor * jnp.where(ixy == 1, S_star, tang),  # rho*u
            factor * jnp.where(ixy == 1, tang, S_star),  # rho*v
            factor * (E + (S_star - u_n) * (S_star + p / (rho * (S - u_n + 1e-6)))),  # rho*E*
            factor * Y                           # rho*Y
        ], axis=0)
        return Ust

    UL_star = U_star(Ul, rhoL, u_nL, tangL, SL, S_star, pL, 'L')
    UR_star = U_star(Ur, rhoR, u_nR, tangR, SR, S_star, pR, 'R')

    F_star_L = FL + SL * (UL_star - Ul)
    F_star_R = FR + SR * (UR_star - Ur)

    # HLLC分段选择
    flux_HLLC = jnp.where(
        SL >= 0, FL,
        jnp.where(S_star >= 0, F_star_L,
        jnp.where(SR > 0, F_star_R, FR))
    )

    return flux_HLLC


@jit
def weno5_HLLC(U, aux, dx, dy):

    Ul = WENO_L_x(U)
    Ur = WENO_R_x(U)
    auxl = WENO_L_x(aux)
    auxr = WENO_R_x(aux)
    aux_l = aux_func.update_aux(Ul, auxl)
    aux_r = aux_func.update_aux(Ur, auxr)
    flux_hllc = HLLC_flux(Ul, Ur, aux_l, aux_r, ixy=1)  # x方向
    dF = (flux_hllc[:, 1:, :] - flux_hllc[:, :-1, :]) / dx

    Ul = WENO_L_y(U)
    Ur = WENO_R_y(U)
    auxl = WENO_L_y(aux)
    auxr = WENO_R_y(aux)
    aux_l = aux_func.update_aux(Ul, auxl)
    aux_r = aux_func.update_aux(Ur, auxr)
    flux_hllc = HLLC_flux(Ul, Ur, aux_l, aux_r, ixy=2)  # y方向
    dG = (flux_hllc[:, :, 1:] - flux_hllc[:, :, :-1]) / dy

    netflux = dF + dG

    return -netflux

@jit
def flux(U, aux, ixy):
    rho,u,v,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[3:4,:,:]

    zx = (ixy == 1) * 1
    zy = (ixy == 2) * 1

    F = zx*jnp.concatenate([rho * u, rho * u ** 2 + p, rho * u * v, u * (rhoE + p), rho * u * Y], axis=0) + zy*jnp.concatenate([rho * v, rho * u * v, rho * v ** 2 + p, v * (rhoE + p), rho * v * Y], axis=0)
    
    return F















