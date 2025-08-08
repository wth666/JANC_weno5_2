"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""

import jax.numpy as jnp
from jax import vmap,jit
from ..preprocess import nondim
from ..thermodynamics import thermo

Rg = 8.314463


def reactionConstant_i(T, X, i, k, n):

# Reaction data
    A = thermo.ReactionParams["A"][i]
    B = thermo.ReactionParams["B"][i]
    EakOverRu = thermo.ReactionParams["Ea/Ru"][i]
    A0 = thermo.ReactionParams["A0"][i]
    B0 = thermo.ReactionParams["B0"][i]
    Ea0 = thermo.ReactionParams["Ea0/Ru"][i]
    Ainf = thermo.ReactionParams["Ainf"][i]
    Binf = thermo.ReactionParams["Binf"][i]
    Eainf = thermo.ReactionParams["Eainf/Ru"][i]
    falloff_type = thermo.ReactionParams["falloff_type"][i]
    falloff_params = thermo.ReactionParams["falloff_params"][i]

    # Stoichiometry
    vf_i = thermo.ReactionParams["vf"][i]
    vb_i = thermo.ReactionParams["vb"][i]
    vf_ik = vf_i[k]
    vb_ik = vb_i[k]
    vf_in = vf_i[n]
    vb_in = vb_i[n]
    vsum = thermo.ReactionParams["vsum"][i]

    # Third-body
    is_third_body = thermo.ReactionParams["is_third_body"][i]
    is_falloff = thermo.ReactionParams["is_falloff"][i]

    aij = thermo.ReactionParams["third_body_coeffs"][i]
    aij_X_sum_TF = jnp.sum(aij * X, axis=0, keepdims=True)
    aij_X_sum = (is_third_body - is_falloff) * aij_X_sum_TF + (1 - (is_third_body - is_falloff))
    #aij_X_sum = is_third_body * aij_X_sum_TF + (1 - is_third_body)

    X = jnp.clip(X,min=1e-50)
    X = X[0:thermo.n,:,:]
    log_X = jnp.log(X)
    ain = thermo.ReactionParams["third_body_coeffs"][i,n]
    Mn = thermo.species_M[n]
    #kf = kf_i*jnp.exp(jnp.sum(vf_i*log_X,axis=0,keepdims=True))

    # Falloff处理


    def falloff_branch(_):
        kf0_i = A0 * T**B0 * jnp.exp(-Ea0 / T)
        kfinf_i = Ainf * T**Binf * jnp.exp(-Eainf / T)
        Pr = kf0_i * aij_X_sum_TF / kfinf_i
        log10_Pr = jnp.log(jnp.clip(Pr, 1e-30)) / jnp.log(10)

        def lindemann_fn(_):
            return jnp.ones_like(T)

        def troe_fn(_):
            A_troe, T3, T1, T2 = falloff_params
            log10_F_cent = jnp.log((1 - A_troe) * jnp.exp(-T / T3) + A_troe * jnp.exp(-T / T1) + jnp.exp(-T2 / T)) / jnp.log(10)
            c = -0.4 - 0.67 * log10_F_cent
            n_f = 0.75 - 1.27 * log10_F_cent
            f1 = (log10_Pr + c) / (n_f - 0.14 * (log10_Pr + c))
            #print(A_troe.shape)
            return 10.0 ** (log10_F_cent / (1.0 + f1**2))

        def unsupported_fn(_):
            return jnp.full_like(T, jnp.nan)  # 或者 jnp.nan，再用 jnp.where 处理报错

        F = jax.lax.switch(falloff_type, [unsupported_fn, lindemann_fn, troe_fn], None)
        kf_i = (kfinf_i * Pr / (1 + Pr)) * F
        dkf_i_drhonYn = kfinf_i * F * 1/(1+Pr)**2 * kf0_i * ain / Mn / kfinf_i
        #kf_i = kfinf_i
        #dkf_i_drhonYn = jnp.zeros_like(T)
        return kf_i, dkf_i_drhonYn

    def regular_branch(_):
        kf_i = A * T**B * jnp.exp(-EakOverRu / T)
        dkf_i_drhonYn = jnp.zeros_like(T)
        return kf_i, dkf_i_drhonYn

    kf_i, dkf_i_drhonYn = jax.lax.cond(is_falloff, falloff_branch, regular_branch, operand=None)



    keq_i = jnp.exp(jnp.sum((vb_i-vf_i)*(get_gibbs(T[0,:,:])),axis=0,keepdims=True))*((101325/P0/T)**vsum)
    #keq_i = jnp.exp(jnp.sum((vb_i-vf_i)*(get_gibbs(T[0,:,:])),axis=0,keepdims=True))*((101325/(T*T0*Rg))**vsum)
    #keq_i = (101325/P0/T)**vsum
    kb_i = kf_i/keq_i
    dkb_i_drhonYn = dkf_i_drhonYn / keq_i

    Cf = jnp.exp(jnp.sum(vf_i*log_X,axis=0,keepdims=True))
    kf = kf_i * Cf

    Cb = jnp.exp(jnp.sum(vb_i*log_X,axis=0,keepdims=True))
    kb = kb_i * Cb

    w_kOverM_i = (vb_ik-vf_ik)*aij_X_sum*(kf-kb)
    vb_in = vb_i[n]
    vf_in = vf_i[n]
    Xn = jnp.expand_dims(X[n,:,:],0)

    dwk_drhonYn_OverMk_i = (vb_ik-vf_ik)*(kf-kb)*ain/Mn*(is_third_body-is_falloff) + 1/(Mn*Xn)*(vb_ik-vf_ik)*aij_X_sum*(vf_in*kf-vb_in*kb) + (vb_ik-vf_ik)*aij_X_sum*(Cf*dkf_i_drhonYn-Cb*dkb_i_drhonYn)
    #dwk_drhonYn_OverMk_i = (vb_ik-vf_ik)*(kf-kb)*ain/Mn + 1/(Mn*Xn)*(vb_ik-vf_ik)*aij_X_sum*(vf_in*kf-vb_in*kb) + (vb_ik-vf_ik)*aij_X_sum*(Cf*dkf_i_drhonYn-Cb*dkb_i_drhonYn)

    return w_kOverM_i, dwk_drhonYn_OverMk_i

def reaction_rate_with_derievative(T,X,k,n):
    Mk = thermo.species_M[k]
    i = jnp.arange(thermo.ReactionParams["num_of_reactions"])
    w_kOverM_i, dwk_drhonYn_OverMk_i = vmap(reactionConstant_i,in_axes=(None,None,0,None,None))(T,X,i,k,n)
    w_k = Mk*jnp.sum(w_kOverM_i,axis=0,keepdims=False)
    dwk_drhonYn = Mk*jnp.sum(dwk_drhonYn_OverMk_i,axis=0,keepdims=False)
    return w_k[0,:,:], dwk_drhonYn[0,:,:]

def construct_matrix_equation(T,X,dt):
    matrix_fcn = vmap(vmap(reaction_rate_with_derievative,in_axes=(None,None,None,0)),in_axes=(None,None,0,None))
    k = jnp.arange(thermo.n)
    n = jnp.arange(thermo.n)
    w_k, dwk_drhonYn = matrix_fcn(T,X,k,n)
    S = jnp.transpose(w_k[:,0:1,:,:],(2,3,0,1))
    DSDU = jnp.transpose(dwk_drhonYn,(2,3,0,1))
    I = jnp.eye(thermo.n)
    A = I/dt - DSDU
    b = S
    return A, b

@jit
def solve_implicit_rate(T,rho,Y,dt):
    Y = thermo.fill_Y(Y)
    rhoY = rho*Y
    X = rhoY/(thermo.Mex)
    A, b = construct_matrix_equation(T,X,dt)
    drhoY = jnp.linalg.solve(A,b)
    drhoY = jnp.transpose(drhoY[:,:,:,0],(2,0,1))
    dY = drhoY/rho
    dY = jnp.clip(dY,min=-Y[0:-1],max=1-Y[0:-1])
    return rho*dY




