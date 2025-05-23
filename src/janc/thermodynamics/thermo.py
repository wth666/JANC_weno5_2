"""
The module extracts thermodynamic parameters 
from Cantera’s database and employs dedicated functions 
to compute critical thermodynamic state variables for gas mixtures: 
gas constant (R), enthalpy (h), specific heat ratio (gamma) and et al.

dependencies: jax & cantera(python version)
"""

import jax.numpy as jnp
from jax import vmap,lax,custom_vjp,debug
from ..preprocess import nondim
from ..preprocess.load import read_reaction_mechanism, get_cantera_coeffs
import os


max_iter = 5000
tol = 5e-9

species_M = None
Mex = None
Tcr = None
cp_cof_low = None
cp_cof_high = None
dcp_cof_low = None
dcp_cof_high = None
h_cof_low = None
h_cof_high = None
h_cof_low_chem = None
h_cof_high_chem = None
s_cof_low = None
s_cof_high = None
logcof_low = None
logcof_high = None
n = None
thermo_settings={'thermo_model':'nasa7'}
ReactionParams = {}

def set_thermo(thermo_config,nondim_config=None):
    global ReactionParams,thermo_settings,n,species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high
    
    if thermo_config['is_detailed_chemistry']:
        assert 'mechanism_diretory' in thermo_config,"You choosed detailed chemistry without specifying the diretory of your mechanism files, please specify 'chemistry_mechanism_diretory' in your dict of settings"
        _, ext = os.path.splitext(thermo_config['mechanism_diretory'])
        assert ext.lower() == '.yaml', "janc only read mech file with 【.yaml】 format, check https://cantera.org/3.1/userguide/ck2yaml-tutorial.html for more details"
        
        
        assert thermo_config['thermo_model']=='nasa7',"detailed chemistry requires thermo model to be 'nasa7'."
        if not os.path.isfile(thermo_config['mechanism_diretory']):
            raise FileNotFoundError('No mechanism file detected in the specified directory.')
    else:
        assert 'species' in thermo_config, "A list of strings containing the name of the species must be provided in the dict of settings with key name 'species'. Example:['H2','O2',...]."
    
    if thermo_config['thermo_model']=='nasa7':
        assert 'mechanism_diretory' in thermo_config,"Please provide the name of the nasa7_mech which can be identified by cantera (for example, 'gri30.yaml'), or the diretory of your own nasa7 mech (for example, '/content/my_own_mech.yaml')."
        _, ext = os.path.splitext(thermo_config['mechanism_diretory'])
        assert ext.lower() == '.yaml', "janc only read mech file with 【.yaml】 format. If you have standalone thermo data with 【.dat】 format, use 【ck2yaml --thermo=therm.dat】 in cantera to convert your file to 【.yaml】 format."
    else:
        if thermo_config['thermo_model']=='constant_gamma':
            assert 'gamma' in thermo_config, "The constant_gamma model require the value of gamma to be specified in the setting dict with key name 'gamma'."
        else:
            raise RuntimeError("The thermo model you specified is not supported, only 'nasa7' or 'constant_gamma' can be specified.")
            
    if thermo_config['is_detailed_chemistry']:
        ReactionParams = read_reaction_mechanism(thermo_config['mechanism_diretory'],nondim_config)
        mech = thermo_config['mechanism_diretory']
        species_list = ReactionParams['species']
        ns = ReactionParams['num_of_species']
        ni = ReactionParams['num_of_inert_species']
        n = ns - ni
    else:
        species_list = thermo_config['species']
        mech = thermo_config['mechanism_diretory']
        n = len(species_list)
    
    species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high = get_cantera_coeffs(species_list,mech,nondim_config)
    thermo_settings = thermo_config


def fill_Y(Y):
    Y_last = 1.0 - jnp.sum(Y,axis=0,keepdims=True)
    return jnp.concatenate([Y,Y_last],axis=0)

def get_R(Y):
    Y = fill_Y(Y)
    #expand_axes = range(species_M.ndim, Y.ndim)
    #Mex = jnp.expand_dims(species_M, tuple(expand_axes))
    R = jnp.sum(1/Mex*Y,axis=0,keepdims=True)
    return R


def get_thermo_properties_single(Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,T):
    mask = T<Tcr
    cp = jnp.where(mask, jnp.polyval(cp_cof_low, T), jnp.polyval(cp_cof_high, T))
    dcp = jnp.where(mask, jnp.polyval(dcp_cof_low, T), jnp.polyval(dcp_cof_high, T))
    h = jnp.where(mask, jnp.polyval(h_cof_low, T), jnp.polyval(h_cof_high, T))
    return cp, dcp, h

def get_gibbs_single(Tcr,h_cof_low,h_cof_high,s_cof_low,s_cof_high,logcof_low,logcof_high,T):
    mask = T<Tcr
    h = jnp.where(mask, jnp.polyval(h_cof_low, T), jnp.polyval(h_cof_high, T))
    s = jnp.where(mask, jnp.polyval(s_cof_low, T) + logcof_low*jnp.log((nondim.T0)*T), jnp.polyval(s_cof_high, T) + logcof_high*jnp.log((nondim.T0)*T))
    g = s - h/T
    return g

    
def get_thermo_properties(T):
    return vmap(get_thermo_properties_single,in_axes=(0,0,0,0,0,0,0,None))(Tcr,cp_cof_low,cp_cof_high,
                                       dcp_cof_low,dcp_cof_high,
                                       h_cof_low,h_cof_high,
                                       T)

def get_gibbs(T):
    return vmap(get_gibbs_single,in_axes=(0,0,0,0,0,0,0,None))(Tcr[0:n],h_cof_low_chem[0:n],h_cof_high_chem[0:n],
                                   s_cof_low[0:n],s_cof_high[0:n],
                                   logcof_low[0:n],logcof_high[0:n],
                                   T)

def get_thermo_nasa7(T, Y):
    """
    thermo properties evaluation with nasa7 polynomial
    """
    R = get_R(Y)
    Y = fill_Y(Y)
    cp_i, dcp_i, h_i = get_thermo_properties(T[0])
    cp = jnp.sum(cp_i*Y,axis=0,keepdims=True)
    h = jnp.sum(h_i*Y,axis=0,keepdims=True)
    dcp = jnp.sum(dcp_i*Y,axis=0,keepdims=True)
    gamma = cp/(cp-R)
    return cp, gamma, h, R, dcp

def e_eqn(T, e, Y):
    cp, gamma, h, R, dcp = get_thermo_nasa7(T, Y)
    res = ((h - R*T) - e)
    dres_dT = (cp - R)
    ddres_dT2 = dcp
    return res, dres_dT, ddres_dT2, gamma

'''
@custom_vjp
def get_T_nasa7(e, Y, initial_T_unused):
    T_min = 0.2
    T_max = 8000.0 / nondim.T0
    N_scan = 100  # number of scan intervals
    #tol = 1e-8
    #max_iter = 50

    spatial_shape = e.shape[1:]  # (1000, 600)
    base_T_scan = jnp.linspace(T_min, T_max, N_scan + 1)  # (101,)
    T_scan = base_T_scan[:, None, None, None]
    T_scan = jnp.broadcast_to(T_scan, (N_scan + 1, 1, *spatial_shape))  # (101, 1, 1000, 600)

    def e_eqn_at_T(T):  # T: (1, 1000, 600)
        cp, gamma, h, R, dcp = get_thermo_nasa7(T, Y)
        res = (h - R * T) - e
        return res

    res_scan = vmap(e_eqn_at_T)(T_scan)  # (101, 1, 1000, 600)
    res_scan = res_scan[:, 0, :, :]  # (101, 1000, 600)

    sign_change = res_scan[:-1] * res_scan[1:] < 0  # (100, 1000, 600)
    found = jnp.any(sign_change, axis=0)  # (1000, 600)
    valid_idx = jnp.argmax(sign_change, axis=0)  # (1000, 600)

    ix = jnp.arange(spatial_shape[0])[:, None]  # (1000,1)
    iy = jnp.arange(spatial_shape[1])[None, :]  # (1,600)

    T0_left = T_scan[valid_idx, 0, ix, iy]       # (1000, 600)
    T0_right = T_scan[valid_idx + 1, 0, ix, iy]  # (1000, 600)
    T0 = 0.5 * (T0_left + T0_right)               # (1000, 600)
    T0 = T0[None, :, :]  # (1, 1000, 600)

    def newton_solver(T0):
        def body_fun(args):
            T, i = args
            cp, gamma, h, R, dcp = get_thermo_nasa7(T, Y)
            res = (h - R * T) - e
            dres_dT = cp - R
            delta_T = -res / (dres_dT + 1e-12)
            delta_T = jnp.clip(delta_T, -0.5, 0.5)
            T_new = jnp.clip(T + delta_T, T_min, T_max)
            return T_new, i + 1

        def cond_fun(args):
            T, i = args
            cp, gamma, h, R, dcp = get_thermo_nasa7(T, Y)
            res = (h - R * T) - e
            max_res = jnp.max(jnp.abs(res))
            return (max_res > tol) & (i < max_iter)

        T_final, _ = lax.while_loop(cond_fun, body_fun, (T0, 0))
        cp, gamma, *_ = get_thermo_nasa7(T_final, Y)
        return jnp.concatenate([gamma, T_final], axis=0)  # (9+1, 1000, 600)

    def no_root_case():
        msg = "Error: no valid root found in get_T_nasa7."
        debug.print(msg)
        #assert False, msg
        dummy = jnp.full_like(e, jnp.nan)
        return jnp.concatenate([dummy, dummy], axis=0)  # (2, 1000, 600), 保证shape与newton_solver一致

    #return lax.cond(jnp.any(found), lambda _: newton_solver(T0), lambda _: no_root_case(), operand=None)
    return newton_solver(T0)
'''

'''
@custom_vjp
def get_T_nasa7(e,Y,initial_T):
    
    initial_res, initial_de_dT, initial_d2e_dT2, initial_gamma = e_eqn(initial_T,e,Y)

    def cond_fun(args):
        res, de_dT, d2e_dT2, T, gamma, i = args
        return (jnp.max(jnp.abs(res)) > tol) & (i < max_iter)

    def body_fun(args):
        res, de_dT, d2e_dT2, T, gamma, i = args
        delta_T = -2*res*de_dT/(2*jnp.power(de_dT,2)-res*d2e_dT2)
        T_new = T + delta_T
        res_new, de_dT_new, d2e_dT2_new, gamma_new = e_eqn(T_new,e,Y)
        return res_new, de_dT_new, d2e_dT2_new, T_new, gamma_new, i + 1

    initial_state = (initial_res, initial_de_dT, initial_d2e_dT2, initial_T, initial_gamma, 0)
    _, _, _, T_final, gamma_final, it = lax.while_loop(cond_fun, body_fun, initial_state)
    return jnp.concatenate([gamma_final, T_final],axis=0)
'''

T_min = 0.2
T_max = 8000.0 / nondim.T0
N_scan = 100  # number of scan intervals
@custom_vjp
def get_T_nasa7(e, Y, initial_T, tol, max_iter, 
                e_eqn, T_min, T_max, N_scan):
   
    # === 牛顿迭代内嵌 ===
    def solve_newton(T0, e, Y):
        def body_fun(val):
            T, _ = val
            res, dres, _, gamma = e_eqn(T, e, Y)
            delta = res / (dres + 1e-10)
            T_next = T - delta
            return (T_next, gamma)

        def cond_fun(val):
            T, _ = val
            res, *_ = e_eqn(T, e, Y)
            err = jnp.max(jnp.abs(res))
            return err > tol

        def newton_loop(T_init):
            val = (T_init, jnp.zeros_like(T_init))
            val = lax.while_loop(cond_fun, body_fun, val)
            return val  # (T_final, gamma)

        # vmap over batch (channel, H, W)
        T_final, gamma_final = newton_loop(T0)
        return T_final, gamma_final


    # === 扫描构造新的 T 初值 ===
    base_T = jnp.linspace(T_min, T_max, N_scan + 1)  # (N_scan+1,)

    def compute_res(T_scalar):
        T_full = jnp.ones((1, *spatial_shape)) * T_scalar
        res, *_ = e_eqn(T_full, e, Y)
        return res  # (1, H, W)

    res_scan = vmap(compute_res)(base_T)  # (N_scan+1, 1, H, W)
    res_scan = res_scan[:, 0, :, :]  # (N_scan+1, H, W)

    sign_change = (res_scan[:-1] * res_scan[1:] < 0)
    valid_idx = jnp.argmax(sign_change, axis=0)  # (H, W)

    T_left = base_T[valid_idx]
    T_right = base_T[valid_idx + 1]
    T0_scan = 0.5 * (T_left + T_right)  # (H, W)

    T0_init = jnp.where(failed, T0_scan, initial_T[0])  # (H, W)
    T0_init = T0_init[None, :, :]  # (1, H, W)

    # === 第三步：重新牛顿迭代 ===
    T_refined, gamma_refined = solve_newton(T0_init, e, Y)

    # === 第四步：合并结果 ===
    gamma_final =  gamma_refined
    T_final = T_refined

    return jnp.concatenate([gamma_final, T_final], axis=0)  # (2, H, W)





    
def get_T_fwd(e,Y,initial_T):
    aux_new = get_T_nasa7(e,Y,initial_T)
    return aux_new, (e,Y,aux_new)
    
def get_T_bwd(res, g):
    e, Y, aux_new = res
    T = aux_new[1:2]
    cp, _, h, R, dcp_dT = get_thermo(T,Y)
    cv = cp - R
    dcv_dT = dcp_dT
    dT_de = 1/cv
    
    dgamma_dT = (dcp_dT*cv-dcv_dT*cp)/(cv**2)
    dgamma_de = dgamma_dT*dT_de
    
    cp_i, dcp_i_dT, h_i = get_thermo_properties(T[0])
    e_i = h_i - 1/Mex*T
    dT_dY = (-e_i[0:-1]+e_i[-1:])/cv
    
    dR_dY = 1/Mex[0:-1]-1/Mex[-1:]
    dcp_dY = cp_i[0:-1]-cp_i[-1:]
    dcv_dY = dcp_dY - dR_dY
    
    dgamma_dY = (dcp_dY*cv-dcv_dY*cp)/(cv**2)
    dgamma_dY = dgamma_dT*dT_dY + dgamma_dY
    
    dL_dgamma = g[0:1]
    dL_dT = g[1:2]
    
    dL_de = dL_dgamma*dgamma_de + dL_dT*dT_de
    dL_dY = dL_dgamma*dgamma_dY + dL_dT*dT_dY
    
    
    return (dL_de, dL_dY, jnp.zeros_like(T))
    
get_T_nasa7.defvjp(get_T_fwd, get_T_bwd)

def get_thermo_constant_gamma(T, Y):
    R = get_R(Y)
    Y = fill_Y(Y)
    gamma = thermo_settings['gamma']
    cp = gamma/(gamma-1)*R
    h = cp*T
    gamma = jnp.full_like(T,gamma)
    return cp, gamma, h, R, None


def get_T_constant_gamma(e,Y,initial_T=None):
    gamma = thermo_settings['gamma']
    R = get_R(Y)
    T_final = e/(R/(gamma-1))
    gamma_final = jnp.full_like(e,gamma)
    return jnp.concatenate([gamma_final, T_final],axis=0)

get_thermo_func_dict = {'nasa7':get_thermo_nasa7,
                        'constant_gamma':get_thermo_constant_gamma}

get_T_func_dict = {'nasa7':get_T_nasa7,
                   'constant_gamma':get_T_constant_gamma}

def get_thermo(T,Y):
    return get_thermo_func_dict[thermo_settings['thermo_model']](T,Y)

def get_T(e,Y,initial_T):
    return get_T_func_dict[thermo_settings['thermo_model']](e,Y,initial_T)



    
    
    
