import cantera as ct
import jax.numpy as jnp
from ..preprocess import nondim
#import os

Rg = 8.314463

def read_reaction_mechanism(file_path, nondim_config=None):

    gas = ct.Solution(file_path)
    species_list = gas.species_names

    vf = gas.reactant_stoich_coeffs.T
    vb = gas.product_stoich_coeffs.T

    n_reactions, n_species = vf.shape

    non_zero_mask = jnp.any(vf + vb != 0, axis=0)
    zero_col_mask = jnp.all(vf + vb == 0, axis=0)
    vf = vf[:, non_zero_mask]
    vb = vb[:, non_zero_mask]
    inert_check = jnp.logical_or(~jnp.any(zero_col_mask), jnp.all(zero_col_mask[jnp.argmax(zero_col_mask):]))
    assert inert_check, "Inert species must be the last elements in species_list"
    num_of_inert_species = jnp.sum(zero_col_mask)

    vf_sum = jnp.sum(vf, axis=1)
    vb_sum = jnp.sum(vb, axis=1)

    A, B, Ea = [], [], []
    A0, B0, Ea0 = [], [], []
    Ainf, Binf, Eainf = [], [], []
    falloff_type = []
    falloff_params = []

    third_body_coeffs = jnp.zeros((n_reactions, n_species))
    is_third_body = jnp.zeros((n_reactions,))
    is_falloff = jnp.zeros((n_reactions,))

    '''
    for i, reaction in enumerate(gas.reactions()):
        rate = reaction.rate
        print(f"Reaction {i}, type: {reaction.reaction_type}, rate type: {type(rate)}")
    '''



    for i, reaction in enumerate(gas.reactions()):
        reaction_order = sum(reaction.reactants.values())
        #print(vf_sum)

        if reaction.reaction_type in ['three-body-Arrhenius', 'falloff-Lindemann', 'falloff-Troe']:
            efficiencies = reaction.third_body.efficiencies
            for j, species in enumerate(species_list):
                third_body_coeffs = third_body_coeffs.at[i, j].set(efficiencies.get(species, 1.0))

        is_third_body = is_third_body.at[i].set(jnp.any(third_body_coeffs[i, :]) * 1)

        if reaction.reaction_type in ['falloff-Lindemann', 'falloff-Troe']:
            is_falloff = is_falloff.at[i].set(1)

            low = reaction.rate.low_rate
            high = reaction.rate.high_rate
            A0.append(low.pre_exponential_factor * (1e-3) ** (reaction_order))
            B0.append(low.temperature_exponent)
            Ea0.append(low.activation_energy / 1000)

            Ainf.append(high.pre_exponential_factor * (1e-3) ** (reaction_order - 1))
            Binf.append(high.temperature_exponent)
            Eainf.append(high.activation_energy / 1000)

            if reaction.reaction_type == 'falloff-Troe':
                f = reaction.rate
                #print(f.falloff_coeffs)
                falloff_type.append(2)
                falloff_params.append(list(f.falloff_coeffs))
            else:
                falloff_type.append(1)
                falloff_params.append([0.0, 0.0, 0.0, 0.0])

            # 统一主数组占位
            A.append(0.0); B.append(0.0); Ea.append(0.0)
        else:
            rate = reaction.rate
            if reaction.reaction_type == 'three-body-Arrhenius':
                A.append(rate.pre_exponential_factor * (1e-3) ** (reaction_order))
            else:
                A.append(rate.pre_exponential_factor * (1e-3) ** (reaction_order - 1))
            B.append(rate.temperature_exponent)
            Ea.append(rate.activation_energy / 1000)

            A0.append(0.0); B0.append(0.0); Ea0.append(0.0)
            Ainf.append(0.0); Binf.append(0.0); Eainf.append(0.0)
            falloff_type.append(0)
            falloff_params.append([0.0, 0.0, 0.0, 0.0])

    # 转为 jnp
    A = jnp.array(A); B = jnp.array(B); Ea = jnp.array(Ea)
    A0 = jnp.array(A0); B0 = jnp.array(B0); Ea0 = jnp.array(Ea0)
    Ainf = jnp.array(Ainf); Binf = jnp.array(Binf); Eainf = jnp.array(Eainf)
    falloff_type = jnp.array(falloff_type)
    falloff_params = jnp.array(falloff_params)

    # 无量纲化
    A = (nondim.t0 / nondim.rho0) * (nondim.T0 ** B) * ((nondim.rho0 / nondim.M0) ** (vf_sum + is_third_body)) * nondim.M0 * A
    EaOverRu = Ea / (nondim.e0 * nondim.M0)

    A0 = (nondim.t0 / nondim.rho0) * (nondim.T0 ** B0) * ((nondim.rho0 / nondim.M0) ** (vf_sum + 1)) * nondim.M0 * A0
    Ea0OverRu = Ea0 / (nondim.e0 * nondim.M0)

    Ainf = (nondim.t0 / nondim.rho0) * (nondim.T0 ** Binf) * ((nondim.rho0 / nondim.M0) ** vf_sum) * nondim.M0 * Ainf
    EainfOverRu = Eainf / (nondim.e0 * nondim.M0)

    falloff_params = falloff_params.at[:,1:].set(falloff_params[:,1:]/nondim.T0)


    # 形状处理
    third_body_coeffs = jnp.expand_dims(third_body_coeffs, (2, 3))
    vf = jnp.expand_dims(vf, (2, 3))
    vb = jnp.expand_dims(vb, (2, 3))

    ReactionParams =  {
        'species': species_list,
        'vf': vf,
        'vb': vb,
        'A': A,
        'B': B,
        'Ea/Ru': EaOverRu,
        'is_third_body': is_third_body,
        'third_body_coeffs': third_body_coeffs,
        'is_falloff': is_falloff,
        'falloff_type': falloff_type,
        'falloff_params': falloff_params,
        'A0': A0,
        'B0': B0,
        'Ea0/Ru': Ea0OverRu,
        'Ainf': Ainf,
        'Binf': Binf,
        'Eainf/Ru': EainfOverRu,
        'num_of_reactions': n_reactions,
        'num_of_species': n_species,
        'num_of_inert_species': num_of_inert_species,
        'vsum': vb_sum - vf_sum
    }
    return ReactionParams


def get_cantera_coeffs(species_list,mech='gri30.yaml',nondim_config=None):
    nondim.set_nondim(nondim_config)
    gas = ct.Solution(mech)
    species_M = []
    Tcr = []
    coeffs_low = []
    coeffs_high = []
    for specie_name in species_list:
        sp = gas.species(specie_name)
        nasa_poly = sp.thermo
        Tcr.append(nasa_poly.coeffs[0])
        coeffs_low.append(nasa_poly.coeffs[8:15])
        coeffs_high.append(nasa_poly.coeffs[1:8])
        species_M.append(sp.molecular_weight)
    
    coeffs_low = jnp.array(coeffs_low)*jnp.array([[1,nondim.T0,nondim.T0**2,nondim.T0**3,nondim.T0**4,1/nondim.T0,1]])
    coeffs_high = jnp.array(coeffs_high)*jnp.array([[1,nondim.T0,nondim.T0**2,nondim.T0**3,nondim.T0**4,1/nondim.T0,1]])
    species_M = jnp.array(species_M)/(1000*nondim.M0)
    Mex = jnp.expand_dims(species_M,(1,2))
    
    Tcr = jnp.array(Tcr)/nondim.T0
    
    cp_cof_low = jnp.flip(coeffs_low[:,0:5],axis=1)/species_M[:,None]
    cp_cof_high = jnp.flip(coeffs_high[:,0:5],axis=1)/species_M[:,None]
    
    dcp_cof_low = cp_cof_low[:,0:-1]*jnp.array([[4,3,2,1]])
    dcp_cof_high = cp_cof_high[:,0:-1]*jnp.array([[4,3,2,1]])
    
    h_cof_low = jnp.flip(jnp.roll(coeffs_low[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])/species_M[:,None]
    h_cof_high = jnp.flip(jnp.roll(coeffs_high[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])/species_M[:,None]
    
    h_cof_low_chem = jnp.flip(jnp.roll(coeffs_low[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])
    h_cof_high_chem = jnp.flip(jnp.roll(coeffs_high[:,0:6],1,axis=1),axis=1)*jnp.array([[1/5,1/4,1/3,1/2,1,1]])
    
    s_cof_low = jnp.flip(jnp.concatenate([coeffs_low[:,-1:],coeffs_low[:,1:5]],axis=1),axis=1)*jnp.array([[1/4,1/3,1/2,1,1]])
    s_cof_high = jnp.flip(jnp.concatenate([coeffs_high[:,-1:],coeffs_high[:,1:5]],axis=1),axis=1)*jnp.array([[1/4,1/3,1/2,1,1]])
    
    logcof_low = coeffs_low[:,0]
    logcof_high = coeffs_high[:,0]
    
    return species_M,Mex,Tcr,cp_cof_low,cp_cof_high,dcp_cof_low,dcp_cof_high,h_cof_low,h_cof_high,h_cof_low_chem,h_cof_high_chem,s_cof_low,s_cof_high,logcof_low,logcof_high




