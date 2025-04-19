import save_functions
import wave_sampling
import wave_specs

import numpy as np
from scipy.stats import beta, multivariate_normal
import multiprocessing
import functools
import sys, json, os
from os.path import exists

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


""" Execute Script with two arguments:
    Argument 1: the name of the file with specified sampling parameters
                (i.e. sampling_specs.json)
    Argument 2: the name of the file with the inverse wave problem specifications
                (i.e. wave_specs.npy)
"""

# Load in sampling parameters from the specified file (as an argument)
sampling_specs_filename = sys.argv[1]
with open(sampling_specs_filename)  as f:
    dict = json.load(f)

# input_sample is either a list of values or "random"
n_chains, n_samples, grid_sizes, input_sample, save_filename, matrix_adapt_start, nuts, target_acceptance, epsilon_adapt, \
    SIR_stages, n_SIR_samples =   \
                                dict["n_chains"],   \
                                dict["n_samples"],  \
                                dict["grid_sizes"],  \
                                dict["initial_sample"],  \
                                dict["save_filename"],   \
                                dict["matrix_adapt_start"], \
                                dict["nuts"],       \
                                dict["target_acceptance"],  \
                                dict["epsilon_adapt"],     \
                                dict["SIR_stages"], \
                                dict["n_SIR_samples"]

# Load in global inverse wave problem specifications from the specified file (as an argument)
wave_specs_filename = sys.argv[2]
wave_specs.init(wave_specs_filename)




def smap(f):
    """ Used for multiprocessing in run_chains """
    return f()


def run_chains(save_filename, chain_filenames, initial_samples, current_sample_obs, current_sample_lls, 
               sample_means, adaptive_matrices, epsilons, nuts, n_chains, n_samples, grid_sizes, 
               target_acceptance, matrix_adapt_start, epsilon_adapt):
    """ Runs multiple sampling chains. """

    wave_samplers = [wave_sampling.Wave_Sampling(initial_samples[i], current_sample_obs[i], current_sample_lls[i], 
                 sample_means[i], grid_sizes, adaptive_matrices[i], epsilons[i]) for i in range(n_chains)]
    
    if nuts:
        # Create a list of sampling functions with the input already specified
        funcs = [functools.partial(wave_samplers[i].sample_multi_fidelity_nuts, 
                                    save_filename=chain_filenames[i], 
                                    n_samples=n_samples, 
                                    matrix_adapt_start=matrix_adapt_start,
                                    epsilon_adapt=epsilon_adapt,
                                    delta=target_acceptance) 
                for i in range(n_chains)]
    else:
        # Create a list of sampling functions with the input already specified
        funcs = [functools.partial(wave_samplers[i].sample_multi_fidelity_rw, 
                                    save_filename=chain_filenames[i], 
                                    n_samples=n_samples, 
                                    matrix_adapt_start=matrix_adapt_start) 
                for i in range(n_chains)]
        
    # Run all sampling functions in parallel
    pool = multiprocessing.Pool(n_chains)
    pool.map(smap, funcs)

    pool.close()

    # Merge and delete all chain files into one data file
    save_functions.merge_data(save_filename, chain_filenames)

    # Save an additional datafile with detransformed samples in sample space
    wave_sampling.detransform_samples(save_filename, save_filename[:-4] + "_detransformed.npy")




if __name__ == "__main__":

    try: multiprocessing.set_start_method('spawn')
    except RuntimeError: pass

    if not isinstance(grid_sizes, list):
        raise ValueError("grid_sizes must be inputted as a list (even if it only contains one element).")
    
    if epsilon_adapt > n_samples:
        raise ValueError("The burn-in phase must finish within the initial sampling session (epsilon_adapt <= n_samples)")

    if epsilon_adapt > n_SIR_samples and SIR_stages != 0:
        raise ValueError("The burn-in phase must finish within each sequential resampling session (epsilon_adapt <= n_SIR_samples)")

    if n_chains == 1 and SIR_stages != 0:
        raise ValueError("One chain cannot be sequentially resampled; please change the number of chains or set sequential_resample = 0.")
    

    # Check if save_filename exists, if not, start a new chain, else continue the chains
    if exists(save_filename):
        print("Continuing Chains")
        
        all_model_samples, _, _, _, _, _, _, _, _, old_grid_sizes, current_sample_obs, current_sample_lls, \
            adaptive_matrices, epsilons, _ = save_functions.load_data(save_filename)

        # Check that the grid sizes match
        if not np.array_equal(grid_sizes, old_grid_sizes):
            raise ValueError("The specified grid_sizes do not match the previously-used grid_sizes.")

        if len(all_model_samples.shape) <= 3:   # There is only one chain
            initial_samples = all_model_samples[-1][-1]
            all_model_samples = all_model_samples[np.newaxis]
            # Add extra dimension for multiprocessing
            current_sample_obs = np.array(current_sample_obs)[np.newaxis]
            current_sample_lls = np.array(current_sample_lls)[np.newaxis]
            adaptive_matrices = adaptive_matrices[np.newaxis]
            epsilons = np.array([epsilons])
        else:
            initial_samples = np.array([chain[-1][-1] for chain in all_model_samples])     

        # Do NOT include the final sample since that will be the next sample used to update the matrix
        sample_means = np.mean(all_model_samples[:,:-1,-1], axis=1)  

    else:
        print("Starting New Chains")

        if input_sample == "random":
            # Randomly draw guesses from prior distribution to initialize chains
            gauss_params = multivariate_normal.rvs(mean=wave_specs.PRIOR_MEAN, cov=np.diag(wave_specs.PRIOR_COV_DIAG), size=n_chains)
            beta_param = beta.rvs(a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B, size=n_chains)

            if n_chains == 1:
                initial_samples = np.hstack(( gauss_params, beta_param ))
            else:
                gauss_params = gauss_params.T
                initial_samples = np.vstack(( gauss_params, beta_param )).T
        else:
            initial_samples = np.array(input_sample)

        # If the sample is 1D, add another dimension
        if len(initial_samples.shape) == 1: 
            initial_samples = initial_samples[np.newaxis]

        # Initialize sample_means as the initial samples
        sample_means = initial_samples.copy()

        # Initialize these values as None
        current_sample_obs = np.array([None for _ in range(n_chains)])
        current_sample_lls = np.array([None for _ in range(n_chains)])
        epsilons = [None for _ in range(n_chains)]
        adaptive_matrices  = np.array([None for _ in range(n_chains)])


    # If the given sample is 1D, add another dimension
    if len(initial_samples.shape) == 1: 
        initial_samples = initial_samples[np.newaxis]

    # Check that the number of chains and states match
    if initial_samples.shape[0] != n_chains:
        raise ValueError("The specified number of chains does not match the number of initial states.")
    

    # Sequentially resample the specified number of times
    for resample in range(SIR_stages):
        print(f"Sequential Resampling Stage {resample+1}")

        # This assumes save_filename ends in ".npy"
        chain_filenames = [save_filename[:-4] + f"_{SIR_stages}resamples_{n_SIR_samples}points_chain{i+1}.npy" for i in range(n_chains)]

        # Save all resample data to the same file
        run_chains(save_filename[:-4] + f"_{SIR_stages}resamples_{n_SIR_samples}points.npy", chain_filenames, 
                   initial_samples, current_sample_obs, current_sample_lls, sample_means, adaptive_matrices, 
                   epsilons, nuts, n_chains, n_SIR_samples, grid_sizes, target_acceptance, matrix_adapt_start, 
                   epsilon_adapt)

        all_samples, _, all_model_log_probs, _, _, _, _, _, _, _, all_current_obs, all_current_lls, \
            _, _, _ = save_functions.load_data(save_filename[:-4] + f"_{SIR_stages}resamples_{n_SIR_samples}points.npy")
        
        # Extract the initial_samples and log probs from the last sample accepted by the highest fidelity model
        initial_samples = np.array([chain[-1][-1] for chain in all_samples])
        high_f_sample_log_probs = np.array([chain[-1][-1] for chain in all_model_log_probs])

        # Create (normed) weights using the log prob of each chain's sample
        weights = np.exp(high_f_sample_log_probs) / np.sum(np.exp(high_f_sample_log_probs))

        # Draw n_chains indices representing which samples to start the new chains with
        new_indices = np.random.choice(n_chains, n_chains, p=weights)

        # Create the new distribution of samples to start with 
        initial_samples = np.array([initial_samples[new_index] for new_index in new_indices])
        current_sample_obs = np.array([all_current_obs[new_index] for new_index in new_indices])
        current_sample_lls = np.array([all_current_lls[new_index] for new_index in new_indices])

    if SIR_stages != 0:
        # Now that we're done sequentially resampling, run the actual chains
        print("Begin Primary Sampling")


    # This assumes save_filename ends in ".npy"
    chain_filenames = [save_filename[:-4] + f"_chain{i+1}.npy" for i in range(n_chains)]

    run_chains(save_filename, chain_filenames, initial_samples, current_sample_obs, current_sample_lls, 
               sample_means, adaptive_matrices, epsilons, nuts, n_chains, n_samples, grid_sizes, 
               target_acceptance, matrix_adapt_start, epsilon_adapt)

