# Repository Overview

This repository contains the code for Multi-Fidelity No-U-Turn Sampling (MF-NUTS) as well as Multi-Fidelity Random-Walk (MF-RW) and their application against an inverse wave problem.
This overview will describe everything one needs to know to use the code in this repository.

## Setup

All packages needed to work with this repository are included in `requirements.txt`.
In a new virtual environment, run the commands

`python -m pip install --requirement requirements.txt`

`pip install -U "jax[cuda12]"`

to get everything set up.


## Multi-Fidelity Sampling

The base sampling class `MF_Sampling` found in `mf_sampling.py` contains all general functions needed for MF-NUTS and MF-RW.

The key to this class is the parameter `model_labels`, which is a list of labels to indicate which forward model to use.
Multi-fidelity sampling will begin with the first forward model indicated in `model_labels` and will then work its way through the models until either the candidate sample is rejected or accepted by the highest-fidelity forward model.
In the case of MF-NUTS, NUTS itself is only run on the lowest-fidelity forward model, and all higher-fidelity forward models are only used to compute the HMC acceptance probabilities.
Note that when only one forward model is indicated in `model_labels` (i.e. `model_labels` has length 1), then MF-NUTS and MF-RW are equivalent to standard NUTS and RW, respectively.

In order to work, `MF_Sampling` must be inherited by another class that defines the methods `initialize_sample_obs_lls()`, `log_prior()`, `log_likelihood()`, `derivative_log_density()` (for NUTS only), `add_data()`, and `save_data()`:

- `initialize_sample_obs_lls()` takes the initial sample and computes its likelihood and corresponding observations for each forward model when they are not already provided. 
- `log_prior()` computes the log prior of a given sample.
- `log_likelihood()` computes the log likelihood of a given sample.
- `derivative_log_density()` computes the derivative of the log density, used during the Leapfrog method in HMC.
- `add_data()` adds all data collected since the last save to an existing data file.
- `save_data()` saves all data collected to a new data file.

All data collected is tracked by `MF_Sampling`, so it should be straightforward to implement `add_data()` and `save_data()` in the desired format.
The exact inputs and outputs for each method is detailed in `mf_sampling.py`.

The current sample `self.current_sample` will produce different observations and likelihoods for each forward model used, which is why we track `self.current_sample_obs` and `self.current_sample_lls` as lists of length `num_models` (the number of forward models).
This way, we never need to recompute the log likelihood of past samples.
Thus, the tracked list of samples in the Markov chain `self.all_model_samples` has shape `n_samples + 1 x num_models x sample_dim` where each sample along the `num_models` dimension is a sample that was accepted by the corresponding forward model. 
As a result, only the samples along the last `num_models` dimension are the ones in the true posterior distribution.
The same logic applies to `self.all_model_obs` and `self.all_model_log_probs`.

## Application to the Inverse Wave Problem

The class `Wave_Sampling` found in `wave_sampling.py` inherits `MF_Sampling` to specifically solve the inverse wave problem.
In this class, `model_labels` is substituted for `grid_sizes`, since the same forward model is used for each fidelity, with the only difference being the grid-refinements (grid_size) used to solve the wave problem.
The forward model `wave_solver()` can be found in `forward_models.py` along with two functions, `get_buoy()` and `get_observations()` which are used with the forward model to map samples to their observations.

The six methods in `MF_Sampling` specified above are overwritten in `Wave_Sampling`, but others are also overwritten.
This is because we perform a *parameter transformation* when solving the inverse wave problem, which requires us to correct the log density by the log determinant Jacobian of the inverse parameter transformation.
Consequently, all functions that compute the log density are overwritten, which are `rw_acceptance_probability()`, `compute_current_hmc_log_posterior()`, and `compute_new_hmc_log_posterior()`.
The log determinant Jacobian of the inverse parameter transformation is itself computed and saved in `compute_current_log_prior()` and `compute_new_log_prior()`, which are also overwritten.
An extra function, `detransform_samples()`, is also provided in `wave_sampling.py` outside of the `Wave_Sampling` class, which allows it to be accessed freely.

The methods `add_data()` and `save_data()` serve as wrappers for functions imported from `save_functions.py`.
The functions in this file save and load the data to `.npy` files.
In addition to `add_data()` and `save_data()` is a function `load_data()` and `merge_data()`, which is used to process data gathered in parallel processing.

### Walkthrough of `main.py`

Although it only addresses the inverse wave problem, `main.py` should still prove valuable as an example for how sampling with the `MF_Sampling` class can be best accomplished.
`main.py` is ran by executing the command 

`python main.py sampling_specs.json wave_specs.npy`

where `sampling_specs.json` contains the details for sampling:

- `n_chains` the number of chains desired for sampling in parallel (can be 1).
- `n_samples` the number of samples that each chain will sample over.
- `grid_sizes` a list of grid sizes that define the fidelity of each forward model.
- `initial_sample` either a sample in list form or `"random"` to draw a sample randomly from the prior (this argument is ignored when continuing chains).
- `save_filename` the name of the file to save the data to.
- `matrix_adapt_start` the number of iterations at which to start adapting the `adaptive_matrix`.
- `nuts` a bool that indicates whether to sample with MF-NUTS (true) or MF-RW (false).
- `target_acceptance` the target acceptance ratio for NUTS.
- `epsilon_adapt` the number of iterations to adapt epsilon in NUTS.
- `SIR_stages` the number of resampling stages to perform with SIR.
- `n_SIR_samples` the number of samples to evaluate during each SIR resampling stage.

Note that `sampling_specs.json` **is the only file that needs to be edited** to change sampling methodologies.

`wave_specs.npy` contains the problem specifications, observational data, and prior and likelihood parameters.
These values can be customized in `generate_wave_problem_specs.py`.
When this file is edited, it should be executed to generate an  updated `wave_specs.npy` file.
All of the values in `wave_specs.npy` are all set as global variables via a call to `wave_specs.init()` found in `wave_specs.py`.

When `main.py` is run, it checks to see if the specified data file `save_filename` already exists or not. 
If it does exist, it extracts the most recent data to continue sampling.
If the file does not exist, it either generates an initial sample randomly or uses one that was provided; it then sets the other sampling parameters as `None` (they will be initialized when the `Wave_Sampling` class is initialized).
`main.py` begins the sampling procedure by running `SIR_stages` number of SIR resampling stages, each with `n_SIR_samples` number of samples.
The data collected during SIR is not saved to `save_filename` (since the Markov chain they define is not reversible), but it is instead saved to another file that is conspicuously labelled to avoid confusion.
Once SIR is finished, the primary sampling begins.

Both SIR and the primary sampling utilize a function in `main.py` called `run_chains()`.
This function contains all the code used to multi-process.
To avoid conflicts with the sampling chains saving data to the same file, we define unique filenames for each chain to save its data to; then, when they have all finished sampling, all chain data files are merged together (with `merge_data()` from `save_functions.py`) into a single data file labelled `save_filename`, and all individual chain data files are deleted.
After all the data is collected and merged together, we then save another file with the samples detransformed back into parameter space, by calling the function `detransform_samples()` from `wave_sampling.py`.

## Plotting Data

Multiple functions for plotting data are provided in `data_plotting_functions.py` and can be run in `data_plotting.ipynb`.
All of these functions assume the data is in the format defined by `add_data()`, `save_data()`, and `merge_data()`.
These functions include:

- `print_stats()` to print out a detailed list of sampling statistics.
- `plot_posteriors()`, `plot_posterior_trace_together()`, `plot_posterior_predictives()` to plot the distribution for all parameters/observations together in a nice format.
- `plot_single_posterior()`, `plot_single_posterior_predictive()` to plot only the distribution of the specified parameter/observation.
- `plot_parameter_heatmap()` to plot a heatmap grid comparing the distribution of each parameter against each other.
- `plot_acceptance_history()` to plot the overall acceptance rate as the sampling progresses.
- `plot_sequential_resampling()` to plot the results of SIR for each parameter in a nice format.
- `plot_log_probs()` to plot the log probability of each sampling drawn for each chain.
- `plot_posterior_model_comparison()`, `plot_posterior_predictive_model_comparison()` to compare the plots of the distributions of each forward model.
- `plot_posterior_log_prob_comparison()`, `plot_posterior_predictive_log_prob_comparison()` to compare the distributions of all samples, separated by a log probability threshold.
- `plot_low_fidelity_chain_comparison()` to plot the lowest-fidelity posterior predictive distributions of each chain separately on the same plots.

Select data files are provided (zipped) in the `data` folder, which can be used to generate these plots.