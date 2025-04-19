import numpy as np
import os
from os.path import exists

""" Saved objects have the following dimensions:
    all_model_samples     = num_chains x num_samples+1 x num_models x sample_dim
    all_model_obs         = num_chains x num_samples+1 x num_models x num_obs
    all_model_log_probs   = num_chains x num_samples+1 x num_models
    overall_time          = num_chains or float
    all_model_times       = num_chains x num_models
    overall_acc_ratio     = num_chains or float
    model_eval_counter    = num_chains x num_models
    model_acceptances     = num_chains x num_models
    acceptance_history    = num_chains x num_samples
    model_labels          = num_models
    current_sample_obs    = num_chains x num_models x num_obs
    current_sample_lls    = num_chains x num_models
    adaptive_matrix       = num_chains x adaptive_matrix
    epsilon               = num_chains or float
    average_tree_depth    = num_chains or float
"""



def save_data(filename, all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times,
              overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, 
              current_sample_lls, adaptive_matrix=None, epsilon=None, average_tree_depth=None):
    data_dict = {'all_model_samples': all_model_samples,
                 'all_model_obs': all_model_obs,
                 'all_model_log_probs': all_model_log_probs,
                 'overall_time': overall_time,
                 'all_model_times': all_model_times,
                 'overall_acc_ratio': overall_acc_ratio,
                 'model_eval_counter': model_eval_counter,
                 'model_acceptances': model_acceptances,
                 'acceptance_history': acceptance_history,
                 'model_labels': model_labels,
                 'current_sample_obs': current_sample_obs,
                 'current_sample_lls': current_sample_lls,
                 'adaptive_matrix': adaptive_matrix,
                 'epsilon': epsilon,
                 'average_tree_depth': average_tree_depth}
    np.save(filename, data_dict)


def load_data(filename):
    data = np.load(filename, allow_pickle=True)

    all_model_samples   = data.item().get('all_model_samples')
    all_model_obs       = data.item().get('all_model_obs')
    all_model_log_probs = data.item().get('all_model_log_probs')
    overall_time        = data.item().get('overall_time')
    all_model_times     = data.item().get('all_model_times')
    overall_acc_ratio   = data.item().get('overall_acc_ratio')
    model_eval_counter  = data.item().get('model_eval_counter')
    model_acceptances   = data.item().get('model_acceptances')
    acceptance_history  = data.item().get('acceptance_history')
    model_labels        = data.item().get('model_labels')
    current_sample_obs  = data.item().get('current_sample_obs')
    current_sample_lls  = data.item().get('current_sample_lls')
    adaptive_matrix     = data.item().get('adaptive_matrix')
    epsilon             = data.item().get('epsilon')
    average_tree_depth  = data.item().get('average_tree_depth')
    return all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times, \
            overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, \
            current_sample_lls, adaptive_matrix, epsilon, average_tree_depth


def add_data(filename, all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times,
            overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, 
            current_sample_lls, adaptive_matrix=None, epsilon=None, average_tree_depth=None):
    
    old_all_model_samples, old_all_model_obs, old_all_model_log_probs, old_overall_time, old_all_model_times, \
            old_overall_acc_ratio, old_model_eval_counter, old_model_acceptances, old_acceptance_history, _, _, _, _, _, \
            old_average_tree_depth = load_data(filename)

    if len(all_model_samples.shape) <= 3:     # Adding 1 chain data
        new_all_model_samples = np.vstack((old_all_model_samples, all_model_samples[1:]))
        new_all_model_obs = np.vstack((old_all_model_obs, all_model_obs[1:]))
        new_all_model_log_probs = np.vstack((old_all_model_log_probs, all_model_log_probs[1:]))
        new_overall_time = old_overall_time + overall_time
        new_all_model_times = old_all_model_times + all_model_times
        new_overall_acc_ratio = (overall_acc_ratio * acceptance_history.shape[0]
                    + old_overall_acc_ratio * old_acceptance_history.shape[0]) / (new_all_model_samples.shape[0] - 1)
        new_model_eval_counter = old_model_eval_counter + model_eval_counter 
        # Be careful in case a fidelity forward model never accepted (count = 0)
        new_model_acceptances = np.array([ (model_acceptances[k] * model_eval_counter[k]
                              + old_model_acceptances[k] * old_model_eval_counter[k]) / new_model_eval_counter[k]
                              if new_model_eval_counter[k] != 0 else 0
                              for k in range(len(model_labels)) ])
        new_acceptance_history = np.hstack((old_acceptance_history, acceptance_history))
        
        new_average_tree_depth = average_tree_depth
        if average_tree_depth:   # average_tree_depth is not None
            new_average_tree_depth = (average_tree_depth * acceptance_history.shape[0] 
                    + old_average_tree_depth * old_acceptance_history.shape[0]) / (new_all_model_samples.shape[0] - 1)
    
    else:                           # Adding data from multiple chains
        num_chains = acceptance_history.shape[0]

        new_all_model_samples = np.array([np.vstack((old_all_model_samples[i], all_model_samples[i][1:])) for i in range(num_chains)])
        new_all_model_obs = np.array([np.vstack((old_all_model_obs[i], all_model_obs[i][1:])) for i in range(num_chains)])
        new_all_model_log_probs = np.array([np.vstack((old_all_model_log_probs[i], all_model_log_probs[i][1:])) for i in range(num_chains)])
        new_overall_time = np.array([old_overall_time[i] + overall_time[i] for i in range(num_chains)])
        new_all_model_times = np.array([old_all_model_times[i] + all_model_times[i] for i in range(num_chains)])
        new_overall_acc_ratio = np.array([(overall_acc_ratio[i] * acceptance_history[i].shape[0]
                    + old_overall_acc_ratio[i] * old_acceptance_history[i].shape[0]) / (new_all_model_samples[i].shape[0] - 1) for i in range(num_chains)])
        new_model_eval_counter = np.array([old_model_eval_counter[i] + model_eval_counter[i] for i in range(num_chains)])
        # Be careful in case a fidelity forward model never accepted (count = 0)
        new_model_acceptances = np.array([ [(model_acceptances[i][k] * model_eval_counter[i][k]
                              + old_model_acceptances[i][k] * old_model_eval_counter[i][k]) / new_model_eval_counter[i][k]
                              if new_model_eval_counter[i][k] != 0 else 0
                              for k in range(len(model_labels)) ]
                              for i in range(num_chains)])
        new_acceptance_history = np.array([np.hstack((old_acceptance_history[i], acceptance_history[i])) for i in range(num_chains)])
        
        new_average_tree_depth = average_tree_depth
        if average_tree_depth.all():  # All elements of average_tree_depth are not None (if one is, they all are)
            new_average_tree_depth = np.array([(average_tree_depth[i] * acceptance_history[i].shape[0] 
                + old_average_tree_depth[i] * old_acceptance_history[i].shape[0]) / (new_all_model_samples[i].shape[0] - 1) for i in range(num_chains)])

    save_data(filename, new_all_model_samples, new_all_model_obs, new_all_model_log_probs, new_overall_time, new_all_model_times,
              new_overall_acc_ratio, new_model_eval_counter, new_model_acceptances, new_acceptance_history, model_labels, current_sample_obs, 
              current_sample_lls, adaptive_matrix, epsilon, new_average_tree_depth)


def merge_data(filename, files):
    """ Merge all the data from the individual files in 'files' (list of strings)
        into one file 'filename', then delete the individual files
    """

    master_all_model_samples   = []
    master_all_model_obs       = []
    master_all_model_log_probs = []
    master_overall_time        = []
    master_all_model_times     = []
    master_overall_acc_ratio   = []
    master_model_eval_counter  = []
    master_model_acceptances   = []
    master_acceptance_history  = []
    master_current_sample_obs  = []
    master_current_sample_lls  = []
    master_adaptive_matrix     = []
    master_epsilon             = []
    master_average_tree_depth  = []

    for file in files:
        all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times, \
        overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, \
        current_sample_lls, adaptive_matrix, epsilon, average_tree_depth = load_data(file)

        master_all_model_samples.append(all_model_samples)
        master_all_model_obs.append(all_model_obs)
        master_all_model_log_probs.append(all_model_log_probs)
        master_overall_time.append(overall_time)
        master_all_model_times.append(all_model_times)
        master_overall_acc_ratio.append(overall_acc_ratio)
        master_model_eval_counter.append(model_eval_counter)
        master_model_acceptances.append(model_acceptances)
        master_acceptance_history.append(acceptance_history)
        master_current_sample_obs.append(current_sample_obs)
        master_current_sample_lls.append(current_sample_lls)
        master_adaptive_matrix.append(adaptive_matrix)
        master_epsilon.append(epsilon)
        master_average_tree_depth.append(average_tree_depth)

    if exists(filename):
        print("Adding Data to Existing File")

        if len(files) == 1:
            add_data(filename, all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times, \
                    overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, \
                    current_sample_lls, adaptive_matrix, epsilon, average_tree_depth)
        else:
            add_data(filename,
                    np.array(master_all_model_samples), 
                    np.array(master_all_model_obs), 
                    np.array(master_all_model_log_probs), 
                    np.array(master_overall_time),
                    np.array(master_all_model_times),
                    np.array(master_overall_acc_ratio),
                    np.array(master_model_eval_counter),
                    np.array(master_model_acceptances),
                    np.array(master_acceptance_history),
                    model_labels,
                    np.array(master_current_sample_obs),
                    np.array(master_current_sample_lls),
                    np.array(master_adaptive_matrix),
                    np.array(master_epsilon),
                    np.array(master_average_tree_depth))
    else:
        print("Creating New Data File")

        if len(files) == 1:
            save_data(filename, all_model_samples, all_model_obs, all_model_log_probs, overall_time, all_model_times, \
                    overall_acc_ratio, model_eval_counter, model_acceptances, acceptance_history, model_labels, current_sample_obs, \
                    current_sample_lls, adaptive_matrix, epsilon, average_tree_depth)
        else:
            save_data(filename,
                    np.array(master_all_model_samples), 
                    np.array(master_all_model_obs), 
                    np.array(master_all_model_log_probs), 
                    np.array(master_overall_time),
                    np.array(master_all_model_times),
                    np.array(master_overall_acc_ratio),
                    np.array(master_model_eval_counter),
                    np.array(master_model_acceptances),
                    np.array(master_acceptance_history),
                    model_labels,
                    np.array(master_current_sample_obs),
                    np.array(master_current_sample_lls),
                    np.array(master_adaptive_matrix),
                    np.array(master_epsilon),
                    np.array(master_average_tree_depth))

    # Clean up by deleting individual chain files
    for file in files:
        os.remove(file)




