import save_functions
import wave_sampling
import wave_specs

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta, norm
import seaborn as sns


# Global variables subject to customization
MODEL_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    ]
PARAMETER_LABELS = [
    r"$\mu_{11}$",
    r"$\mu_{12}$",
    r"$\mu_{21}$",
    r"$\mu_{22}$",
    r"$C_1$",
    r"$C_2$",
    r"$c$",
    ]
OBSERVATION_LABELS = [rf"$W_{i + 1}$" 
                      for i in range(wave_specs.OBS_VALS.shape[0])] + [
                      rf"$T_{i + 1}$" 
                      for i in range(wave_specs.OBS_VALS.shape[0])
                      ]


def algorithm_name(epsilon, model_labels):
    """ Helper function to return name of algorithm given data. """

    if epsilon[0]:
        algorithm = "HMC"
    else:
        algorithm = "RW"

    if len(model_labels) > 1:
        algorithm = "Multi-Fidelity " + algorithm

    return algorithm


def convert_seconds(seconds):
    """ Helper function used in plot_stats(). """
    h = seconds // (60 * 60)
    m = (seconds - h * 60 * 60) // 60
    s = seconds - (h * 60 * 60) - (m * 60)
    return [int(h), int(m), s]


def print_stats(filename, print_adaptive_matrix=False):
    (
        all_model_samples,
        _,
        _,
        overall_time,
        all_model_times,
        total_acc_ratio,
        model_eval_counter,
        model_acceptances,
        _,
        model_labels,
        _,
        _,
        adaptive_matrix,
        epsilon,
        average_tree_depth,
    ) = save_functions.load_data(filename)

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)
    if all_model_samples.shape[0] == 1:
        all_model_times = np.expand_dims(all_model_times, axis=0)
        model_eval_counter = np.expand_dims(model_eval_counter, axis=0)
        model_acceptances = np.expand_dims(model_acceptances, axis=0)
        epsilon = np.array([epsilon])
        average_tree_depth = np.array([average_tree_depth])
        overall_time = np.array([overall_time])

    algorithm = algorithm_name(epsilon, model_labels)

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    full_data = np.vstack(all_model_samples)
    full_model_times = np.sum(np.vstack(all_model_times), axis=0)
    full_model_eval_counter = np.sum(model_eval_counter, axis=0)
    full_model_acceptances = (
        np.sum(model_acceptances * model_eval_counter, axis=0) / full_model_eval_counter
        )
    overall_acc_ratio = (
        np.sum(total_acc_ratio * all_model_samples.shape[1]) / full_data.shape[0]
        )
    num_chains = all_model_samples.shape[0]

    # Process times
    chain_time_string = "Overall Chain Times             = ("
    for time_count in overall_time:
        hrs, mins, secs = convert_seconds(time_count)
        chain_time_string += f"{hrs}:{mins}:{round(secs, 2)}"
        if time_count != overall_time[-1]:
            chain_time_string += ", "
        else:
            chain_time_string += ")"

    total_time = sum(overall_time)
    total_hrs, total_mins, total_secs = convert_seconds(total_time)

    # Print important results
    model_label_str = np.array2string(
        np.array(model_labels),
        precision=2,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")
    model_counts_str = np.array2string(
        model_eval_counter,
        precision=2,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")

    ind_ratios_str = np.array2string(
        model_acceptances,
        precision=3,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")
    chain_acc_ratios = np.array2string(
        np.array(
            [ratio for ratio in total_acc_ratio]
            if num_chains > 1 else [total_acc_ratio]
            ),
        precision=4,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")
    overall_acc_ratios = np.array2string(
        full_model_acceptances,
        precision=3,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")

    all_model_average_times_str = np.array2string(
        all_model_times / model_eval_counter,
        precision=4,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")
    chain_average_times = np.array2string(
        overall_time / all_model_samples.shape[1],
        precision=4,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")
    overall_average_model_times = np.array2string(
        full_model_times / full_model_eval_counter,
        precision=4,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        ).replace("\n", "")

    tree_depth_str = np.array2string(
        average_tree_depth,
        precision=3,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=", ",
        )

    print(f"{algorithm} Final Results:\n")
    print(f"Model Labels (Grid Sizes)       = {model_label_str}")
    print(f"Total Number of Samples         = {full_data.shape[0]}")
    print(f"Model Evaluations               = {model_counts_str[1:-1]}\n")

    print(f"Chain Model Acc. Ratios         = {ind_ratios_str[1:-1]}")
    print(f"Overall Chain Acc. Ratios       = ({chain_acc_ratios[1:-1]})")
    print(f"Overall Model Acc. Ratios       = {overall_acc_ratios}")
    print(f"Overall Acc. Ratio              = {overall_acc_ratio}\n")

    print(f"Chain Avg Model Time/Sample     = {all_model_average_times_str[1:-1]}")
    print(f"Overall Chain Avg Time/Sample   = ({chain_average_times[1:-1]})")
    print(f"Overall Avg Model Time/Sample   = {overall_average_model_times}")
    print(
        f"Overall Avg Time/Sample         = {np.round(total_time / full_data.shape[0], 4)}\n"
    )

    if epsilon[0] is not None:
        print(
            f"Chain Epsilons                  = {[float(round(eps, 4)) for eps in epsilon]}"
        )
        print(f"Avg Tree Depths                 = {tree_depth_str[1:-1]}\n")

    print(chain_time_string)
    print(
        f"Overall Time                    = {total_hrs}:{total_mins}:{round(total_secs, 2)}\n"
    )

    if print_adaptive_matrix:
        print(f"Adaptive Matrices               = {np.array2string(adaptive_matrix)}\n")


def plot_posteriors(filename, save_filename=None, plot_prior=True, bw_adjust=1):
    """ Plot Posterior Plots in a nice format. """

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
        )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)
    if all_model_samples.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    # We only care about the samples accepted by the final model
    full_data = np.vstack(all_model_samples)[:, -1]  

    upper = np.percentile(full_data, 95, axis=0)
    lower = np.percentile(full_data, 5, axis=0)

    fig = plt.figure(figsize=(20, 8))

    for i in range(full_data.shape[1]):  # Number of parameters
        ax1 = plt.subplot2grid((2, 4), (i // 4, i % 4))

        for j in range(all_model_samples.shape[0]):  # Number of chains
            sns.kdeplot(
                all_model_samples[j][:, -1][:, i],
                alpha=0.5,
                bw_adjust=bw_adjust,
                ax=ax1,
                )
        axes = sns.kdeplot(
            full_data[:, i],
            color="r",
            linewidth=3,
            label="KDE Plot Over All Chains",
            bw_adjust=bw_adjust,
            ax=ax1,
            )

        x, y = axes.lines[-1].get_data()

        ax1.axvline(
            x=lower[i], color="k", label="95% Confidence Interval", linewidth=0.5
            )
        ax1.axvline(
            x=wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Value",
            )
        ax1.axvline(x=upper[i], color="k", linewidth=0.5)
        ax1.set_title(
            PARAMETER_LABELS[i] + f" MAP: {round(x[y == np.max(y)][0], 3)}", fontsize=22
            )

        # Plot prior
        if plot_prior:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            if i < full_data.shape[1] - 1:
                ax1.plot(
                    domain,
                    norm.pdf(
                        domain,
                        loc=wave_specs.PRIOR_MEAN[i],
                        scale=wave_specs.PRIOR_COV_DIAG[i],
                        ),
                    "--",
                    color="k",
                    label="Prior",
                    )
            else:
                ax1.plot(
                    domain,
                    beta.pdf(domain, a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B),
                    "--",
                    color="k",
                    label="Prior",
                    )

        # Only label the outer-most subplots
        if i % 4 != 0:
            ax1.set_ylabel(None)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.87, 0.23),
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_trace_together(
    filename, save_filename=None, plot_prior=True, bw_adjust=1
):
    """Plot Posterior and Trace Plots all together in a nice format"""

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
        )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)
    if all_model_samples.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    # We only care about the samples accepted by the final model
    full_data = np.vstack(all_model_samples)[:, -1]  

    upper = np.percentile(full_data, 95, axis=0)
    lower = np.percentile(full_data, 5, axis=0)

    fig = plt.figure(figsize=(18, 18))

    for i in range(full_data.shape[1]):
        ax1 = plt.subplot2grid((4, 4), (i // 2, 2 * (i % 2)))
        ax2 = plt.subplot2grid((4, 4), (i // 2, 2 * (i % 2) + 1))

        for j in range(all_model_samples.shape[0]):  # Number of chains
            sns.kdeplot(
                all_model_samples[j][:, -1][:, i],
                alpha=0.5,
                bw_adjust=bw_adjust,
                ax=ax1,
                )
        axes = sns.kdeplot(
            full_data[:, i],
            color="r",
            linewidth=3,
            label="KDE Plot Over All Chains",
            bw_adjust=bw_adjust,
            ax=ax1,
            )

        x, y = axes.lines[-1].get_data()

        ax1.axvline(
            x=lower[i], color="k", label="95% Confidence Interval", linewidth=0.5
            )
        ax1.axvline(
            x=wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Value",
            )
        ax1.axvline(x=upper[i], color="k", linewidth=0.5)
        ax1.set_title(
            PARAMETER_LABELS[i] + f" MAP: {round(x[y == np.max(y)][0], 3)}", fontsize=22
            )

        # Plot prior
        if plot_prior:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            if i < full_data.shape[1] - 1:
                ax1.plot(
                    domain,
                    norm.pdf(
                        domain,
                        loc=wave_specs.PRIOR_MEAN[i],
                        scale=wave_specs.PRIOR_COV_DIAG[i],
                        ),
                    "--",
                    color="k",
                    label="Prior",
                    )
            else:
                ax1.plot(
                    domain,
                    beta.pdf(domain, a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B),
                    "--",
                    color="k",
                    label="Prior",
                    )

        for j in range(all_model_samples.shape[0]):
            ax2.plot(all_model_samples[j][:, -1][:, i], alpha=0.5)
        ax2.plot(
            np.ones(all_model_samples[j].shape[0]) * wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            )
        ax2.set_title(PARAMETER_LABELS[i] + rf" Trace Plot", fontsize=22)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.12),
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_predictives(
    filename, save_filename=None, plot_likelihood=True, bw_adjust=1
):
    """ Plot the Posterior Predictive Plots of all observations. """

    _, all_model_obs, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(all_model_obs.shape) <= 3:
        all_model_obs = np.expand_dims(all_model_obs, axis=0)
    if all_model_obs.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_obs = all_model_obs[:, 1:]  

    # We only care about the observations generated by the final model
    full_obs = np.vstack(all_model_obs)[:, -1]  
    obs_len = wave_specs.OBS_VALS.shape[0]

    upper = np.percentile(full_obs, 95, axis=0)
    lower = np.percentile(full_obs, 5, axis=0)

    fig = plt.figure(figsize=(20, 8))

    for i in range(obs_len):
        ax1 = plt.subplot2grid((2, obs_len), (0, i))
        ax2 = plt.subplot2grid((2, obs_len), (1, i))

        # Plot histogram or kde plot for each chain
        for j in range(all_model_obs.shape[0]):
            sns.kdeplot(
                all_model_obs[j][:, -1][:, i], alpha=0.5, bw_adjust=bw_adjust, ax=ax1
                )

        # Plot kde for all data
        axs = sns.kdeplot(
            full_obs[:, i],
            color="r",
            linewidth=3,
            label="KDE Plot Over All Chains",
            bw_adjust=bw_adjust,
            ax=ax1,
            )
        x, y = axs.lines[-1].get_data()

        ax1.axvline(
            x=lower[i], color="k", label="95% Confidence Interval", linewidth=0.5
            )
        ax1.axvline(
            x=wave_specs.OBS_VALS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax1.axvline(x=upper[i], color="k", linewidth=0.5)
        ax1.set_title(
            OBSERVATION_LABELS[i] + f" MAP: {round(x[y == np.max(y)][0], 3)}",
            fontsize=22,
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax1.plot(
                domain,
                norm.pdf(
                    domain, loc=wave_specs.OBS_VALS[i], scale=wave_specs.LL_COV_DIAG[i]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        # Plot histogram or kde plot for each chain
        for j in range(all_model_obs.shape[0]):
            sns.kdeplot(
                all_model_obs[j][:, -1][:, i + obs_len],
                alpha=0.5,
                bw_adjust=bw_adjust,
                ax=ax2,
                )

        # Plot kde for all data
        axs = sns.kdeplot(
            full_obs[:, i + obs_len],
            color="r",
            linewidth=3,
            label="KDE Plot Over All Chains",
            bw_adjust=bw_adjust,
            ax=ax2,
            warn_singular=False,
            )

        if not np.isclose(np.var(full_obs[:, i + obs_len]), 0.0):
            x, y = axs.lines[-1].get_data()
        else:
            x = full_obs[:, i + obs_len]
            y = full_obs[:, i + obs_len]

        obs_time_secs = (
            (wave_specs.OBS_TIMES[i] + 1) / wave_specs.TIMESTEPS * wave_specs.T
            )

        ax2.axvline(
            x=lower[i + obs_len],
            color="k",
            label="95% Confidence Interval",
            linewidth=0.5,
            )
        ax2.axvline(
            x=obs_time_secs,
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax2.axvline(x=upper[i + obs_len], color="k", linewidth=0.5)
        ax2.set_title(
            OBSERVATION_LABELS[i + obs_len] + f" MAP: {round(x[y == np.max(y)][0], 3)}",
            fontsize=22,
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax2.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax2.plot(
                domain,
                norm.pdf(
                    domain, loc=obs_time_secs, scale=wave_specs.LL_COV_DIAG[i + obs_len]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        # Only label the outer-most subplots
        if i != 0:
            ax1.set_ylabel(None)
            ax2.set_ylabel(None)

    # Get labels and handles from the plots
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_single_posterior(
    filename,
    param_index,
    save_filename=None,
    plot_prior=True,
    bw_adjust=1,
    plot_traces=True,
):
    """Plot individual Posterior Plot for specified parameter (by index)"""

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    full_data = np.vstack(all_model_samples)[:, -1]

    upper = np.percentile(full_data, 95, axis=0)
    lower = np.percentile(full_data, 5, axis=0)

    if plot_traces:
        fig = plt.figure(figsize=(12, 7))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
    else:
        fig = plt.figure(figsize=(5, 5))
        ax1 = plt.subplot(111)

    for j in range(all_model_samples.shape[0]):
        sns.kdeplot(
            all_model_samples[j][:, -1][:, param_index],
            alpha=0.5,
            bw_adjust=bw_adjust,
            ax=ax1,
            )

    axes = sns.kdeplot(
        full_data[:, param_index],
        color="r",
        linewidth=3,
        label="KDE Plot Over All Chains",
        bw_adjust=bw_adjust,
        ax=ax1,
        )
    x, y = axes.lines[-1].get_data()

    ax1.axvline(
        x=lower[param_index], color="k", label="95% Confidence Interval", linewidth=0.5
        )
    ax1.axvline(
        x=wave_specs.TRUE_PARAMETERS[param_index],
        linestyle=":",
        color="purple",
        linewidth=2,
        label="True Value",
        )
    ax1.axvline(x=upper[param_index], color="k", linewidth=0.5)
    ax1.set_title(
        PARAMETER_LABELS[param_index] + f" MAP is {round(x[y == np.max(y)][0], 3)}",
        fontsize=22,
        )

    # Plot prior
    if plot_prior:
        x_min, x_max = ax1.get_xlim()
        domain = np.linspace(x_min, x_max, 300)
        if param_index < full_data.shape[1] - 1:
            ax1.plot(
                domain,
                norm.pdf(
                    domain,
                    loc=wave_specs.PRIOR_MEAN[param_index],
                    scale=wave_specs.PRIOR_COV_DIAG[param_index],
                    ),
                "--",
                color="k",
                label="Prior",
                )
        else:
            ax1.plot(
                domain,
                beta.pdf(domain, a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B),
                "--",
                color="k",
                label="Prior",
                )

    if plot_traces:
        for j in range(all_model_samples.shape[0]):
            ax2.plot(all_model_samples[j][:, -1][:, param_index], alpha=0.5)
        ax2.plot(
            np.ones(all_model_samples[j].shape[0])
            * wave_specs.TRUE_PARAMETERS[param_index],
            linestyle=":",
            color="purple",
            linewidth=2,
            )
        ax2.set_title(PARAMETER_LABELS[param_index] + rf" Trace Plot", fontsize=22)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_single_posterior_predictive(
    filename, obs_index, save_filename=None, plot_likelihood=True, bw_adjust=1
):
    """ Plot the Posterior Predictive Plots of specified observation (by index). """

    _, all_model_obs, _, _, _, _, _, _, _, _, _, _, _, _, _ = save_functions.load_data(
        filename
    )

    # For single chains, we need to add a dimension
    if len(all_model_obs.shape) <= 3:
        all_model_obs = np.expand_dims(all_model_obs, axis=0)

    # Remove data gathered from initial sample
    all_model_obs = all_model_obs[:, 1:]  

    full_obs = np.vstack(all_model_obs)[:, -1]
    obs_len = wave_specs.OBS_VALS.shape[0]

    upper = np.percentile(full_obs, 95, axis=0)
    lower = np.percentile(full_obs, 5, axis=0)

    fig = plt.figure(figsize=(12, 7))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    # Plot histogram or kde plot for each chain
    for j in range(all_model_obs.shape[0]):
        sns.kdeplot(
            all_model_obs[j][:, -1][:, obs_index],
            alpha=0.5,
            bw_adjust=bw_adjust,
            ax=ax1,
            )

    # Plot kde for all data
    axs = sns.kdeplot(
        full_obs[:, obs_index],
        color="r",
        linewidth=3,
        label="KDE Plot",
        bw_adjust=bw_adjust,
        ax=ax1,
        )
    x, y = axs.lines[-1].get_data()

    ax1.axvline(
        x=lower[obs_index], color="k", label="95% Confidence Interval", linewidth=0.5
        )
    ax1.axvline(
        x=wave_specs.OBS_VALS[obs_index],
        linestyle=":",
        color="purple",
        linewidth=2,
        label="True Obs",
        )
    ax1.axvline(x=upper[obs_index], color="k", linewidth=0.5)
    ax1.set_title(
        OBSERVATION_LABELS[obs_index] + " MAP:" + f"\n{round(x[y == np.max(y)][0], 3)}",
        fontsize=22,
        )

    # Plot likelihood
    if plot_likelihood:
        x_min, x_max = ax1.get_xlim()
        domain = np.linspace(x_min, x_max, 300)
        ax1.plot(
            domain,
            norm.pdf(
                domain,
                loc=wave_specs.OBS_VALS[obs_index],
                scale=wave_specs.LL_COV_DIAG[obs_index],
                ),
            "--",
            color="k",
            label="Likelihood",
            )

    # Plot histogram or kde plot for each chain
    for j in range(all_model_obs.shape[0]):
        sns.kdeplot(
            all_model_obs[j][:, -1][:, obs_index + obs_len],
            alpha=0.5,
            bw_adjust=bw_adjust,
            ax=ax2,
            )

    # Plot kde for all data
    axs = sns.kdeplot(
        full_obs[:, obs_index + obs_len],
        color="r",
        linewidth=3,
        label="KDE Plot",
        bw_adjust=bw_adjust,
        ax=ax2,
        warn_singular=False,
        )

    if not np.isclose(np.var(full_obs[:, obs_index + obs_len]), 0.0):
        x, y = axs.lines[-1].get_data()
    else:
        x = full_obs[:, obs_index + obs_len]
        y = full_obs[:, obs_index + obs_len]

    obs_time_secs = (
        (wave_specs.OBS_TIMES[obs_index] + 1) / wave_specs.TIMESTEPS * wave_specs.T
        )

    ax2.axvline(
        x=lower[obs_index + obs_len],
        color="k",
        label="95% Confidence Interval",
        linewidth=0.5,
        )
    ax2.axvline(
        x=obs_time_secs, linestyle=":", color="purple", linewidth=2, label="True Obs"
        )
    ax2.axvline(x=upper[obs_index + obs_len], color="k", linewidth=0.5)
    ax2.set_title(
        OBSERVATION_LABELS[obs_index + obs_len]
        + " MAP:"
        + f"\n{round(x[y == np.max(y)][0], 3)}",
        fontsize=22,
        )

    # Plot likelihood
    if plot_likelihood:
        x_min, x_max = ax2.get_xlim()
        domain = np.linspace(x_min, x_max, 300)
        ax2.plot(
            domain,
            norm.pdf(
                domain,
                loc=obs_time_secs,
                scale=wave_specs.LL_COV_DIAG[obs_index + obs_len],
                ),
            "--",
            color="k",
            label="Likelihood",
            )

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_parameter_heatmap(
    filename, save_filename=None, plot_diag_posts=True, gridsize=50, bw_adjust=1
):
    """ Create a cool posterior marginal density heatmap grid. """

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    full_data = np.vstack(all_model_samples)[:, -1]
    num_params = full_data.shape[1]

    fig, axes = plt.subplots(nrows=num_params, ncols=num_params, figsize=(15, 15))

    for i in range(num_params):
        for j in range(num_params):
            if i == num_params - j - 1 and plot_diag_posts:
                for chain in range(all_model_samples.shape[0]):
                    sns.kdeplot(
                        all_model_samples[chain][:, num_params - j - 1],
                        alpha=0.5,
                        bw_adjust=bw_adjust,
                        ax=axes[i, j],
                        )
                sns.kdeplot(
                    full_data[:, num_params - j - 1],
                    color="r",
                    linewidth=3,
                    bw_adjust=bw_adjust,
                    ax=axes[i, j],
                    )
            else:
                # i and j are "switched" to align ticks along rows and columns
                axes[i, j].hexbin(
                    full_data[:, num_params - j - 1], 
                    full_data[:, i], 
                    gridsize=gridsize
                    )

            # Remove tick labels on all inner subplots of grid
            axes[i, j].label_outer()  

    for ax, col in zip(axes[-1], PARAMETER_LABELS[::-1]):
        ax.set_xlabel(col, size="large", labelpad=5)

    for ax, row in zip(axes[:, 0], np.arange(len(PARAMETER_LABELS))):
        ax.set_ylabel(PARAMETER_LABELS[row], rotation=0, size="large", labelpad=15)

    plt.suptitle("Marginal Densities of Parameters", size="large", y=1)
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_acceptance_history(filename, save_filename=None):
    """ Plots the acceptance rate history of the sampling. """

    _, _, _, _, _, _, _, _, acceptance_history, _, _, _, _, _, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(acceptance_history.shape) == 1:
        acceptance_history = np.expand_dims(acceptance_history, axis=0)

    for chain_acceptance_history in acceptance_history:
        plt.plot(chain_acceptance_history)

    plt.title("Sample Acceptance History")
    plt.xlabel("Samples Drawn")
    plt.ylabel("Acceptance Ratio")

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_sequential_resampling(
    filename, save_filename=None, n_sir_stages=5, samples_per_sir_stage=1000
):
    """Plot cool plot showing sequential resampling results"""

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    # We only care about the samples accepted by the final model
    full_data = np.vstack(all_model_samples)[:, -1]  

    fig = plt.figure(figsize=(15, 10))

    for i in range(full_data.shape[1]):
        ax1 = plt.subplot2grid((4, 2), (i // 2, i % 2))

        for j in range(all_model_samples.shape[0]):
            ax1.plot(all_model_samples[j][:, -1][:, i], alpha=0.5)
        ax1.plot(
            np.ones(all_model_samples[j].shape[0]) * wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Value",
            )
        ax1.set_ylabel(
            PARAMETER_LABELS[i], fontsize=22, rotation="horizontal", labelpad=20
            )

        for j in range(n_sir_stages):
            ax1.axvline(
                x=(j + 1) * samples_per_sir_stage,
                linestyle="--",
                color="red",
                label="Resampling Points",
                )

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.12),
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_log_probs(filename, save_filename=None, report_stats=None):
    """ Plots the log probability of each sample drawn, with the option of 
    reporting some statistics.
    """

    _, _, all_model_log_probs, _, _, _, _, _, _, _, _, _, _, _, _ = (
        save_functions.load_data(filename)
    )

    grid_size = 400
    wave_sim = wave_sampling.Wave_Sampling(
        wave_specs.TRUE_PARAMETERS, None, None, None, [grid_size], None, None
    )
    true_log_prob = wave_sim.all_model_log_probs[0][0]

    # For single chains, we need to add a dimension
    if len(all_model_log_probs.shape) <= 2:
        all_model_log_probs = np.expand_dims(all_model_log_probs, axis=0)
        num_chains = 1
    else:
        num_chains = len(all_model_log_probs)

    # Remove data gathered from initial sample
    all_model_log_probs = all_model_log_probs[:, 1:]  

    fig, axes = plt.subplots(nrows=1, ncols=num_chains, sharey=True, figsize=(20, 5))

    if num_chains == 1:
        axes = [axes]

    if report_stats:
        below_percentage = np.zeros(
            (num_chains, all_model_log_probs.shape[2])
        )  # num_chains x num_models
        chain_above_avgs = np.zeros(num_chains)
        chain_below_avgs = np.zeros(num_chains)

    for i in range(num_chains):
        labels = (
            [f"Low-Fidelity Log Probs"]
            + [
                f"Mid-Fidelity {k + 1} Log Probs"
                for k in range(1, all_model_log_probs[i].shape[1] - 1)
            ]
            + [f"High-Fidelity Log Probs"]
            )

        if report_stats:
            temp_above_avgs = []
            temp_below_avgs = []

        for j in range(all_model_log_probs[i].shape[1]):  # num_models
            # If there is only one forward model, we don't need to distinguish 
            # between them
            if all_model_log_probs[i].shape[1] == 1:
                labels = ["Log Probs"]

            if report_stats:
                below_percentage[i][j] = (
                    np.sum(all_model_log_probs[i][:, j] < report_stats)
                    / all_model_log_probs[i][:, j].shape[0]
                    )
                temp_above_avgs += list(
                    all_model_log_probs[i][:, j][
                        all_model_log_probs[i][:, j] >= report_stats
                    ])
                temp_below_avgs += list(
                    all_model_log_probs[i][:, j][
                        all_model_log_probs[i][:, j] < report_stats
                    ])

            axes[i].plot(
                all_model_log_probs[i][:, j],
                "o",
                markersize=0.5,
                alpha=0.3,
                color=MODEL_COLORS[j],
                label=labels[j],
                )
            
            # Remove tick labels on all inner subplots of grid
            axes[i].label_outer()  

        domain = np.arange(all_model_log_probs[i][:, j].shape[0])
        axes[i].plot(
            domain,
            true_log_prob * np.ones_like(domain),
            color="k",
            label="True Log Prob",
            )

        if report_stats:
            axes[i].plot(
                domain,
                report_stats * np.ones_like(domain),
                "--",
                color="k",
                label="Log Prob Threshold",
                )
            chain_above_avgs[i] = np.mean(temp_above_avgs)
            chain_below_avgs[i] = np.mean(temp_below_avgs)

        axes[i].set_title(f"Chain {i + 1}", fontsize=22)

    plt.tight_layout()
    fig.supxlabel("Samples Drawn", y=0.24)
    fig.supylabel("Log Probability", x=-0.005)
    fig.subplots_adjust(bottom=0.35)

    leg = fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        markerscale=20,
        )
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    if report_stats:
        print(f"Percentage of Log-Probabilities Below {report_stats} for Each Model:")
        print(below_percentage)
        print(f"\nMean Log-Probabilities Above {report_stats} for Each Chain:")
        print(chain_above_avgs)
        print(f"\nMean Log-Probabilities Below {report_stats} for Each Chain:")
        print(chain_below_avgs)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_model_comparison(
    filename, save_filename=None, plot_prior=True, bw_adjust=1
):
    """ Plot all samples accepted by each model seperately on the same plots. """

    all_model_samples, _, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
    )

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)
    if all_model_samples.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  

    full_model_samples = np.vstack(all_model_samples)

    fig = plt.figure(figsize=(20, 8))

    for i in range(full_model_samples.shape[2]):  # Sample dimension
        ax1 = plt.subplot2grid((2, 4), (i // 4, i % 4))

        labels = (
            [f"Low-Fidelity Model"]
            + [
                f"Mid-Fidelity {k + 1} Model"
                for k in range(1, full_model_samples[i].shape[0] - 1)
            ]
            + [f"High-Fidelity Model"]
            )

        # Plot kde for all sample parameters
        for j in range(full_model_samples.shape[1]):  # Number of models
            sns.kdeplot(
                full_model_samples[:, j][:, i],
                color=MODEL_COLORS[j],
                linewidth=3,
                label=labels[j],
                bw_adjust=bw_adjust,
                ax=ax1,
                )

        # Plot prior
        if plot_prior:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            if i < full_model_samples.shape[2] - 1:
                ax1.plot(
                    domain,
                    norm.pdf(
                        domain,
                        loc=wave_specs.PRIOR_MEAN[i],
                        scale=wave_specs.PRIOR_COV_DIAG[i],
                        ),
                    "--",
                    color="k",
                    label="Prior",
                    )
            else:
                ax1.plot(
                    domain,
                    beta.pdf(domain, a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B),
                    "--",
                    color="k",
                    label="Prior",
                    )

        ax1.axvline(
            x=wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Value",
            )
        ax1.set_title(PARAMETER_LABELS[i], fontsize=22)

        # Only label the outer-most subplots
        if i % 4 != 0:
            ax1.set_ylabel(None)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.87, 0.23),
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_predictive_model_comparison(
    filename, save_filename=None, plot_likelihood=True, bw_adjust=1
):
    """ Plot the Posterior Predictive Plots of each model seperately on the 
    same plots.
    """

    _, all_model_obs, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
    )

    obs_len = wave_specs.OBS_VALS.shape[0]

    # For single chains, we need to add a dimension
    if len(all_model_obs.shape) <= 3:
        all_model_obs = np.expand_dims(all_model_obs, axis=0)
    if all_model_obs.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_obs = all_model_obs[:, 1:]  

    full_model_obs = np.vstack(all_model_obs)

    fig = plt.figure(figsize=(20, 8))

    for i in range(obs_len):
        ax1 = plt.subplot2grid((2, obs_len), (0, i))
        ax2 = plt.subplot2grid((2, obs_len), (1, i))

        labels = (
            [f"Low-Fidelity Model"]
            + [
                f"Mid-Fidelity {k + 1} Model"
                for k in range(1, full_model_obs[i].shape[0] - 1)
            ]
            + [f"High-Fidelity Model"]
            )

        # Plot kde for all wave obserations
        for j in range(full_model_obs.shape[1]):
            sns.kdeplot(
                full_model_obs[:, j][:, i],
                color=MODEL_COLORS[j],
                linewidth=3,
                label=labels[j],
                bw_adjust=bw_adjust,
                ax=ax1,
                )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax1.plot(
                domain,
                norm.pdf(
                    domain, loc=wave_specs.OBS_VALS[i], scale=wave_specs.LL_COV_DIAG[i]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax1.axvline(
            x=wave_specs.OBS_VALS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax1.set_title(OBSERVATION_LABELS[i], fontsize=22)

        # Plot kde for all time observations
        for j in range(full_model_obs.shape[1]):
            sns.kdeplot(
                full_model_obs[:, j][:, i + obs_len],
                color=MODEL_COLORS[j],
                linewidth=3,
                label=labels[j],
                bw_adjust=bw_adjust,
                ax=ax2,
                warn_singular=False,
                )

        obs_time_secs = (
            (wave_specs.OBS_TIMES[i] + 1) / wave_specs.TIMESTEPS * wave_specs.T
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax2.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax2.plot(
                domain,
                norm.pdf(
                    domain, loc=obs_time_secs, scale=wave_specs.LL_COV_DIAG[i + obs_len]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax2.axvline(
            x=obs_time_secs,
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax2.set_title(OBSERVATION_LABELS[i + obs_len], fontsize=22)

        # Only label the outer-most subplots
        if i != 0:
            ax1.set_ylabel(None)
            ax2.set_ylabel(None)

    # Create legend using the extracted labels and handles
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_log_prob_comparison(
    filename, log_prob_threshold, save_filename=None, plot_prior=True, bw_adjust=1
):
    """ Plot all samples as two different distributions based on their 
    corresponding log probabilities.
    """

    (
        all_model_samples,
        _,
        all_model_log_probs,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        epsilon,
        _,
    ) = save_functions.load_data(filename)

    # For single chains, we need to add a dimension
    if len(all_model_samples.shape) <= 3:
        all_model_samples = np.expand_dims(all_model_samples, axis=0)
    if all_model_samples.shape[0] == 1:
        all_model_log_probs = np.expand_dims(all_model_log_probs, axis=0)
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_samples = all_model_samples[:, 1:]  
    all_model_log_probs = all_model_log_probs[:, 1:]

    full_model_samples = np.vstack(all_model_samples)
    full_model_log_probs = np.vstack(all_model_log_probs)

    samples_higher = []
    samples_lower = []
    for i in range(full_model_samples.shape[0]):  # num_samples
        for j in range(full_model_samples.shape[1]):  # num_models
            if full_model_log_probs[i][j] > log_prob_threshold:
                samples_higher.append(full_model_samples[i][j])
            else:
                samples_lower.append(full_model_samples[i][j])
    samples_higher = np.array(samples_higher)
    samples_lower = np.array(samples_lower)

    if samples_lower.shape[0] == 0:
        return "There is nothing to compare!"

    fig = plt.figure(figsize=(20, 8))

    for i in range(full_model_samples.shape[2]):  # Sample dimension
        ax1 = plt.subplot2grid((2, 4), (i // 4, i % 4))

        # Plot kde for all sample parameters
        sns.kdeplot(
            samples_lower[:, i],
            color=MODEL_COLORS[0],
            linewidth=3,
            label=rf"Log Prob $\leq$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax1,
            )
        sns.kdeplot(
            samples_higher[:, i],
            color=MODEL_COLORS[1],
            linewidth=3,
            label=rf"Log Prob $>$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax1,
            )

        # Plot prior
        if plot_prior:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            if i < full_model_samples.shape[2] - 1:
                ax1.plot(
                    domain,
                    norm.pdf(
                        domain,
                        loc=wave_specs.PRIOR_MEAN[i],
                        scale=wave_specs.PRIOR_COV_DIAG[i],
                        ),
                    "--",
                    color="k",
                    label="Prior",
                    )
            else:
                ax1.plot(
                    domain,
                    beta.pdf(domain, a=wave_specs.PRIOR_A, b=wave_specs.PRIOR_B),
                    "--",
                    color="k",
                    label="Prior",
                    )

        ax1.axvline(
            x=wave_specs.TRUE_PARAMETERS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Value",
            )
        ax1.set_title(PARAMETER_LABELS[i], fontsize=22)

        # Only label the outer-most subplots
        if i % 4 != 0:
            ax1.set_ylabel(None)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.87, 0.23),
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_posterior_predictive_log_prob_comparison(
    filename, log_prob_threshold, save_filename=None, plot_likelihood=True, bw_adjust=1
):
    """ Plot all observations as two different distributions based on their 
    corresponding log probabilities.
    """

    _, all_model_obs, all_model_log_probs, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
    )

    obs_len = wave_specs.OBS_VALS.shape[0]

    # For single chains, we need to add a dimension
    if len(all_model_obs.shape) <= 3:
        all_model_obs = np.expand_dims(all_model_obs, axis=0)
    if all_model_obs.shape[0] == 1:
        all_model_log_probs = np.expand_dims(all_model_log_probs, axis=0)
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_obs = all_model_obs[:, 1:]  
    all_model_log_probs = all_model_log_probs[:, 1:]

    full_model_obs = np.vstack(all_model_obs)
    full_model_log_probs = np.vstack(all_model_log_probs)

    obs_higher = []
    obs_lower = []
    for i in range(full_model_obs.shape[0]):  # num_samples
        for j in range(full_model_obs.shape[1]):  # num_models
            if full_model_log_probs[i][j] > log_prob_threshold:
                obs_higher.append(full_model_obs[i][j])
            else:
                obs_lower.append(full_model_obs[i][j])
    obs_higher = np.array(obs_higher)
    obs_lower = np.array(obs_lower)

    if obs_lower.shape[0] == 0:
        return "There is nothing to compare!"

    fig = plt.figure(figsize=(20, 8))

    for i in range(obs_len):
        ax1 = plt.subplot2grid((2, obs_len), (0, i))
        ax2 = plt.subplot2grid((2, obs_len), (1, i))

        # Plot kde for all wave obserations
        sns.kdeplot(
            obs_lower[:, i],
            color=MODEL_COLORS[0],
            linewidth=3,
            label=rf"Log Prob $\leq$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax1,
            )
        sns.kdeplot(
            obs_higher[:, i],
            color=MODEL_COLORS[1],
            linewidth=3,
            label=rf"Log Prob $>$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax1,
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax1.plot(
                domain,
                norm.pdf(
                    domain, loc=wave_specs.OBS_VALS[i], scale=wave_specs.LL_COV_DIAG[i]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax1.axvline(
            x=wave_specs.OBS_VALS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax1.set_title(OBSERVATION_LABELS[i], fontsize=22)

        # Plot kde for all time obserations
        sns.kdeplot(
            obs_lower[:, i + obs_len],
            color=MODEL_COLORS[0],
            linewidth=3,
            label=rf"Log Prob $\leq$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax2,
            warn_singular=False,
            )
        sns.kdeplot(
            obs_higher[:, i + obs_len],
            color=MODEL_COLORS[1],
            linewidth=3,
            label=rf"Log Prob $>$ {log_prob_threshold}",
            bw_adjust=bw_adjust,
            ax=ax2,
            warn_singular=False,
            )

        obs_time_secs = (
            (wave_specs.OBS_TIMES[i] + 1) / wave_specs.TIMESTEPS * wave_specs.T
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax2.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax2.plot(
                domain,
                norm.pdf(
                    domain, loc=obs_time_secs, scale=wave_specs.LL_COV_DIAG[i + obs_len]
                ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax2.axvline(
            x=obs_time_secs,
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )
        ax2.set_title(OBSERVATION_LABELS[i + obs_len], fontsize=22)

        # Only label the outer-most subplots
        if i != 0:
            ax1.set_ylabel(None)
            ax2.set_ylabel(None)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()


def plot_low_fidelity_chain_comparison(
    filename, save_filename, good_chains, bad_chains, plot_likelihood=True, bw_adjust=1
):
    """Plot the low-fidelity Posterior Predictive Plots of each chain seperately 
    on the same plots, coloring them by groups of good_chains and bad_chains.
    """

    _, all_model_obs, _, _, _, _, _, _, _, _, _, _, _, epsilon, _ = (
        save_functions.load_data(filename)
    )

    obs_len = wave_specs.OBS_VALS.shape[0]

    # For single chains, we need to add a dimension
    if len(all_model_obs.shape) <= 3:
        all_model_obs = np.expand_dims(all_model_obs, axis=0)
    if all_model_obs.shape[0] == 1:
        epsilon = np.array([epsilon])

    # Remove data gathered from initial sample
    all_model_obs = all_model_obs[:, 1:]  

    fig = plt.figure(figsize=(20, 8))

    for i in range(obs_len):
        ax1 = plt.subplot2grid((2, obs_len), (0, i))
        ax2 = plt.subplot2grid((2, obs_len), (1, i))

        # Plot kde for all wave obserations
        for j in range(all_model_obs.shape[0]):
            if j + 1 in bad_chains:
                label = f" Chains {np.array2string(np.array(bad_chains), separator=', ')[1:-1]}"
                color = MODEL_COLORS[0]  # Blue
            else:
                label = f" Chains {np.array2string(np.array(good_chains), separator=', ')[1:-1]}"
                color = MODEL_COLORS[1]  # Red

            sns.kdeplot(
                all_model_obs[j][:, 0][:, i],
                color=color,
                linewidth=3,
                label=label,
                bw_adjust=bw_adjust,
                ax=ax1,
                )

        ax1.axvline(
            x=wave_specs.OBS_VALS[i],
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax1.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax1.plot(
                domain,
                norm.pdf(
                    domain, loc=wave_specs.OBS_VALS[i], scale=wave_specs.LL_COV_DIAG[i]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax1.set_title(OBSERVATION_LABELS[i], fontsize=22)

        # Plot kde for all time observations
        for j in range(all_model_obs.shape[0]):
            if j + 1 in bad_chains:
                label = f" Chains {np.array2string(np.array(bad_chains), separator=', ')[1:-1]}"
                color = MODEL_COLORS[0]  # Blue
            else:
                label = f" Chains {np.array2string(np.array(good_chains), separator=', ')[1:-1]}"
                color = MODEL_COLORS[1]  # Red

            sns.kdeplot(
                all_model_obs[j][:, 0][:, i + obs_len],
                color=color,
                linewidth=3,
                label=label,
                bw_adjust=bw_adjust,
                ax=ax2,
                warn_singular=False,
                )

        obs_time_secs = (
            (wave_specs.OBS_TIMES[i] + 1) / wave_specs.TIMESTEPS * wave_specs.T
            )

        ax2.axvline(
            x=obs_time_secs,
            linestyle=":",
            color="purple",
            linewidth=2,
            label="True Observation",
            )

        # Plot likelihood
        if plot_likelihood:
            x_min, x_max = ax2.get_xlim()
            domain = np.linspace(x_min, x_max, 300)
            ax2.plot(
                domain,
                norm.pdf(
                    domain, loc=obs_time_secs, scale=wave_specs.LL_COV_DIAG[i + obs_len]
                    ),
                "--",
                color="k",
                label="Likelihood",
                )

        ax2.set_title(OBSERVATION_LABELS[i + obs_len], fontsize=22)

        # Only label the outer-most subplots
        if i != 0:
            ax1.set_ylabel(None)
            ax2.set_ylabel(None)

    # Create legend using the extracted labels and handles
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="center",
        bbox_to_anchor=(0.75, 0.15),
        ncol=2,
        fontsize=20,
        framealpha=1.0,
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28, hspace=0.4)

    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight")
    plt.show()
