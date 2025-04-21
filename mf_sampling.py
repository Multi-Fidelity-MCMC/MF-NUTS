import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import time
from os.path import exists


class MF_Sampling:
    """ A class for multi-fidelity MCMC sampling. Methods are provided for multi-
    fidelity RW and multi-fidelity NUTS. Running multi-fidelity RW or NUTS with 
    only one forward model is equivalent to standard RW and standard NUTS,
    respectively.

    The following methods must be overwritten for this class to work:
        initialize_sample_obs_lls()
        log_prior()
        log_likelihood()
        derivative_log_density()    (for NUTS only)
        add_data()
        save_data()
    In addition, the attribute model_labels defaults to [0, ..., num_models - 1] 
    and may need to be overwritten to better reflect which forward models are 
    being used.

    The following data (with their size) is saved by this class:
        self.all_model_samples    = num_samples+1 x num_models x sample_dim
        self.all_model_obs        = num_samples+1 x num_models x num_obs
        self.all_model_log_probs  = num_samples+1 x num_models
        self.overall_time         = float
        self.overall_acc_ratio    = float
        self.all_model_times      = num_models
        self.model_eval_counter   = num_models
        self.model_acceptances    = num_models
        self.acceptance_history   = num_samples
        self.model_labels         = num_models
        self.current_sample_obs   = num_models x num_obs
        self.current_sample_lls   = num_models
        self.adaptive_matrix      = (differs between RW and NUTS)
        self.epsilon              = float
        self.average_tree_depth   = float
    """

    def __init__(
        self,
        current_sample,
        current_sample_obs,
        current_sample_lls,
        sample_mean,
        num_models,
        adaptive_matrix=None,
        epsilon=None,
    ):
        """ Initializes the MF_Sampling class.

        Parameters
        ----------
        current_sample (ndarray):
            The current state at which to begin sampling
        current_sample_obs (ndarray):
            The observations mapped from current_sample by each forward model; 
            if they are not provided (i.e. is None), they will be computed 
            automatically
        current_sample_lls (ndarray):
            The log likelihoods of current_sample computed with each forward 
            model; if they are not provided (i.e. is None), they will be 
            computed automatically
        sample_mean (ndarray):
            The sample mean of all drawn samples from the Markov chain, used for 
            matrix adaptation
        num_models (int):
            Indicates how many forward models will be used for multi-fidelity 
            sampling (must be at least 1)
        adaptive_matrix (ndarray):
            The adaptive matrix, corresponding to the step covariance in RW and 
            mass matrix in NUTS
        epsilon (float):
            The Leapfrog stepsize used in NUTS
        """

        self.current_sample = current_sample

        # Compute the observations and likelihood of the initial sample if not 
        # already provided
        if current_sample_obs is None:
            self.current_sample, self.current_sample_obs, self.current_sample_lls = (
                self.initialize_sample_obs_lls(current_sample, num_models)
                )
        else:
            self.current_sample_obs = current_sample_obs
            self.current_sample_lls = current_sample_lls

        self.sample_mean = sample_mean
        self.num_models = num_models  # Number of forward models used
        self.adaptive_matrix = adaptive_matrix

        # NUTS Specific attributes
        self.epsilon = epsilon
        self.max_tree_depth = 5  # Maximum depth of binary trees we'll allow
        self.delta_max = 1000
        self.averaged_epsilon = 1.0  # epsilon_bar in paper
        self.H_k = 0.0
        self.mu = None  # Initialized in NUTS
        self.gamma = 0.05
        self.t0 = 10.0
        self.kappa = 0.75

        self.sample_dim = len(self.current_sample)
        # This array helps label which model is which (should be overwritten)
        self.model_labels = np.arange(num_models)  
        self.current_log_prior = self.log_prior(self.current_sample)

        ### Global statistic trackers ###

        # These three statistics record the samples and corresponding obs/lls 
        # accepted by each model. If a model rejects a sample, it and all 
        # following models repeat the previous sample/obs/ll. As such, the final 
        # sample/obs/ll of each sample are the ones that constitute the posterior
        self.all_model_samples = [[
                self.current_sample 
                for _ in range(self.num_models)
                ]]
        self.all_model_obs = [self.current_sample_obs]
        self.all_model_log_probs = [[
                current_sample_ll + self.current_log_prior
                for current_sample_ll in self.current_sample_lls
                ]]

        self.overall_time = 0.0  # The total time spent running the algorithm
        self.overall_acc_ratio = 0.0  # The total acceptance ratio over all models

        self.all_model_times = np.zeros(self.num_models, dtype=float)
        self.model_eval_counter = np.zeros(
            self.num_models
            )  # The number of times each model is evaluated
        self.model_acceptances = np.zeros(
            self.num_models
            )  # Number of acceptances for each model

        self.total_accepted = 0  # Total values are for this entire run
        self.total_sample_tracker = 0
        self.acceptance_history = []  # Running history of acceptance ratio
        self.tree_depths = []  # Running history of NUTS tree depths
        self.average_tree_depth = (
            None  # The average value of binary tree depths used by NUTS
            )

        ### Current save statistic trackers ###

        # These values are for current stretch between data saves
        self.current_save_accepted = 0
        self.current_save_sample_tracker = 0


    def initialize_sample_obs_lls(self, initial_sample, num_models):
        """ Initialize the current state with its observations and likelihoods 
        generated by each forward model. This should also perform any parameter 
        transformations to map the initial state to sample space.
        """
        pass


    ######################################################################
    # GENERAL HELPER FUNCTIONS
    ######################################################################


    def log_prior(self, sample):
        """ Computes the log prior probability of the given sample.

        Parameters
        ----------
        sample (ndarray):
            A sample drawn from parameter space

        Returns
        -------
        log_prior (float):
            The log prior probability of the given sample
        """
        pass


    def compute_current_log_prior(self, current_sample):
        """ Called at the start of each iteration. """
        self.current_log_prior = self.log_prior(current_sample)


    def compute_new_log_prior(self, new_sample):
        """ Called at the start of each iteration. """
        self.new_log_prior = self.log_prior(new_sample)


    def log_likelihood(self, sample, model):
        """ Computes the log likelihood and forward model observations of the 
            given sample.

        Parameters
        ----------
        sample (ndarray):
            A sample drawn from parameter space
        model (int):
            Indicates which forward model to evaluate to compute the likelihood,
            and ranges in [0, self.num_models]

        Returns
        -------
        log_likelihood (float):
            The log likelihood of the given sample
        observations (ndarray):
            The observations returned by the forward model used to compute the 
            likelihood
        """
        pass


    ######################################################################
    # SAMPLER-SPECIFIC HELPER FUNCTIONS
    ######################################################################


    def initialize_covariance(self, matrix_adapt_start):
        """ Called just before RW sampling begins. """

        # If adaptive_matrix is None, then adaptive_matrix, sample_mean, and 
        # sample_cov all need to be initialized (if specified)
        if self.adaptive_matrix is None:
            self.adaptive_matrix = np.eye(
                self.sample_dim
                ) # Initialize as the identity
            if matrix_adapt_start != None:
                self.matrix_adapt_start = matrix_adapt_start
                self.sample_mean = self.current_sample.copy()
                self.sample_cov = np.zeros_like(self.adaptive_matrix, dtype=float)
        else:
            # If adaptive_matrix and sample_mean are already specified 
            # (i.e. we're continuing a chain), continue adapting adaptive_matrix 
            # from the first iteration (if specified)
            if matrix_adapt_start != None:
                self.matrix_adapt_start = 0
                self.sample_cov = self.adaptive_matrix.copy()


    def update_covariance(self, iteration):
        """ Called at the start of each RW iteration. """

        # Update the sample mean and sample covariance
        new_sample_mean = ((iteration + 1) * self.sample_mean 
                            + self.current_sample) / (iteration + 2)
        self.sample_cov = (
            iteration * self.sample_cov
            + np.outer(self.current_sample, self.current_sample)
            + (iteration + 1) * np.outer(self.sample_mean, self.sample_mean)
            - (iteration + 2) * np.outer(new_sample_mean, new_sample_mean)
            + 0.0001 * np.eye(self.sample_dim)
            ) / (iteration + 1)
        self.sample_mean = new_sample_mean

        # Update covariance
        if iteration + 1 >= self.matrix_adapt_start:
            self.adaptive_matrix = self.sample_cov


    def initialize_variance(self, matrix_adapt_start):
        """ Called just before NUTS sampling begins. """

        # If epsilon is None, then adaptive_matrix, sample_mean, and sample_var 
        # all need to be initialized (if specified)
        if self.epsilon is None:
            self.adaptive_matrix = np.ones(
                self.sample_dim, dtype=float
                )  # Initialize mass matrix as the identity
            if matrix_adapt_start != None:
                self.matrix_adapt_start = matrix_adapt_start
                self.sample_mean = self.current_sample.copy()
                self.sample_var = np.zeros_like(self.adaptive_matrix, dtype=float)
        else:
            # If epsilon, adaptive_matrix, and sample_mean are already specified 
            # (i.e. we're continuing a chain), continue adapting adaptive_matrix 
            # from the first iteration (if specified)
            if matrix_adapt_start != None:
                self.matrix_adapt_start = 0
                self.sample_var = (
                    1 / self.adaptive_matrix
                    )  # Note that M^{-1} = var(samples)


    def update_variance(self, iteration):
        """ Called at the start of each NUTS iteration. """

        # Update the sample mean and sample variance
        new_sample_mean = ((iteration + 1) * self.sample_mean 
                            + self.current_sample) / (iteration + 2)
        self.sample_var = (
            iteration * self.sample_var
            + self.current_sample**2
            + (iteration + 1) * self.sample_mean**2
            - (iteration + 2) * new_sample_mean**2
            + 0.0001 * np.ones(self.sample_dim, dtype=float)
            ) / (iteration + 1)
        self.sample_mean = new_sample_mean

        # Update mass matrix
        if iteration + 1 >= self.matrix_adapt_start:
            self.adaptive_matrix = 1 / self.sample_var


    def adapt_epsilon(self, iteration, delta, alpha, n_alpha):
        """ Called in each NUTS iteration. """

        # Update dual averaging
        self.H_k = (1 - 1 / ((iteration + 1) + self.t0)) * self.H_k + (
            1 / ((iteration + 1) + self.t0)
            ) * (delta - alpha / n_alpha)
        self.epsilon = np.exp(
            self.mu - (np.sqrt(iteration + 1) / self.gamma) * self.H_k
            )
        self.averaged_epsilon = np.exp(
            (iteration + 1) ** (-self.kappa) * np.log(self.epsilon)
            + (1 - (iteration + 1) ** (-self.kappa)) * np.log(self.averaged_epsilon)
            )


    def rw_acceptance_probability(self, current_sample_ll, new_sample_ll):
        return (
            new_sample_ll
            + self.new_log_prior
            - current_sample_ll
            - self.current_log_prior
            )


    def compute_current_hmc_log_posterior(self, current_p, current_sample_ll):
        self.current_hmc_log_posterior = (
            current_sample_ll
            + self.current_log_prior
            - np.sum(current_p**2 / self.adaptive_matrix) / 2
            )


    def compute_new_hmc_log_posterior(self, new_p, new_sample_ll):
        self.new_hmc_log_posterior = (
            new_sample_ll
            + self.new_log_prior
            - np.sum(new_p**2 / self.adaptive_matrix) / 2
            )


    def derivative_log_density(self, sample, model):
        """ Computes the derivative of the log density evaluated at the given 
        sample. Even if the derivative does not require the likelihood to be 
        evaluated, it should still be evaluated and returned by this function 
        since NUTS requires it for every sample integrated over by the Leapfrog 
        method.

        Parameters
        ----------
        sample (ndarray):
            A sample drawn from parameter space
        model (int):
            Indicates which forward model to use for the likelihood,
            and ranges in [0, self.num_models]

        Returns
        -------
        gradient (ndarray):
            The gradient of the log density at the given sample
        log_likelihood (float):
            The log likelihood of the given sample
        observations (ndarray):
            The observations returned by the forward model used to compute the 
            likelihood
        """
        pass


    def single_leapfrog(self, current_q, current_p, eps, gradient):
        """ Take a single leapfrog step for position q and momentum p. """

        # Make a half step for momentum at the beginning
        new_p = current_p + eps * gradient / 2

        # Take a whole step for position
        new_q = current_q + eps * new_p / self.adaptive_matrix

        # Make a half step for momentum at the end
        new_grad, new_ll, new_ob = self.derivative_log_density(new_q, 0)
        new_p += eps * new_grad / 2

        return new_q, new_p, new_ll, new_ob, new_grad


    def find_reasonable_epsilon(self, current_q, current_grad):
        """ Called at the beginning of NUTS when no epsilon is provided. """

        # Randomly pick momentum
        current_p = np.random.normal(0, self.adaptive_matrix)

        current_sample_ll = self.current_sample_lls[0]
        self.compute_current_log_prior(current_q)
        self.compute_current_hmc_log_posterior(current_p, current_sample_ll)

        self.epsilon = 1
        new_q, new_p, new_ll, _, _ = self.single_leapfrog(
            current_q, current_p, self.epsilon, current_grad
            )
        self.compute_new_log_prior(new_q)
        self.compute_new_hmc_log_posterior(new_p, new_ll)

        hmc_log_prob = self.new_hmc_log_posterior - self.current_hmc_log_posterior

        a = 1 if hmc_log_prob > np.log(0.5) else -1
        # Keep updating epsilon until the HMC Proposal crosses 1/2
        while a * hmc_log_prob > -a * np.log(2):  # We're dealing with logs
            self.epsilon *= 2**a
            new_q, new_p, new_ll, _, _ = self.single_leapfrog(
                current_q, current_p, self.epsilon, current_grad
                )
            self.compute_new_log_prior(new_q)

            self.compute_new_hmc_log_posterior(new_p, new_ll)
            hmc_log_prob = self.new_hmc_log_posterior - self.current_hmc_log_posterior


    def build_tree(self, q, p, u, dir, depth, gradient):
        """ Recursively builds the binary tree as outlined in the paper. Note, 
        in this code we return extra values for added efficiency: grad_minus, 
        grad_plus, grad_prime, p_prime, hmc_log_prob, ll_prime, and ob_prime.
        """

        if depth == 0:
            # Base case -- take one leapfrog step in direction dir
            q_prime, p_prime, ll_prime, ob_prime, grad_prime = self.single_leapfrog(
                q, p, dir * self.epsilon, gradient
                )

            # If u = 0, np.log(u) will throw a divide by zero error unless you 
            # use np.errstate()
            with np.errstate(divide="ignore"):  
                logu = np.log(u)

            self.compute_new_log_prior(q_prime)  # Saves as self.new_log_prior
            self.compute_new_hmc_log_posterior(
                p_prime, ll_prime
                )  # Saves as self.new_hmc_log_posterior

            n_prime = logu <= self.new_hmc_log_posterior
            s_prime = logu < (self.delta_max + self.new_hmc_log_posterior)

            hmc_log_prob = self.new_hmc_log_posterior - self.current_hmc_log_posterior

            return (
                q_prime,
                p_prime,
                grad_prime,
                q_prime,
                p_prime,
                grad_prime,
                q_prime,
                p_prime,
                grad_prime,
                hmc_log_prob,
                ll_prime,
                ob_prime,
                n_prime,
                s_prime,
                np.exp(min(0.0, hmc_log_prob)),
                1,
            )

        else:
            # Recursion -- implicitly build the left and right subtrees
            (
                q_minus,
                p_minus,
                grad_minus,
                q_plus,
                p_plus,
                grad_plus,
                q_prime,
                p_prime,
                grad_prime,
                hmc_log_prob_prime,
                ll_prime,
                ob_prime,
                n_prime,
                s_prime,
                alpha_prime,
                n_alpha_prime,
            ) = self.build_tree(q, p, u, dir, depth - 1, gradient)

            if s_prime == 1:
                if dir == -1:
                    (
                        q_minus,
                        p_minus,
                        grad_minus,
                        _,
                        _,
                        _,
                        q_prime_prime,
                        p_prime_prime,
                        grad_prime_prime,
                        hmc_log_prob_prime_prime,
                        ll_prime_prime,
                        ob_prime_prime,
                        n_prime_prime,
                        s_prime_prime,
                        alpha_prime_prime,
                        n_alpha_prime_prime,
                    ) = self.build_tree(
                        q_minus, p_minus, u, dir, depth - 1, grad_minus
                        )
                else:
                    (
                        _,
                        _,
                        _,
                        q_plus,
                        p_plus,
                        grad_plus,
                        q_prime_prime,
                        p_prime_prime,
                        grad_prime_prime,
                        hmc_log_prob_prime_prime,
                        ll_prime_prime,
                        ob_prime_prime,
                        n_prime_prime,
                        s_prime_prime,
                        alpha_prime_prime,
                        n_alpha_prime_prime,
                    ) = self.build_tree(
                        q_plus, p_plus, u, dir, depth - 1, grad_plus
                        )

                # This max() hack ensures that we don't get a divide by 0 error 
                # without disrupting the math
                if np.random.uniform() <= (
                    n_prime_prime / max((n_prime + n_prime_prime), 1)
                ):  # Accepted
                    q_prime = q_prime_prime
                    p_prime = p_prime_prime
                    grad_prime = grad_prime_prime
                    hmc_log_prob_prime = hmc_log_prob_prime_prime
                    ll_prime = ll_prime_prime
                    ob_prime = ob_prime_prime

                alpha_prime += alpha_prime_prime
                n_alpha_prime += n_alpha_prime_prime
                # The mass matrix is diagonal, so M^{-1} @ p = p / mass_matrix
                s_prime = (
                    s_prime_prime
                    * ((q_plus - q_minus) @ (p_minus / self.adaptive_matrix) >= 0)
                    * ((q_plus - q_minus) @ (p_plus / self.adaptive_matrix) >= 0)
                    )
                n_prime += n_prime_prime

            return (
                q_minus,
                p_minus,
                grad_minus,
                q_plus,
                p_plus,
                grad_plus,
                q_prime,
                p_prime,
                grad_prime,
                hmc_log_prob_prime,
                ll_prime,
                ob_prime,
                n_prime,
                s_prime,
                alpha_prime,
                n_alpha_prime,
            )


    ######################################################################
    # TRACKER FUNCTIONS
    ######################################################################


    def initialize_temp_trackers(self):
        """ Called at the start of each iteration. """

        self.temp_model_samples = []
        self.temp_model_obs = []
        self.temp_model_log_probs = []

        self.temp_new_sample_obs = []
        self.temp_new_sample_lls = []


    def update_temp_trackers(self, new_sample, new_sample_ob, new_sample_ll):
        """ Called after each multi-fidelity acceptance. """

        self.temp_model_samples.append(new_sample)
        self.temp_model_obs.append(new_sample_ob)
        self.temp_model_log_probs.append(new_sample_ll + self.new_log_prior)

        self.temp_new_sample_obs.append(new_sample_ob)
        self.temp_new_sample_lls.append(new_sample_ll)


    def update_all_trackers(self, overall_accepted):
        """ Called after the sample has been processed by any/all models. """

        # Update current save trackers
        self.current_save_accepted += overall_accepted
        self.current_save_sample_tracker += 1

        # Fix temp trackers if multi-fidelity didn't finish
        if overall_accepted == False:  # If the high fidelity model did not accept
            model_num = len(self.temp_model_log_probs)

            self.temp_model_samples += [
                self.current_sample for _ in range(model_num, self.num_models)
                ]
            self.temp_model_obs += list(self.current_sample_obs)[model_num:]
            self.temp_model_log_probs += [
                current_sample_ll + self.current_log_prior
                for current_sample_ll in self.current_sample_lls[model_num:]
                ]

        # Update global trackers
        self.all_model_samples.append(self.temp_model_samples)
        self.all_model_obs.append(self.temp_model_obs)
        self.all_model_log_probs.append(self.temp_model_log_probs)

        self.total_accepted += overall_accepted
        self.total_sample_tracker += 1

        self.acceptance_history.append(self.total_accepted / self.total_sample_tracker)


    def save_data_reset_trackers(self, save_filename, total_start_time):
        """ Saves data every specified number of iterations and after all 
        sampling. 
        """

        # Make sure individual acceptance ratios don't return Nans
        self.model_acceptances = np.array([
                self.model_acceptances[k] / self.model_eval_counter[k]
                if self.model_eval_counter[k] != 0
                else 0
                for k in range(self.num_models)
            ])
        self.overall_time = time.perf_counter() - total_start_time
        self.overall_acc_ratio = (
            self.current_save_accepted / self.current_save_sample_tracker
            )

        # Either save the data as a new file or add it to an existing file
        if exists(save_filename):
            self.add_data(save_filename)
        else:
            self.save_data(save_filename)

        # Reset statistic trackers
        self.all_model_samples = [self.temp_model_samples]
        self.all_model_obs = [self.temp_model_obs]
        self.all_model_log_probs = [self.temp_model_log_probs]
        self.overall_time = 0.0
        self.overall_acc_ratio = 0.0
        self.all_model_times = np.zeros(self.num_models, dtype=float)

        self.current_save_accepted = 0
        self.current_save_sample_tracker = 0
        self.model_eval_counter = np.zeros(self.num_models)
        self.model_acceptances = np.zeros(self.num_models)
        self.acceptance_history = []
        self.tree_depths = []


    def add_data(self, save_filename):
        """ Adds all data to an existing file called save_filename. """
        pass


    def save_data(self, save_filename):
        """ Saves all data to a new file called save_filename. """
        pass


    ######################################################################
    # PRIMARY SAMPLER FUNCTIONS
    ######################################################################


    def sample_multi_fidelity_rw(
        self, save_filename, n_samples, matrix_adapt_start, save_iters=1000
    ):
        """ Implementation of adaptive Multi-Fidelity RW, saving all data to 
        save_filename

        Parameters
        ----------
        save_filename (string):
            Name of file to save all data
        n_samples (int):
            The number of samples to evaluate
        matrix_adapt_start (int):
            The number of iterations at which to start adapting the step 
            covariance matrix (saved as self.adaptive_matrix); if None, the 
            matrix will not adapt
        save_iters (int):
            The number of iterations at which to save all data
        """

        total_start_time = time.perf_counter()

        # Initialize covariance adaptation term
        phi = (2.38) ** 2 / self.sample_dim

        self.initialize_covariance(matrix_adapt_start)

        # Initialize loading bar
        loop = tqdm(total=n_samples, position=0)
        for i in range(n_samples):

            # Adapt adaptive_matrix if specified
            if matrix_adapt_start != None:
                self.update_covariance(i)

            # Randomly select next step (random walk centered at previous state)
            new_sample = multivariate_normal.rvs(
                mean=self.current_sample, cov=phi * self.adaptive_matrix
                )

            # Compute the log priors of current and new sample 
            # -- ONLY NEED TO COMPUTE ONCE THIS ITERATION
            self.compute_current_log_prior(self.current_sample)
            self.compute_new_log_prior(new_sample)

            alpha_sam_nsam, alpha_nsam_sam = (
                1.0,
                1.0,
                )  # Initialize acceptance probabilities for lowest fidelity
            overall_accepted = (
                False  # Indicates whether the sample was accepted by all models
                )

            self.initialize_temp_trackers()

            for model in range(self.num_models):

                model_start_time = time.perf_counter()

                # Increment the number of times this model was accessed
                self.model_eval_counter[model] += 1

                current_sample_ll = self.current_sample_lls[model]

                # Compute new log likelihood
                new_sample_ll, new_sample_ob = self.log_likelihood(new_sample, model)

                # If the log likelihood is (negative) infinity, break out now
                if np.isinf(new_sample_ll):
                    self.all_model_times[model] += (
                        time.perf_counter() - model_start_time
                        )
                    break

                # Calculate acceptance probability
                acceptance = (
                    self.rw_acceptance_probability(current_sample_ll, new_sample_ll)
                    + np.log(alpha_nsam_sam)
                    - np.log(alpha_sam_nsam)
                    )

                # Check if sample should be accepted (using log scale)
                alpha_sam_nsam = np.exp(
                    min(0.0, acceptance)
                    )  # alpha(sample, new_sample)
                alpha_nsam_sam = np.exp(
                    min(0.0, -acceptance)
                    )  # alpha(new_sample, sample) but we're dealing with logs still

                if np.random.uniform() > alpha_sam_nsam:  # Rejected
                    self.all_model_times[model] += (
                        time.perf_counter() - model_start_time
                        )
                    break

                # Increment the number of times this model was accepted
                self.model_acceptances[model] += 1

                # Update temp statistic trackers
                self.update_temp_trackers(new_sample, new_sample_ob, new_sample_ll)

                if model == self.num_models - 1:  # The sample was accepted by all models
                    overall_accepted = True
                    self.current_sample = new_sample
                    self.current_sample_obs = self.temp_new_sample_obs
                    self.current_sample_lls = self.temp_new_sample_lls

                self.all_model_times[model] += time.perf_counter() - model_start_time

            # Process and update statistic trackers
            self.update_all_trackers(overall_accepted)

            # Update loading bar
            loop.set_description(
                "Acceptance Ratio: {:.4f}".format(self.acceptance_history[-1])
                )
            loop.update(1)

            # Save data every save_iters iterations and at the end
            if (i + 1) % save_iters == 0 or i + 1 == n_samples:
                self.save_data_reset_trackers(save_filename, total_start_time)
                total_start_time = time.perf_counter()


    def sample_multi_fidelity_nuts(
        self,
        save_filename,
        n_samples,
        matrix_adapt_start,
        epsilon_adapt,
        delta=0.65,
        save_iters=500,
    ):
        """ Implementation of adaptive Multi-Fidelity NUTS, saving all data to 
        save_filename

        Parameters
        ----------
        save_filename (string):
            Name of file to save all data
        n_samples (int):
            The number of samples to evaluate
        matrix_adapt_start (int):
            The number of iterations at which to start adapting the mass matrix
            (saved as self.adaptive_matrix); if None, the matrix will not adapt
        epsilon_adapt (int):
            The number of samples to "burn in" the adaptive epsilon
        delta (float):
            The desired ratio of samples accepted (between 0 and 1)
        save_iters (int):
            The number of iterations at which to save all data
        """
        total_start_time = time.perf_counter()

        # Initialize gradient for leapfrog step
        gradient, _, _ = self.derivative_log_density(self.current_sample, 0)

        self.initialize_variance(matrix_adapt_start)

        # Initialize epsilon if not already specified
        if self.epsilon is None:
            self.find_reasonable_epsilon(self.current_sample, gradient)
            print(f"Reasonable epsilon found: {self.epsilon}")
        else:
            self.averaged_epsilon = (
                self.epsilon
                )  # epsilon was set as averaged_epsilon in previous run

        # Initialize NUTS attributes
        self.mu = np.log(10 * self.epsilon)
        self.tree_depths = []

        # Initialize loading bar
        loop = tqdm(total=n_samples, position=0)
        for i in range(n_samples):

            # Adapt adaptive_matrix if specified
            if matrix_adapt_start != None:
                self.update_variance(i)

            # Randomly draw new, independent value for momentum p using the mass matrix
            current_p = np.random.normal(0, self.adaptive_matrix)

            # Compute the log prior of current sample 
            # -- ONLY NEED TO COMPUTE ONCE THIS ITERATION
            self.compute_current_log_prior(self.current_sample)

            alpha_sam_nsam, alpha_nsam_sam = (
                1.0,
                1.0,
                )  # Initialize acceptance probabilities for lowest fidelity
            overall_accepted = (
                False  # Indicates whether the sample was accepted by all models
                )

            self.initialize_temp_trackers()


            ##### PERFORM NUTS ON LOWEST FIDELITY FORWARD MODEL #####

            model_start_time = time.perf_counter()

            # Increment the number of times the first model was accessed
            self.model_eval_counter[0] += 1

            current_sample_ll = self.current_sample_lls[0]

            self.compute_current_hmc_log_posterior(current_p, current_sample_ll)

            # Draw slice sample u
            u = np.random.uniform(0, np.exp(self.current_hmc_log_posterior))

            # Initialize the new sample as the current sample
            new_sample = self.current_sample.copy()
            new_sample_ob = self.current_sample_obs[0].copy()
            new_sample_ll = self.current_sample_lls[0].copy()

            # Initialize NUTS Variables
            q_minus = self.current_sample.copy()
            q_plus = self.current_sample.copy()
            p_minus = current_p.copy()
            p_plus = current_p.copy()
            grad_minus = gradient.copy()
            grad_plus = gradient.copy()
            depth, n, s = 0, 1, 1

            for _ in range(self.max_tree_depth):
                direction = np.random.choice([-1, 1])
                if direction == -1:
                    (
                        q_minus,
                        p_minus,
                        grad_minus,
                        _,
                        _,
                        _,
                        q_prime,
                        p_prime,
                        grad_prime,
                        hmc_log_prob_prime,
                        ll_prime,
                        ob_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self.build_tree(
                        q_minus, p_minus, u, direction, depth, grad_minus
                        )
                else:
                    (
                        _,
                        _,
                        _,
                        q_plus,
                        p_plus,
                        grad_plus,
                        q_prime,
                        p_prime,
                        grad_prime,
                        hmc_log_prob_prime,
                        ll_prime,
                        ob_prime,
                        n_prime,
                        s_prime,
                        alpha,
                        n_alpha,
                    ) = self.build_tree(
                        q_plus, p_plus, u, direction, depth, grad_plus
                        )

                if s_prime == 1 and np.random.uniform() < min(1.0, n_prime / n):
                    new_sample = q_prime
                    new_p = p_prime
                    gradient = grad_prime
                    new_hmc_log_prob = hmc_log_prob_prime  # log P(q', p') - log P(q, p)
                    new_sample_ll = ll_prime
                    new_sample_ob = ob_prime

                    # Check if sample should be accepted (using log scale)
                    alpha_sam_nsam = np.exp(
                        min(0.0, new_hmc_log_prob)
                        )  # alpha(sample, new_sample)
                    alpha_nsam_sam = np.exp(
                        min(0.0, -new_hmc_log_prob)
                        )  # alpha(new_sample, sample), but we're dealing with logs still

                n += n_prime
                # The mass matrix is diagonal, so M^{-1} @ p = p / mass_matrix
                s = (
                    s_prime
                    * ((q_plus - q_minus) @ (p_minus / self.adaptive_matrix) >= 0)
                    * ((q_plus - q_minus) @ (p_plus / self.adaptive_matrix) >= 0)
                    )
                depth += 1

                if s == 0:  # Stopping Criteria flagged
                    break

            self.tree_depths.append(depth)

            # If epsilon_adapt is nonzero, adapt epsilon regardless of it being provided
            if i > 0 and i < epsilon_adapt:
                self.adapt_epsilon(i, delta, alpha, n_alpha)

            # At the last iteration of epsilon_adapt, set the new epsilon
            if i + 1 == epsilon_adapt:
                self.epsilon = self.averaged_epsilon

            self.all_model_times[0] += time.perf_counter() - model_start_time

            ##### END OF NUTS #####


            # Check if a new sample was accepted
            if not np.array_equal(new_sample, self.current_sample):

                # Increment the number of times this model was accepted
                self.model_acceptances[0] += 1

                # Compute log prior of new sample 
                # -- ONLY NEED TO COMPUTE ONCE FOR THE REMAINING ITERATION
                self.compute_new_log_prior(new_sample)

                # Update temp statistic trackers
                self.update_temp_trackers(new_sample, new_sample_ob, new_sample_ll)


                ##### COMPUTE ACCEPTANCE PROBABILITIES ON REMAINING FORWARD MODELS #####

                for model in range(1, self.num_models):

                    model_start_time = time.perf_counter()

                    # Increment the number of times this model was accessed
                    self.model_eval_counter[model] += 1

                    current_sample_ll = self.current_sample_lls[model]

                    # Compute new log likelihood
                    new_sample_ll, new_sample_ob = self.log_likelihood(
                        new_sample, model
                        )

                    # If the new likelihood is (negative) infinity, or if either 
                    # acceptance ratio is (basically) zero, break out now
                    if (
                        np.isinf(new_sample_ll)
                        or np.isclose(alpha_nsam_sam, 0.0, atol=1e-100)
                        or np.isclose(alpha_sam_nsam, 0.0, atol=1e-100)
                    ):
                        self.all_model_times[model] += (
                            time.perf_counter() - model_start_time
                            )
                        break

                    # Calculate HMC acceptance probability
                    self.compute_new_hmc_log_posterior(new_p, new_sample_ll)
                    new_hmc_log_prob = (
                        np.log(alpha_nsam_sam)
                        + self.new_hmc_log_posterior
                        - np.log(alpha_sam_nsam)
                        - self.current_hmc_log_posterior
                        )

                    # Check if sample should be accepted (using log scale)
                    alpha_sam_nsam = np.exp(
                        min(0.0, new_hmc_log_prob)
                        )  # alpha(sample, new_sample)
                    alpha_nsam_sam = np.exp(
                        min(0.0, -new_hmc_log_prob)
                        )  # alpha(new_sample, sample), but we're dealing with logs still

                    if np.random.uniform() > alpha_sam_nsam:  # Rejected
                        self.all_model_times[model] += (
                            time.perf_counter() - model_start_time
                            )
                        break

                    # Increment the number of times this model was accepted
                    self.model_acceptances[model] += 1

                    # Update temp statistic trackers
                    self.update_temp_trackers(new_sample, new_sample_ob, new_sample_ll)

                    if (
                        model == self.num_models - 1
                    ):  # The sample was accepted by all models
                        overall_accepted = True

                    self.all_model_times[model] += (
                        time.perf_counter() - model_start_time
                        )

                if (
                    overall_accepted or self.num_models == 1
                ):  # The sample was accepted by all models
                    overall_accepted = (
                        True  # In the case that there's only one model being used
                        )
                    self.current_sample = new_sample
                    self.current_sample_obs = self.temp_new_sample_obs
                    self.current_sample_lls = self.temp_new_sample_lls

            # Process and update statistic trackers
            self.update_all_trackers(overall_accepted)

            # Update loading bar
            loop.set_description(
                "Acc Ratio: {:.4f}, Tree Depth: {}, Eps: {:.8f}".format(
                    self.acceptance_history[-1], self.tree_depths[-1], self.epsilon
                ))
            loop.update(1)

            # Save data every save_iters samples and at the end
            if (i + 1) % save_iters == 0 or i + 1 == n_samples:
                self.average_tree_depth = np.mean(self.tree_depths)
                self.save_data_reset_trackers(save_filename, total_start_time)
                total_start_time = time.perf_counter()
