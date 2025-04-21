import numpy as np
from scipy import sparse as sp
from functools import partial
from jax import numpy as jnp, jit, lax, config
from jax.experimental import sparse

config.update("jax_enable_x64", True)


def wave_solver(params, x_min, x_max, y_min, y_max, grid_size, dt, timesteps):
    """ Solves the forward wave problem given by

        u_tt - c^2(u_xx + u_yy) = 0
        u(t=0) = f, u_t(t=0) = 0
        u(x=x_min) = u(x=x_max) = u(y=y_min) = u(y=y_max) = 0

    This same function is used for all fidelity forward models by solving
    with a finite difference scheme over different gridsizes. JIT compiled.

    Parameters
        ----------
        params (ndarray):
            List of parameters over which to define the wave propagation
        x_min (float):
            The left-hand boundary for x
        x_max (float):
            The right-hand boundary for x
        y_min (float):
            The lower boundary for y
        y_max (float):
            The upper boundary for y
        grid_size (int):
            The number of discrete points to solve the solution in both the
            x and y direction
        dt (float):
            The temporal distance between each timestep
        timesteps (float):
            The number of timesteps to take

        Returns
        -------
        wave_sols (ndarray):
            The entire wave simulation for all timesteps
    """

    mean_11, mean_12, mean_21, mean_22, coeff_1, coeff_2, c = params

    # Multivariate Gaussian using jax numpy
    mv_gauss = (
        lambda coord, mean: 1 / (2 * jnp.pi)
        * jnp.exp(-0.5
            * (jnp.array(coord) - jnp.array(mean)).T
            @ (jnp.array(coord) - jnp.array(mean))
        ))

    f = lambda x, y: coeff_1 * mv_gauss(
        [x, y], [mean_11, mean_12]
        ) + coeff_2 * mv_gauss([x, y], [mean_21, mean_22])
    f = jnp.vectorize(f)

    wave_sols = jnp.zeros((timesteps, grid_size + 1, grid_size + 1))

    # Build the space grids
    (x, h), y = (
        jnp.linspace(x_min, x_max, grid_size + 1, retstep=True),
        jnp.linspace(y_min, y_max, grid_size + 1),
        )
    X, Y = jnp.meshgrid(x, y)

    # Build the matrix A
    tri_diag = sp.diags(
        ([1] * (grid_size - 2), [-4] * (grid_size - 1), [1] * (grid_size - 2)),
        offsets=(-1, 0, 1),
        )
    A = sp.block_diag([tri_diag] * (grid_size - 1)) + sp.diags(
        (
            [1] * ((grid_size - 1) ** 2 - grid_size + 1),
            [1] * ((grid_size - 1) ** 2 - grid_size + 1),
        ),
        offsets=(-grid_size + 1, grid_size - 1),
        )
    A = A.tocsr()

    B = sp.diags(([2] * (grid_size - 1) ** 2))
    B = B.tocsr()

    # Convert this sparse matrix to jax format
    A = sparse.BCOO.from_scipy_sparse(A)
    B = sparse.BCOO.from_scipy_sparse(B)

    # Multiply all elements by theta
    theta = (c * dt / h) ** 2
    A = B + (A * theta)

    U = jnp.zeros((grid_size + 1, grid_size + 1))
    U = U.at[1:-1, 1:-1].set(f(X[1:-1, 1:-1], Y[1:-1, 1:-1]))

    wave_sols = wave_sols.at[0].set(U)  # This is jax for wave_sols[0] = U

    temp_layer = U[1:-1, 1:-1] + (theta / 2) * (
        U[:-2, 1:-1] + U[2:, 1:-1] + U[1:-1, :-2] + U[1:-1, 2:] - 4 * U[1:-1, 1:-1]
        )

    wave_sols = wave_sols.at[1, 1:-1, 1:-1].set(temp_layer)

    wave_sols = propagate(
        A, wave_sols, grid_size, timesteps
        )  # Call function to propagate through for-loop
    return jnp.array(wave_sols)


@partial(jit, static_argnums=(2, 3))
def propagate(A, wave_sols, grid_size, timesteps):
    """ This function enables precompiling for-loop to increase optimization. """

    def _propagate(k: int, args):
        (A, wave_sols) = args

        nonlocal grid_size  # Call the grid_size defined in outer function

        wave_sols = wave_sols.at[k, 1:-1, 1:-1].set(
            (
                A @ wave_sols[k - 1][1:-1, 1:-1].flatten()
                - wave_sols[k - 2][1:-1, 1:-1].flatten()
            ).reshape((grid_size - 1, grid_size - 1))
            )
        return (A, wave_sols)

    (_, wave_sols) = lax.fori_loop(2, timesteps, _propagate, (A, wave_sols))

    return wave_sols


def get_buoy(x, y, grid_size):
    """ Generates indices for buoy of specified x and y ratios. """
    return [int(x * (grid_size + 1)), int(y * (grid_size + 1))]


def get_observations(wave_sols, buoys):
    """ Returns maximum values and corresponding timesteps attained at each buoy. """
    # Note that buoys is listed as [x-coord, y-coord]
    buoy_vals = wave_sols[:, buoys[:, 1], buoys[:, 0]]
    max_times = np.argmax(buoy_vals, axis=0)
    max_vals = np.max(buoy_vals, axis=0)
    return max_vals, max_times
