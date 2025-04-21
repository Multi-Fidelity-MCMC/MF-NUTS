import forward_models

import numpy as np


if __name__ == "__main__":
    # TRUE VALUES
    MEAN_11, MEAN_12 = -3, -3
    MEAN_21, MEAN_22 = -15, 15
    COEFF_1, COEFF_2 = 1, 4
    C = 0.8

    TRUE_PARAMETERS = np.array(
        [MEAN_11, MEAN_12, MEAN_21, MEAN_22, COEFF_1, COEFF_2, C]
        )

    XMIN, XMAX, YMIN, YMAX = -20, 20, -20, 20
    GRIDSIZE = 200
    T = 30
    TIMESTEPS = 400

    _, dt = np.linspace(0, T, TIMESTEPS, retstep=True)
    U_sols = forward_models.wave_solver(
        TRUE_PARAMETERS, XMIN, XMAX, YMIN, YMAX, GRIDSIZE, dt, TIMESTEPS
        )

    # Scattered Locations [x-location, y-location] where each location is the 
    # percentage of the distance from the min to max value
    BUOY_RATS = np.array([[0.25, 0.4], 
                          [0.5, 0.25], 
                          [0.5, 0.5], 
                          [0.6, 0.9], 
                          [0.8, 0.4]])

    # Indices of each buoy (dependent on gridsize)
    BUOYS = np.array([
            forward_models.get_buoy(buoy_rat[0], buoy_rat[1], GRIDSIZE)
            for buoy_rat in BUOY_RATS
        ])

    OBS_VALS, OBS_TIMES = forward_models.get_observations(U_sols, BUOYS)
    OBS_TIMES_SECS = (OBS_TIMES + 1) / TIMESTEPS * T
    ALL_OBS = np.hstack((OBS_VALS, OBS_TIMES_SECS))

    PRIOR_MEAN = np.array([-2, -2, -14, 14, 2, 2])
    PRIOR_COV_DIAG = np.array([1, 1, 1, 1, 1, 1])
    PRIOR_A = 16
    PRIOR_B = 4

    LL_COV_DIAG = np.hstack(
        (np.ones_like(OBS_VALS) * 0.03, np.ones_like(OBS_TIMES_SECS) * 4)
        )

    dict = {
        "TRUE_PARAMETERS": TRUE_PARAMETERS,
        "XMIN": XMIN,
        "XMAX": XMAX,
        "YMIN": YMIN,
        "YMAX": YMAX,
        "T": T,
        "TIMESTEPS": TIMESTEPS,
        "BUOY_RATS": BUOY_RATS,
        "OBS_VALS": OBS_VALS,
        "OBS_TIMES": OBS_TIMES,
        "ALL_OBS": ALL_OBS,
        "PRIOR_MEAN": PRIOR_MEAN,
        "PRIOR_COV_DIAG": PRIOR_COV_DIAG,
        "PRIOR_A": PRIOR_A,
        "PRIOR_B": PRIOR_B,
        "LL_COV_DIAG": LL_COV_DIAG,
    }

    np.save("wave_specs", dict)
