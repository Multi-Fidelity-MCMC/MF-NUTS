import numpy as np


TRUE_PARAMETERS = None
XMIN = None
XMAX = None
YMIN = None
YMAX = None
T = None
TIMESTEPS = None
BUOY_RATS = None
OBS_VALS = None
OBS_TIMES = None
ALL_OBS = None
PRIOR_MEAN = None
PRIOR_COV_DIAG = None
PRIOR_A = None
PRIOR_B = None
LL_COV_DIAG = None


def init(variable_filename):
    """Load in global model variables from specified file"""

    global \
        TRUE_PARAMETERS, \
        XMIN, \
        XMAX, \
        YMIN, \
        YMAX, \
        T, \
        TIMESTEPS, \
        BUOY_RATS, \
        OBS_VALS, \
        OBS_TIMES, \
        ALL_OBS, \
        PRIOR_MEAN, \
        PRIOR_COV_DIAG, \
        PRIOR_A, \
        PRIOR_B, \
        LL_COV_DIAG

    dict = np.load(variable_filename, allow_pickle=True)

    TRUE_PARAMETERS = dict.item().get("TRUE_PARAMETERS")
    XMIN = dict.item().get("XMIN")
    XMAX = dict.item().get("XMAX")
    YMIN = dict.item().get("YMIN")
    YMAX = dict.item().get("YMAX")
    T = dict.item().get("T")
    TIMESTEPS = dict.item().get("TIMESTEPS")
    BUOY_RATS = dict.item().get("BUOY_RATS")
    OBS_VALS = dict.item().get("OBS_VALS")
    OBS_TIMES = dict.item().get("OBS_TIMES")
    ALL_OBS = dict.item().get("ALL_OBS")
    PRIOR_MEAN = dict.item().get("PRIOR_MEAN")
    PRIOR_COV_DIAG = dict.item().get("PRIOR_COV_DIAG")
    PRIOR_A = dict.item().get("PRIOR_A")
    PRIOR_B = dict.item().get("PRIOR_B")
    LL_COV_DIAG = dict.item().get("LL_COV_DIAG")
