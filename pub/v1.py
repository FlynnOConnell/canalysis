import canalysis as ca
from copy import deepcopy
import suite2p
import mbo_utilities as mbo
import lbm_suite2p_python as lsp
import fastplotlib as fpl
from pathlib import Path
import numpy as np
import tifffile

#%%

fpath = Path().home().joinpath("diloren", "PGT13", "052622", "suite2p", "plane0")
ops = np.load(fpath.joinpath("ops.npy"), allow_pickle=True).item()
iscell = np.load(fpath.joinpath("iscell.npy"), allow_pickle=True)[:, 0].astype("bool")
spks = np.load(fpath.joinpath("spks.npy"), allow_pickle=True)

savepath = fpath.joinpath("./outputs")
savepath.mkdir(exist_ok=True)

#%%
fraw, fneu, spks = lsp.load_traces(ops)

data = ca.get_data()

data.plot_stim(savepath=savepath.joinpath("stim.png"))
data.plot_session(savepath=savepath.joinpath("session.png"))
data.plot_cells(savepath=savepath.joinpath("cells.png"))
data.plot_zoom(savepath=savepath.joinpath("zoom.png"))