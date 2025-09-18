#!/usr/bin/env python

"""@package docstring
File: gen_init_mts.py
Author: Adam Lamson
Email: adam.r.lamson@gmail.com
Description:

"""

import sys
from pathlib import Path
import numpy as np
import yaml
import h5py
import re


def main(h5_file, frame, output_dir):
    print(frame)
    with h5py.File(h5_file, "r") as f:
        t_grps = [grp for grp in f if grp.startswith("t")]
        frame_grp = sorted(
            t_grps,
            key=lambda x: int(re.findall(r"\d+", x)[0]),
        )[frame]

        filaments = f[frame_grp]["filaments"][:]
        nucleators = f[frame_grp]["nucleators"][:]
        centrosomes = f[frame_grp]["centrosomes"][:]

        params = yaml.safe_load(f.attrs["params"])

        # Number of timesteps leading to frame
        if frame >= 0:
            n_time_steps = frame * int(params["t_snap"] / params["dt"])
        elif frame < 0:
            n_time_steps = (len(t_grps) + frame) * \
                int(params["t_snap"] / params["dt"])

        params["rng_iteration"] = n_time_steps + params.get("rng_iteration", 0)

    # Save initial state
    np.savetxt(output_dir /
               params.get("mt_file", "mt_init.csv"),
               filaments,
               delimiter=",",
               fmt=[
                   "%d",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%d",
                   "%d",
                   "%d",
               ],
               )
    np.savetxt(output_dir /
               params.get("nuc_file", "nucleators_init.csv"),
               nucleators,
               delimiter=",",
               fmt=[
                   "%d",
                   "%d",
                   "%d",
                   "%.5f",
               ],
               )
    np.savetxt(output_dir /
               params.get("cent_file", "centrosome_init.csv"),
               centrosomes,
               delimiter=",",
               fmt=[
                   "%d",
                   "%.5f",
                   "%.5f",
                   "%.5f",
                   "%.5f",
               ],
               )
    # Save parameters
    with open(f"{output_dir / Path(params['file_name']).stem}.yaml", "w") as f:
        yaml.dump(params, f)


if __name__ == "__main__":
    h5_file = sys.argv[1]

    frame = -1 if len(sys.argv) < 3 else int(sys.argv[2])

    output_dir = Path.cwd()

    main(h5_file, frame)
