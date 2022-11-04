import os
import pickle
from pkgutil import get_data

import ase.io
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from tqdm import tqdm

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.datasets import (
    LmdbDataset,
    SinglePointLmdbDataset,
    TrajectoryLmdbDataset,
)
from ocpmodels.preprocessing import AtomsToGraphs


def get_db_keys_vals_stat(path: str):
    """iterator over all db keys and values"""

    db = lmdb.open(
        path,  # name of lmdb file
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin()
    keys = []
    vals = []
    for key, value in txn.cursor():
        # key = txn.get("some_string".encode('ascii'))  # some_string = 0, 1, .., length
        # value = txn.get(key)

        k = key.decode("ascii")
        v = pickle.loads(value)

        if k != "length":
            v = pyg2_data_transform(v)

        keys.append(k)
        vals.append(v)

    return keys, vals, db.stat()


if __name__ == "__main__":

    # import sys
    # adslab = pickle.load(open('/home/artur/code/Open-Catalyst-Dataset/outputs/random5000006/adslabatoms.pkl', 'rb'))
    # print(adslab)
    # sys.exit()

    root = "/home/atoshev/code/ocp/"
    ref_lmdb_path = "data/s2ef/200k/train/data.0002.lmdb"

    # relax_name = 'CuCO-EMT'  # CuCO-EMT or CuC3H8-GemNet
    # relax_name = 'CuC3H8-GemNet'
    relax_name = "set"

    if relax_name == "CuCO-EMT":
        my_lmdb_path = "data/distill/CuCO.lmdb"
        my_traj_path = "data/distill/CuCO.traj"
    elif relax_name == "CuC3H8-GemNet":
        my_lmdb_path = "data/distill/CuC3H8.lmdb"
        my_traj_path = "data/distill/CuC3H8.traj"
    elif relax_name == "set":
        my_lmdb_path = "data/distill/set.lmdb"
        my_traj_path = "data/distill/set/"  # TODO: directory?
        my_adslabs_src_path = root + "outputs3/"
        os.makedirs(root + my_traj_path, exist_ok=True)

    os.makedirs(root + "data/distill", exist_ok=True)
    ##########################################################################

    """manually created adslab -> traj (using calculator)"""

    # checkpoint_path = root + "checkpoints/gemnet_t_direct_h512_all.pt"
    # config_yml_path = root + "configs/s2ef/all/gemnet/gemnet-dT.yml"
    checkpoint_path = root + "checkpoints/gemnet_oc_base_s2ef_2M.pt"
    config_yml_path = root + "configs/s2ef/2M/gemnet/gemnet-oc.yml"

    if relax_name == "CuCO-EMT":
        # based on tutorials/lmdb_dataset_creation.ipynb

        adslab = fcc100("Cu", size=(2, 2, 3))
        ads = molecule("CO")
        add_adsorbate(adslab, ads, 3, offset=(1, 1))
        cons = FixAtoms(
            indices=[atom.index for atom in adslab if (atom.tag == 3)]
        )
        adslab.set_constraint(cons)
        adslab.center(vacuum=13.0, axis=2)
        adslab.set_pbc(True)

        adslab.set_calculator(EMT())  # Effective Medium Theory calculator

        dyn = BFGS(adslab, trajectory=root + my_traj_path, logfile=None)
        dyn.run(fmax=0, steps=1000)

    elif relax_name == "CuC3H8-GemNet":
        # based on tutorials/OCP_Tutorials.ipynb#OCP-Calculator

        # Construct a sample structure
        adslab = fcc100("Cu", size=(3, 3, 3))
        adsorbate = molecule("C3H8")
        add_adsorbate(adslab, adsorbate, 3, offset=(1, 1))
        tags = np.zeros(len(adslab))
        tags[18:27] = 1
        tags[27:] = 2
        adslab.set_tags(tags)
        cons = FixAtoms(
            indices=[atom.index for atom in adslab if (atom.tag == 0)]
        )
        adslab.set_constraint(cons)
        adslab.center(vacuum=13.0, axis=2)
        adslab.set_pbc(True)
        # print(adslab)
        # >> Atoms(symbols='Cu27C3H8', pbc=True, cell=[7.65796644025031, 7.65796644025031, 33.266996999999996], tags=..., constraint=FixAtoms(indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
        # print(adslab.get_tags())
        # >> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        # tags: 0-sub-surface, 1-surface, 2-adsorbate

        # Define the calculator
        calc = OCPCalculator(
            config_yml=config_yml_path, checkpoint=checkpoint_path
        )
        # Set up the calculator
        adslab.calc = calc

        # os.makedirs(root+"data/sample_ml_relax", exist_ok=True)
        opt = BFGS(adslab, trajectory=root + my_traj_path)
        opt.run(fmax=0.05, steps=100)

    elif relax_name == "set":

        dirs = os.listdir(my_adslabs_src_path)
        dirs = sorted(dirs)

        for dir in dirs:

            adslab_path = my_adslabs_src_path + dir + "/adslabatoms.pkl"
            adslab = pickle.load(open(adslab_path, "rb"))

            # Set up the calculator
            adslab.calc = OCPCalculator(
                config_yml=config_yml_path, checkpoint=checkpoint_path
            )

            opt = BFGS(adslab, trajectory=root + my_traj_path + dir + ".traj")
            opt.run(fmax=0.03, steps=200)

    """1 traj -> 1 lmdb + save lmdb"""

    # # tags (optional): 0 - subsurface, 1 - surface, 2 - adsorbate
    # # fid: Frame index along the trajcetory
    # # sid: Unique system identifier, arbitrary

    # # target dataset format
    # dataset = LmdbDataset({"src": root + ref_lmdb_path})
    # print(len(dataset), dataset[0])

    # # my dataset
    # # based on tutorials/lmdb_dataset_creation.ipynb

    # a2g = AtomsToGraphs(
    #     max_neigh=50,
    #     radius=6,
    #     r_energy=True,    # False for test data
    #     r_forces=True,
    #     r_distances=False,
    #     r_edges=False,
    #     r_fixed=True,
    # )

    # db = lmdb.open(
    #     root + my_lmdb_path,  # name of lmdb file
    #     map_size=1099511627776 * 2,
    #     subdir=False,
    #     meminit=False,
    #     map_async=True,
    # )

    # system_paths = [root+my_traj_path]  # list of all used trajectories
    # sid = 5000000 # system ID for the fiven adsorbate + catalyst configuration

    # for system in system_paths:
    #     traj = ase.io.read(system, ":")
    #     tags = traj[0].get_tags()  # tags don't change along a trajectory
    #     # raw_data = ase.io.read(root+"data/sample_ml_relax/toy_c3h8_relax2.traj", ":")
    #     data_objects = a2g.convert_all(traj, disable_tqdm=True)

    #     for fid, data in tqdm(enumerate(data_objects)):  # iterate 1 trajectory
    #         data.tags = torch.LongTensor(tags)
    #         data.fid = torch.LongTensor([fid])
    #         data.sid = torch.LongTensor([sid])  # TODO: Unique system identifier, arbitrary

    #         # Filter data if necessary
    #         # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

    #         # # no neighbor edge case check  # TODO: really needed?
    #         # if data.edge_index.shape[1] == 0:  # NOTE: by `r_edges=False` we don't store edges
    #         #     print("no neighbors")
    #         #     continue

    #         # Write to LMDB
    #         txn = db.begin(write=True)
    #         # fid makes sense if only one trajectory per lmdb file
    #         txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
    #         txn.commit()

    #     # print(data_objects[1])

    # txn = db.begin(write=True)
    # txn.put(f"length".encode("ascii"),  # TODO: the length of lmdb data elements?
    #         pickle.dumps(len(data_objects), protocol=-1))
    # txn.commit()

    # db.sync()
    # db.close()

    # # ks, vs, _ = get_db_keys_vals_stats(root + "tutorials/sample_CuCO.lmdb")

    # # load my dataset as durint training
    # dataset_ = LmdbDataset({"src": root + my_lmdb_path})
    # print(len(dataset_), dataset_[0])
