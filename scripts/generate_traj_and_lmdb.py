"""
This script can
1. run a relaxation trajectory given some initial structure and a trained model
2. transform multiple trajectories to one .lmdb file
3. filter out some of the trajectories
"""

import argparse
import os
import pickle

import ase.io
import lmdb
import torch
from ase.optimize import BFGS
from tqdm import tqdm

from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.datasets import LmdbDataset
from ocpmodels.preprocessing import AtomsToGraphs


def run_relaxations(args):

    if not os.path.isdir(args.adslabs_path):
        raise RuntimeError("Wrong adslabs_path")
    else:
        print(f"Using adslabs from {args.adslabs_path}")

    os.makedirs(args.trajs_path, exist_ok=True)
    print(f"Storing trajectories in {args.trajs_path}")

    # paths to initial adslabs (adsorbate + catalyst) for relaxation
    dirs = sorted(os.listdir(args.adslabs_path))

    for dir in dirs:
        adslab_path = os.path.join(args.adslabs_path, dir, "adslabatoms.pkl")
        adslab = pickle.load(open(adslab_path, "rb"))

        # Set up the calculator
        # ["cpu", "0"] where "0"="cuda:0"
        adslab.calc = OCPCalculator(
            config_yml=args.config_yml, checkpoint=args.checkpoint, device="0"
        )

        # set up optimizer/iterator
        opt = BFGS(
            adslab, trajectory=os.path.join(args.trajs_path, dir + ".traj")
        )

        # use threshold values as in the OC20 paper:
        # max force among all atoms of 0.03 and a max of 200 iterations
        opt.run(fmax=0.03, steps=200)


def run_trajs_to_lmdb(args):
    """Stack each iteration of multiple trajectories into one .lmbd file.
    Based on tutorials/lmdb_dataset_creation.ipynb

    Each database element has the following property:
    pos: [N,3]
    cell: [1,3,3]
    atomic_numbers: [N]
    natoms: N
    y: float, energy
    force: [N,3]
    fixed: [N] implements fixation of tag=0 atoms
    tags: 0 - subsurface, 1 - surface, 2 - adsorbate (for each particle)
    fid: Frame index along the trajcetory
    sid: Unique system/trajectory identifier, arbitrary
    """

    if not os.path.isdir(args.trajs_path):
        raise RuntimeError("Wrong trajs_path")
    else:
        print(f"Using trajectories from {args.trajs_path}")

    os.makedirs(args.lmdb_path, exist_ok=True)
    print(f"Storing lmdb in {args.lmdb_path}.")

    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,  # False for test data
        r_forces=True,
        r_distances=False,
        r_edges=False,
        r_fixed=True,
    )

    db = lmdb.open(
        os.path.join(args.lmdb_path, "data.0000.lmdb"),  # name of lmdb file
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    # list of all used trajectories
    dirs = sorted(os.listdir(args.trajs_path))

    # enumerate each element of the db
    idx = 0

    for dir in dirs:
        traj = ase.io.read(os.path.join(args.trajs_path, dir), ":")
        # Tags don't change along a trajectory. Get them from first state.
        tags = traj[0].get_tags()
        data_objects = a2g.convert_all(traj, disable_tqdm=True)

        # system ID for the given adsorbate + catalyst configuration
        # TODO: better way for enumeration? For now seeds.
        sid = int(dir[-12:-5])

        for fid, data in tqdm(enumerate(data_objects)):  # iterate 1 trajectory
            data.tags = torch.LongTensor(tags)
            data.fid = torch.LongTensor([fid])
            data.sid = torch.LongTensor([sid])

            # Add a flag attirbute to distinguis whether the sample comes from
            # a DFT relaxation, ML-based relaxation, or any other.
            # The OC20 data doesn't have this attribute.
            data.origin = "gemnet-oc-2M"

            # Write to LMDB
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
            txn.commit()

            idx += 1

    # length of lmdb file
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()


def run_filtering(args):
    """Filter trajectories with adsorbate flying away and other conditions
    mentioned in the OC20 paper
    """

    if not os.path.isdir(args.lmdb_path):
        raise RuntimeError("Wrong lmdb_path")
    else:
        print(f"Using trajectories from {args.lmdb_path}")

    os.makedirs(args.lmdb_filter_path, exist_ok=True)
    print(f"Storing lmdb in {args.lmdb_filter_path}.")

    db = lmdb.open(
        os.path.join(args.lmdb_filter_path, "data.0000.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    dataset = LmdbDataset(
        {"src": os.path.join(args.lmdb_path, "data.0000.lmdb")}
    )

    l1, l2 = [], {}
    idx = 0

    for i in range(len(dataset) - 1):
        # skip the last element as this is the length of the database
        data = dataset[i]

        # 1. We implement a heuristic detecting whether an adsorbate flies away
        # we check whether the highest surface atom is more than half of the
        # interaction radius (3A) lower than the lowest adsorbate atom

        # largest z coordinate of surface atoms
        max_surf = (data.pos[data.tags == 1, 2]).max()
        # lowest z coordinate of adsorbate
        min_ads = (data.pos[data.tags == 2, 2]).min()
        if (max_surf + 3 < min_ads) and (data.sid.item() not in l1):
            l1.append(data.sid.item())

    for i in range(len(dataset) - 1):
        # skip the last element as this is the length of the database
        data = dataset[i]

        # implement 1 from above to discard the whole trajectory
        if data.sid.item() in l1:
            continue

        # 2. We apply a heuristic used for the generation of the OCP dataset
        # reference: lmdb_dataset_creation.ipynb
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

        energy = abs(data.y)
        force_max = data.force.norm(dim=1).max()
        if energy > 10 or force_max > 50:
            k, v = data.sid.item(), data.fid.item()
            if k not in l2:
                l2[k] = [v]
            else:
                l2[k].append(v)

            l2.append((data.sid.item(), data.fid.item()))
            continue

        # Write to new filtered LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

        idx += 1

    # length of filtered lmdb file
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    print("LOGGING")
    # (trajs: 44)
    print(
        f"Trajs with adsorbate flying away (trajs:{len(l1)}).",
        f"First 1000: {l1[:1000]}",
    )
    print(
        f"Frames violating E or F (trajs:{len(l2.keys())}, "
        f"frames:{sum([len(l2[k]) for k in l2])}). {l2}"
    )


def main(args):
    mode = args.mode

    if mode == "relax" or mode == "all":
        run_relaxations(args)
    if mode == "lmdb" or mode == "all":
        run_trajs_to_lmdb(args)
    if mode == "filter" or mode == "all":
        run_filtering(args)


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help="""One of the following: [relax, lmdb, filter, all].
        relax runs relaxations -> .traj files,
        lmdb transforms multiple .traj files to one .lmdb file.""",
    )

    # for relaxation only
    parser.add_argument("--config_yml", type=str, help="Configuration file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model to compute forces at each step of relaxation.",
    )
    parser.add_argument(
        "--adslabs_path",
        type=str,
        help="""Path to directory containing other directories, each of which
        having an 'adslabatoms.pkl' file.""",
    )

    # for relaxation and lmdb
    parser.add_argument(
        "--trajs_path", type=str, help="Directory of trajectory files."
    )

    # for lmdb only
    parser.add_argument(
        "--lmdb_path",
        type=str,
        help="Directory where to store the .lmdb file.",
    )

    # for filtering only
    parser.add_argument(
        "--filter",
        action="store_true",
        default=True,
        help="Whether to filter out parts of trajectories.",
    )
    parser.add_argument(
        "--lmdb_filter_path",
        type=str,
        help="Directory where to store filtered .lmdb file.",
    )

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    main(args)
