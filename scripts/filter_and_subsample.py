"""
This script can
0. count how many frames there are in directory with .traj files
1. filter out trajectories with adsorbate flying away
2. filter out frames with E>10, F>50
4. generate an array of [[random_seed, frame_idx], ...]
5. shuffle
6. subsample 10% of that list -> indices of samples of interest
7. Write .extxyz and .txt files of size 5k for our ~1M frames
8 compress .extxyz and .txt to [...].xz
9. preprocess the .extxyz and .txt to .lmdb
"""

import argparse
import glob
import lzma
import multiprocessing as mp
import os
import pickle

import ase.io
import numpy as np
import preprocess_ef as preprocess
from tqdm import tqdm


def body_count(trajs_path):
    traj = ase.io.read(trajs_path, ":")
    return len(traj)


def body_filtering(trajs_path):

    fname_base = os.path.basename(trajs_path)
    traj = ase.io.read(trajs_path, ":")
    seed = int(fname_base.split(".")[0][6:])

    # 1. We implement a heuristic detecting whether an adsorbate flies away
    # we check whether the highest surface atom is more than half of the
    # interaction radius (3A) lower than the lowest adsorbate atom

    for frame in traj:

        # largest z coordinate of surface atoms
        max_surf = (frame.get_positions()[frame.get_tags() == 1, 2]).max()
        # lowest z coordinate of adsorbate
        min_ads = (frame.get_positions()[frame.get_tags() == 2, 2]).min()
        if max_surf + 3 < min_ads:
            return []

    # 2. We apply a heuristic used for the generation of the OCP dataset
    # reference: lmdb_dataset_creation.ipynb
    # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

    frames_list = []  # full list: [[seed, i] for i in range(len(traj))]

    for i, frame in enumerate(traj):

        energy = abs(frame.get_potential_energy())
        force_max = np.linalg.norm(frame.get_forces(), axis=1).max()
        if energy < 10 or force_max < 50:
            frames_list.append([seed, i])

    return frames_list


def body_save(input):
    """save .extxyz and .txt files for each batch of 5k samples

    input: (index, args,  ndarray of shape [5000, 2])
    """

    i, args, frames = input
    trajs_path, extxyz_path = args.trajs_path, args.extxyz_path

    extxyz_data = []
    txt_data = []
    for frame in frames:
        # open trajectory from which we want one frame
        traj_file = "random" + str(frame[0]) + ".traj"
        traj = ase.io.read(os.path.join(trajs_path, traj_file), ":")

        # get frame from relevant trajectory
        extxyz_data.append(traj[frame[1]])
        txt_data.append(
            "random" + str(frame[0]) + ",frame" + str(frame[1]) + ",0.0"
        )

    # write .extxyz and .txt files
    ase.io.write(os.path.join(extxyz_path, str(i) + ".extxyz"), extxyz_data)

    with open(os.path.join(extxyz_path, str(i) + ".txt"), "w") as f:
        for item in txt_data:
            f.write("%s\n" % item)


def compress_list_of_files(ip_op_pair):
    inpfile, outfile = ip_op_pair

    with open(inpfile, "rb") as f:
        contents = lzma.compress(f.read())
        with open(outfile, "wb") as op:
            op.write(contents)


def main(args):

    if (
        ("count" in args.mode)
        or ("filter_and_list" in args.mode)
        or ("subsample" in args.mode)
        or ("save" in args.mode)
    ):

        if os.path.isdir(args.trajs_path):
            print(f"Using trajectories from {args.trajs_path}")

        meta_dir = args.trajs_path.split("/")[-1]
        root_dir = args.trajs_path.split(meta_dir)[0]
        meta_path = root_dir + meta_dir + "_meta"
        args.meta_path = meta_path
        os.makedirs(meta_path, exist_ok=True)

        traj_list = glob.glob(os.path.join(args.trajs_path, "*.traj"))

    pool = mp.Pool(args.num_workers)

    if "count" in args.mode:
        # count number of frames
        counts = list(
            tqdm(
                pool.imap(body_count, traj_list),
                total=len(traj_list),
                desc=f"Count # frames of trajs in {args.trajs_path}",
            )
        )

        print("Number of frames before filtering: ", sum(counts))

    if "filter_and_list" in args.mode:
        # filter frames
        frames_list_ = list(
            tqdm(
                pool.imap(body_filtering, traj_list),
                total=len(traj_list),
                desc=f"Filter traj {args.trajs_path}",
            )
        )

        # frames_list_: a set of pairs [[traj X,frame 0],[traj X,frame 1],...]
        tmp = [np.array(traj) for traj in frames_list_ if len(traj) > 0]
        frames_array = np.vstack((tmp))

        print("Number of frames after filtering: ", frames_array.shape[0])

        with open(os.path.join(meta_path, "filtered_array.pkl"), "wb") as f:
            pickle.dump(frames_array, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "subsample" in args.mode:

        with open(os.path.join(meta_path, "filtered_array.pkl"), "rb") as f:
            frames_array = pickle.load(f)

        # shuffle in place
        np.random.seed(123)
        np.random.shuffle(frames_array)
        # sample 1M frames out of the dataset
        subset = frames_array[:1000000]
        print(f"1M frames = {1e8/len(frames_array):.4f}% of all frames")

        with open(
            os.path.join(meta_path, "filtered_array_subset.pkl"), "wb"
        ) as f:
            pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "save" in args.mode:

        os.makedirs(args.extxyz_path, exist_ok=True)

        with open(
            os.path.join(meta_path, "filtered_array_subset.pkl"), "rb"
        ) as f:
            subset = pickle.load(f)

        chunks = [
            subset[5000 * i : 5000 * (i + 1)]
            for i in range(len(subset) // 5000)
        ]
        # enumerate for naming of .extxyz files
        chunks = [(i, args, chunk) for i, chunk in enumerate(chunks)]

        list(
            tqdm(
                pool.imap(body_save, chunks),
                total=len(chunks),
                desc=f"Storing chunks of 5k frames in {args.extxyz_path}",
            )
        )

        print("Saving .extxyz and .txt finished successfully!")

    if "compress" in args.mode:

        os.makedirs(args.extxyz_xz_path, exist_ok=True)

        filelist = glob.glob(
            os.path.join(args.extxyz_path, "*txt")
        ) + glob.glob(os.path.join(args.extxyz_path, "*extxyz"))
        ip_op_pairs = []
        for i in filelist:
            fname_base = os.path.basename(i)
            ip_op_pairs.append(
                (i, os.path.join(args.extxyz_xz_path, fname_base + ".xz"))
            )

        list(
            tqdm(
                pool.imap(compress_list_of_files, ip_op_pairs),
                total=len(ip_op_pairs),
                desc=f"Compressing {args.extxyz_path}",
            )
        )

    if "preprocess" in args.mode:

        parser = preprocess.get_parser()
        _args, _ = parser.parse_known_args()
        _args.data_path = args.extxyz_path
        _args.out_path = args.lmdb_path
        _args.num_workers = args.num_workers
        preprocess.main(_args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        help="one of [count,filter_and_list,subsample,save,compress,preprocess]",
    )
    parser.add_argument(
        "--trajs_path", type=str, help="Path to compressed dataset directory"
    )
    parser.add_argument(
        "--num_workers", type=int, help="# of processes to parallelize across"
    )
    parser.add_argument(
        "--extxyz_path", type=str, help="Path to store the uncompressed data"
    )
    parser.add_argument(
        "--extxyz_xz_path", type=str, help="Path to store the compressed data"
    )
    parser.add_argument(
        "--lmdb_path", type=str, help="Path to store the lmdb files"
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
