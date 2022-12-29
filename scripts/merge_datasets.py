import argparse
import os
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np

import ocpmodels


def gen_metadata_simple(path, num_workers):
    import make_lmdb_sizes

    _args = Namespace(data_path=path, num_workers=num_workers)
    make_lmdb_sizes.main(_args)

    print(f"Done generating metadata for {path}")


def merge_files(path0, path1, target):
    target_file_index = 0
    for i, path in enumerate([path0, path1]):
        path_files = sorted(Path(path).glob("*.lmdb"))
        assert len(path_files) > 0, f"No LMDBs found in '{path_files}'"

        # copy to new folder
        for src_file in path_files:
            target_file_name = (
                "data." + str(target_file_index).zfill(4) + ".lmdb"
            )
            shutil.copy2(src_file, os.path.join(target, target_file_name))
            target_file_index += 1

    print("Done merging files.")


def gen_metadata_merged(path0, path1, target):

    meta1 = np.load(os.path.join(path0, "metadata.npz"))
    meta2 = np.load(os.path.join(path1, "metadata.npz"))
    target_meta_path = os.path.join(target, "metadata.npz")

    target_natoms = np.concatenate([meta1["natoms"], meta2["natoms"]])

    # denote with tag "origin" = 0 DFT data and with 1 our relaxed frames
    origin1 = np.zeros(meta1["natoms"].shape, dtype=meta1["natoms"].dtype)
    origin2 = np.ones(meta2["natoms"].shape, dtype=meta2["natoms"].dtype)
    target_origin = np.concatenate([origin1, origin2])

    # TODO: implement edges if provided

    np.savez(target_meta_path, natoms=target_natoms, origin=target_origin)

    print("Done generating target metadata.")


def main(args):
    """Merge a DFT-based and a further synthetic dataset

    In addition, a metadata file of the merged dataset will be generated. This
    extends the original metadata file generated with `make_lmdb_sizes.py` by
    adding a further vector containing tag information. The tags are:

    0: DFT data (src_dft)
    1: synthetic data (src_synt), e.g. generated with a SOTA model like GemNet

    The merged dataset will be stored in args.target
    """

    root = os.path.dirname(ocpmodels.__path__[0])
    path0 = os.path.join(root, args.src_dft)
    path1 = os.path.join(root, args.src_synt)
    target = os.path.join(root, args.target)

    # create directory for merged data
    os.makedirs(args.target, exist_ok=True)

    # if there is no `metadata.npz` in each of the subsets, then generate it
    for path_i in [path0, path1]:
        metadata_file = os.path.join(path_i, "metadata.npz")
        if not os.path.isfile(metadata_file):
            gen_metadata_simple(path_i, args.num_workers)

    # put all files in one place
    merge_files(path0, path1, target)

    # create combined metadata file including "origin" label 0/1
    gen_metadata_merged(path0, path1, target)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dft", type=str, default="data/s2ef/2M/train")
    parser.add_argument("--src_synt", type=str, default="data/s2ef/d1M/train")
    parser.add_argument("--target", type=str, default="data/s2ef/2M+d1M/train")
    parser.add_argument("--num-workers", type=int, default=1)
    args, _ = parser.parse_known_args()
    main(args)
