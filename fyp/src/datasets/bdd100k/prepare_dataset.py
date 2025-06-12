########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import argparse as ap
from collections import OrderedDict
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from bdd100k import Bdd100kBase


RGB_DIR = "leftImg8bit"
PARAMETERS_RAW_DIR = "camera"
DISPARITY_RAW_DIR = "disparity"
LABEL_DIR = "gtFine"


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype="uint8"))
    img.putpalette(list(np.asarray(colormap, dtype="uint8").flatten()))
    img.save(filepath, "PNG")


def get_files_by_extension(
    path, extension=".png", flat_structure=False, recursive=False, follow_links=True
):
    # check input args
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))

    if flat_structure:
        filelist = []
    else:
        filelist = {}

    # path is a file
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist

    # get filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True, followlinks=follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend((os.path.join(root, f) for f in filenames))
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break

    # return
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))


if __name__ == "__main__":
    # argument parser
    parser = ap.ArgumentParser(
        description="Prepare Cityscapes dataset for segmentation."
    )
    parser.add_argument("output_path", type=str, help="path where to store dataset")
    parser.add_argument(
        "image_path",
        type=str,
        help="filepath iamges downloaded (and uncompressed) ",
    )
    parser.add_argument(
        "mask_path",
        type=str,
        help="filepath masks downloaded (and uncompressed) ",
    )
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    rgb_filepaths = os.path.expanduser(args.image_path)
    label_filepaths = os.path.expanduser(args.mask_path)

    rgb_filepaths = get_files_by_extension(
        rgb_filepaths,
        extension=".jpg",
        flat_structure=True,
        recursive=True,
    )

    label_filepaths = get_files_by_extension(
        label_filepaths,
        extension=".png",
        flat_structure=True,
        recursive=True,
    )

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    label_filepaths = [
        fp for fp in label_filepaths if os.path.basename(fp).find(".png") > -1
    ]

    def get_basename(fp):
        # e.g. berlin_000000_000019_camera.json -> berlin_000000_000019
        return os.path.basename(fp).split(".")[0]

    rgb_basenames = [get_basename(f) for f in rgb_filepaths]
    label_basenames = [get_basename(f) for f in label_filepaths]

    print("Found {} rgb files".format(len(rgb_basenames)))
    print("Found {} label files".format(len(label_basenames)))

    # take union of rgb and label basenames
    basenames = set(rgb_basenames).intersection(set(label_basenames))
    print("Found {} common files".format(len(basenames)))

    filelists = {
        s: {
            "rgb": [],
            "labels_33": [],
            "labels_19": [],
        }
        for s in Bdd100kBase.SPLITS
    }

    rgb_filepaths_by_basename = {get_basename(fp): fp for fp in rgb_filepaths}
    label_filepaths_by_basename = {get_basename(fp): fp for fp in label_filepaths}

    assert len([fp for fp in rgb_filepaths_by_basename if fp in basenames]) == len(
        basenames
    )
    assert len([fp for fp in label_filepaths_by_basename if fp in basenames]) == len(
        basenames
    )

    # copy rgb images
    print("Copying rgb files")
    for rgb_fp in tqdm(basenames):
        rgb_fp = rgb_filepaths_by_basename.get(rgb_fp, None)
        if rgb_fp is None:
            print(f"Warning: RGB file {rgb_fp} not found, skipping.")
            continue
        basename = os.path.basename(rgb_fp)
        subset = os.path.basename(os.path.dirname(rgb_fp))
        subset = "valid" if subset == "val" else subset

        dest_path = os.path.join(args.output_path, subset, Bdd100kBase.RGB_DIR)
        os.makedirs(dest_path, exist_ok=True)

        # print(rgb_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(rgb_fp, os.path.join(dest_path, basename))
        filelists[subset]["rgb"].append(basename)

    for l_fp in tqdm(basenames):
        l_fp = label_filepaths_by_basename.get(l_fp, None)
        if l_fp is None:
            print(f"Warning: Label file {l_fp} not found, skipping.")
            continue
        basename = os.path.basename(l_fp)
        subset = os.path.basename(os.path.dirname(l_fp))
        subset = "valid" if subset == "val" else subset

        # load label with 1+19 classes
        label_full = cv2.imread(l_fp, cv2.IMREAD_UNCHANGED)
        label_full[label_full == -1]
        label_full += 1

        # full: 1+19 classes (original label file -> just copy file)
        dest_path = os.path.join(args.output_path, subset, Bdd100kBase.LABELS_DIR)
        os.makedirs(dest_path, exist_ok=True)
        # print(l_fp, '->', os.path.join(dest_path, basename))
        img = Image.fromarray(np.asarray(label_full, dtype="uint8"))
        img.save(os.path.join(dest_path, basename), "PNG")
        filelists[subset]["labels_19"].append(basename)

        # full: 1+19 classes colored
        dest_path = os.path.join(
            args.output_path, subset, Bdd100kBase.LABELS_COLORED_DIR
        )
        os.makedirs(dest_path, exist_ok=True)
        save_indexed_png(
            os.path.join(dest_path, basename),
            label_full,
            colormap=Bdd100kBase.CLASS_COLORS,
        )

    # ensure that filelists are valid and faultless
    def get_identifier(filepath):
        return os.path.basename(filepath).split(".")[0]

    n_samples = 0
    for subset in Bdd100kBase.SPLITS:
        identifier_lists = []
        for filelist in filelists[subset].values():
            identifier_lists.append([get_identifier(fp) for fp in filelist])

        # assert all(l == identifier_lists[0] for l in identifier_lists[1:])
        n_samples += len(identifier_lists[0])

    # assert n_samples == 5000

    # save meta files
    print("Writing meta files")
    np.savetxt(
        os.path.join(output_path, "class_names_1+19.txt"),
        Bdd100kBase.CLASS_NAMES,
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        os.path.join(output_path, "class_colors_1+19.txt"),
        Bdd100kBase.CLASS_COLORS,
        delimiter=",",
        fmt="%s",
    )

    for subset in Bdd100kBase.SPLITS:
        subset_dict = filelists[subset]
        for key, filelist in subset_dict.items():
            np.savetxt(
                os.path.join(output_path, f"{subset}_{key}.txt"),
                filelist,
                delimiter=",",
                fmt="%s",
            )
