########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase
from .bdd100k import Bdd100kBase


class Bdd100k(Bdd100kBase, DatasetBase):
    def __init__(
        self,
        data_dir=None,
        n_classes=19,
        split="train",
        with_input_orig=False,
        overfit=False,
        classes=19,
    ):
        super(Bdd100k, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        print(split)
        self._n_classes = classes
        self._split = split
        self._with_input_orig = with_input_orig
        self._cameras = ["camera1"]  # just a dummy camera name
        self.overfit = overfit

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            assert os.path.exists(data_dir)
            self._data_dir = data_dir

            # load file lists
            # if self.overfit:
            #     self.images_path = os.path.join(data_dir, "leftImg8bit", "val")
            #     self.labels_path = os.path.join(data_dir, "gtFine", "val")
            # else:
            #     self.images_path = os.path.join(data_dir, "leftImg8bit", split)
            #     self.labels_path = os.path.join(data_dir, "gtFine", split)
            def _loadtxt(filename, prefix_path):
                filepath = os.path.join(self._data_dir, filename)
                assert os.path.exists(filepath)
                paths = []
                with open(filepath, "r") as f:
                    paths = f.read().split()
                result = []
                for p in paths:
                    rp = os.path.join(data_dir, prefix_path, p)
                    result.append(rp)
                    assert os.path.exists(rp)
                return result

            self._files = {
                "rgb": _loadtxt(
                    f"{self._split}_rgb.txt", os.path.join(self._split, "rgb")
                ),
                "label": _loadtxt(
                    f"{self._split}_labels_{self._n_classes}.txt",
                    os.path.join(self._split, f"labels_{self._n_classes}"),
                ),
            }
            assert all(len(l) == len(self._files["rgb"]) for l in self._files.values())
            self.images = self._files["rgb"]
            self.labels = self._files["label"]

            # for filename in glob.iglob(self.images_path + "/**/*.*", recursive=True):
            #     self.images.append(filename)
            # for filename in glob.iglob(
            #     self.labels_path + "/**/*labelTrainIds.png", recursive=True
            # ):
            #     self.labels.append(filename)
            self.images.sort()
            self.labels.sort()
            if self.overfit:
                self.images = self.images[:16]
                self.labels = self.labels[:16]

            self._files = {}

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")
        # class names, class colors, and label directory
        self._class_names = self.CLASS_NAMES
        self._class_colors = np.array(self.CLASS_COLORS, dtype="uint8")
        self._label_dir = self.LABELS_DIR

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, filename):
        # all the other files are pngs
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def load_name(self, idx):
        return self.images[idx]

    def load_image(self, idx):
        return self._load(self.images[idx])

    def load_label(self, idx):
        label = self._load(self.labels[idx]) + 1
        return label

    def __len__(self):
        return len(self.images)
