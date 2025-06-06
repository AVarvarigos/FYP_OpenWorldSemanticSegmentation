########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import pickle
import abc

import numpy as np
from torch.utils.data import Dataset


class DatasetBase(abc.ABC, Dataset):
    def __init__(self):
        self._camera = None
        self._default_preprocessor = lambda x: x
        self.preprocessor = self._default_preprocessor

    def filter_camera(self, camera):
        assert camera in self.cameras
        self._camera = camera
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._camera = None

    @abc.abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        sample = {
            "image": self.load_image(idx),
            "label": self.load_label(idx),
            "name": self.load_name(idx),
        }

        if self.split != "train":
            # needed to compute mIoU on original image size
            sample["label_orig"] = sample["label"].copy()

        if self.with_input_orig:
            sample["image_orig"] = sample["image"].copy()
        sample = self.preprocessor(sample)

        return sample

    @property
    @abc.abstractmethod
    def cameras(self):
        pass

    @property
    @abc.abstractmethod
    def class_names(self):
        pass

    @property
    @abc.abstractmethod
    def class_names_without_void(self):
        pass

    @property
    @abc.abstractmethod
    def class_colors(self):
        pass

    @property
    @abc.abstractmethod
    def class_colors_without_void(self):
        pass

    @property
    @abc.abstractmethod
    def n_classes(self):
        pass

    @property
    @abc.abstractmethod
    def n_classes_without_void(self):
        pass

    @property
    @abc.abstractmethod
    def split(self):
        pass

    @property
    @abc.abstractmethod
    def source_path(self):
        pass

    @property
    @abc.abstractmethod
    def with_input_orig(self):
        pass

    @property
    def camera(self):
        return self._camera

    @abc.abstractmethod
    def load_image(self, idx):
        pass

    @abc.abstractmethod
    def load_label(self, idx):
        pass

    def color_label(self, label, with_void=True):
        if with_void:
            colors = self.class_colors
        else:
            colors = self.class_colors_without_void
        cmap = np.asarray(colors, dtype="uint8")

        return cmap[label]

    @staticmethod
    def static_color_label(label, colors):
        cmap = np.asarray(colors, dtype="uint8")
        return cmap[label]

    def compute_class_weights(self, weight_mode="median_frequency", c=1.02):
        assert weight_mode in ["median_frequency", "logarithmic", "linear"]

        # build filename
        class_weighting_filepath = os.path.join(
            self.source_path,
            f"weighting_{weight_mode}_" f"1+{self.n_classes_without_void}",
        )
        if weight_mode == "logarithmic":
            class_weighting_filepath += f"_c={c}"

        class_weighting_filepath += f"_{self.split}.pickle"

        if os.path.exists(class_weighting_filepath):
            class_weighting = pickle.load(open(class_weighting_filepath, "rb"))
            print(f"Using {class_weighting_filepath} as class weighting")
            return class_weighting

        print("Compute class weights")

        n_pixels_per_class = np.zeros(self.n_classes)
        n_image_pixels_with_class = np.zeros(self.n_classes)
        for i in range(len(self)):
            label = self.load_label(i) - 1
            h, w = label.shape
            current_dist = np.bincount(label.flatten(), minlength=self.n_classes)
            n_pixels_per_class += current_dist

            # For median frequency we need the pixel sum of the images where
            # the specific class is present. (It only matters if the class is
            # present in the image and not how many pixels it occupies.)
            class_in_image = current_dist > 0
            n_image_pixels_with_class += class_in_image * h * w

            print(f"\r{i+1}/{len(self)}", end="")
        print()

        # remove void
        n_pixels_per_class = n_pixels_per_class[1:]
        n_image_pixels_with_class = n_image_pixels_with_class[1:]

        if weight_mode == "linear":
            class_weighting = n_pixels_per_class

        elif weight_mode == "median_frequency":
            frequency = n_pixels_per_class / n_image_pixels_with_class
            class_weighting = np.median(frequency) / frequency

        elif weight_mode == "logarithmic":
            probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
            class_weighting = 1 / np.log(c + probabilities)

        if np.isnan(np.sum(class_weighting)):
            print(f"n_pixels_per_class: {n_pixels_per_class}")
            print(f"n_image_pixels_with_class: {n_image_pixels_with_class}")
            print(f"class_weighting: {class_weighting}")
            raise ValueError("class weighting contains NaNs")

        with open(class_weighting_filepath, "wb") as f:
            pickle.dump(class_weighting, f)
        print(f"Saved class weights under {class_weighting_filepath}.")
        return class_weighting
