from label import labels


class Bdd100kBase:
    SPLITS = ["train", "valid", "test", "val"]

    # number of classes without void/unlabeled and license plate (class 34)
    N_CLASSES = [19]

    # 1+19 classes (0: void)
    CLASS_NAMES = ["void"] + [label.name for label in labels if not label.ignoreInEval]
    CLASS_COLORS = [(0, 0, 0)] + [
        label.color for label in labels if not label.ignoreInEval
    ]

    RGB_DIR = "rgb"

    LABELS_DIR = "labels_19"
    LABELS_COLORED_DIR = "labels_19_colored"
