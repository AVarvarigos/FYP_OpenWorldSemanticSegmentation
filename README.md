# Open-World-Semantic-Segmentation-Including-Class-Similarity
FYP
-Paper can be found here: https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/sodano2024cvpr.pdf

-Set up Environment
Download dependencies:
conda env create -f requirements.yml 
conda active

-Download Dataset Cityscapes 
https://www.cityscapes-dataset.com/downloads/

src/
├── __init__.py
├── args.py
├── build_model.py
├── contrastive_loss.py
├── prepare_data.py
├── preprocessing.py
├── utils.py
├── datasets/
│   ├── __init__.py
│   ├── dataset_base.py
│   └── cityscapes/
│       ├── __init__.py
│       ├── cityscapes.py
│       ├── prepare_dataset.py
│       ├── pytorch_dataset.py
│       ├── requirements.txt
│       ├── weighting_linear_1+16_val.pickle
│       ├── weighting_linear_1+17_val.pickle
│       ├── weighting_linear_1+19_val.pickle
│       └── weighting_median_frequency_1+19_train.pickle
└── models/
    ├── __init__.py
    ├── model.py
    └── resnet.py
    ├── decoder.py
    ├── neck.py
    └── resnet.py
├── trained_models/imagnet/
│   ├── r34_NBt1D.pth
