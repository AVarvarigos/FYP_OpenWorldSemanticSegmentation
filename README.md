### Objective & Goals of Research
In this project, we aim to improve on Open-World Segmentation (OWS) on Cityscapes dataset and BDDAnomaally dataset.

We are reproducing and extending the work of _[Sodano et al.](https://arxiv.org/pdf/2403.07532)_.

Repo structure:
```
📦 
├─ .gitignore
├─ README.md
├─ fyp
│  ├─ .gitignore
│  ├─ .vscode
│  │  └─ launch.json
│  ├─ README.md
│  ├─ mavs.pickle
│  ├─ overfitall.sh
│  ├─ proprocessing_unknown_known.py
│  ├─ requirements.yml
│  ├─ scripts
│  │  └─ run_all.pbs
│  ├─ src
│  │  ├─ __init__.py
│  │  ├─ args.py
│  │  ├─ build_model.py
│  │  ├─ datasets
│  │  │  ├─ __init__.py
│  │  │  ├─ cityscapes
│  │  │  │  ├─ README.md
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ cityscapes.py
│  │  │  │  ├─ prepare_dataset.py
│  │  │  │  ├─ pytorch_dataset.py
│  │  │  │  ├─ requirements.txt
│  │  │  │  ├─ weighting_linear_1+16_val.pickle
│  │  │  │  ├─ weighting_linear_1+17_val.pickle
│  │  │  │  ├─ weighting_linear_1+19_train.pickle
│  │  │  │  ├─ weighting_linear_1+19_val.pickle
│  │  │  │  ├─ weighting_linear_1+19_valid.pickle
│  │  │  │  └─ weighting_median_frequency_1+19_train.pickle
│  │  │  └─ dataset_base.py
│  │  ├─ losses
│  │  │  ├─ __init__.py
│  │  │  ├─ ce_loss.py
│  │  │  ├─ contrastive_loss.py
│  │  │  ├─ dice_loss.py
│  │  │  ├─ focal_loss.py
│  │  │  ├─ objectosphere_loss.py
│  │  │  └─ ow_loss.py
│  │  ├─ models_v2
│  │  │  ├─ __init__.py
│  │  │  ├─ context_modules.py
│  │  │  ├─ decoder.py
│  │  │  ├─ model.py
│  │  │  ├─ model_utils.py
│  │  │  ├─ neck.py
│  │  │  ├─ resnet.py
│  │  │  └─ tru_for_decoder.py
│  │  ├─ prepare_data.py
│  │  ├─ preprocessing.py
│  │  └─ utils.py
│  ├─ start.sh
│  ├─ train.py
│  └─ vars.pickle
├─ scripts
│  ├─ downlioad_cityscapes.sh
│  ├─ run.sh
│  └─ run_all.pbs
└─ vars.pickle
```

### Original Model
We implement the Encoder-Decoder architecture defined in the ContMAV paper. We initially use ResNet34 and train our model for 500 epochs using a learning rate of 0.004, as in the paper, on CityScapes Dataset.
The cityscapes data is a large dataset of street-level images with pixel-level annotations for 19 classes. The dataset is split into training, validation, and test sets.

### Tru For
We aim to improve the performance of the model by using a different architecture. We add an additional decoder to the model from [Guillaro et al.](https://arxiv.org/pdf/2212.10957) to the model and combine the logits from both decoders.

### Different Loss Functions
We also try using different loss functions:
- Focal Loss
- Focal Loss with Dice Loss
- Infonce Loss for Ananomaly Detection

### Infonce Loss
We separate crops from the image, extracting the known and unknown classes. We then use the Infonce loss to train the model to distinguish between the known and unknown classes. The Infonce loss is defined as:
```python
def infonce_loss(known, unknown):
    known = F.normalize(known, dim=1)
    unknown = F.normalize(unknown, dim=1)
    logits = torch.mm(known, unknown.t())
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss
```
This method allows for the model to learn to distinguish between the known and unknown classes, improving the performance of the model on the OWS task. Especially since the samples of unknown classes are usually not balanced.

### Synco
From the synco paper, we utilise the idea of generating hard negatives from the known classes. 

Which is better?:
- We use the known classes to generate hard negatives for the unknown classes. This allows the model to learn to distinguish between the known and unknown classes, improving the performance of the model on the OWS task.
- We also use the unknown classes to generate hard negatives for the known classes. This allows the model to learn to distinguish between the known and unknown classes, improving the performance of the model on the OWS task.

