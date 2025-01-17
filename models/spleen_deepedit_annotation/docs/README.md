# Description
This is a pre-trained model for 3D segmentation of the spleen organ from CT images using DeepEdit.

# Model Overview
DeepEdit is an algorithm that combines the power of two models in one single architecture.
It allows the user to perform inference, as a standard segmentation method (i.e. UNet), and also to interactively
segment part of an image using clicks [2].

DeepEdit aims to facilitate the user experience and at the same time the development of new active learning techniques.

The model was trained on 32 images and validated on 9 images.

## Data
For this example, the training dataset Task09_Spleen.tar was used. More datasets are available at http://medicaldecathlon.com/.

## Training configuration
The training could be performed with a 12GB-memory GPU.

## Input and output formats
Input: 3 channels CT image - one channel representing clicks for each segment (i.e. spleen and background and the image)

Output: 2 channels: Label 1: spleen; Label 0: everything else - This depends on the dictionary "label_names" defined in train.json and inference.json ("label_names": {"spleen": 1, "background": 0},)

## Model Performance

The achieved Dice score on the validation set is:

Spleen dice > 0.90 (depending on the number of simulated clicks)

Command to execute:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```


### Train Dice

A graph showing the train dice over 90 epochs.

![Train Dice](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_train_dice.png)


### Train loss

A graph showing the training loss over 90 epochs.

![Train loss](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_train_loss.png)


### Validation Dice

A graph showing the validation dice over 90 epochs.

![Validation Dice](https://developer.download.nvidia.com/assets/Clara/Images/monai_spleen_deepedit_annotation_val_dice.png)


## commands example
Execute training:

```
python -m monai.bundle run training --meta_file configs/metadata.json --config_file configs/train.json --logging_file configs/logging.conf
```

Override the `train` config to execute multi-GPU training:

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 -m monai.bundle run training --meta_file configs/metadata.json --config_file "['configs/train.json','configs/multi_gpu_train.json']" --logging_file configs/logging.conf
```

Please note that the distributed training related options depend on the actual running environment, thus you may need to remove `--standalone`, modify `--nnodes` or do some other necessary changes according to the machine you used.
Please refer to [pytorch's official tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more details.

Override the `train` config to execute evaluation with the trained model:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file "['configs/train.json','configs/evaluate.json']" --logging_file configs/logging.conf
```

Execute inference:

```
python -m monai.bundle run evaluating --meta_file configs/metadata.json --config_file configs/inference.json --logging_file configs/logging.conf
```

# References
[1] Diaz-Pinto, Andres, et al. DeepEdit: Deep Editable Learning for Interactive Segmentation of 3D Medical Images. MICCAI Workshop on Data Augmentation, Labelling, and Imperfections. MICCAI 2022.

[2] Diaz-Pinto, Andres, et al. "MONAI Label: A framework for AI-assisted Interactive Labeling of 3D Medical Images." arXiv preprint arXiv:2203.12362 (2022).

[3] Sakinis, Tomas, et al. "Interactive segmentation of medical images through fully convolutional neural networks." arXiv preprint arXiv:1903.08205 (2019).

# License
Copyright (c) MONAI Consortium

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
