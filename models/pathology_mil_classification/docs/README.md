# Model Overview
A pre-trained baseline model for Multiple Instance Learning (MIL) classification from Whole Slide Images (WSI). 

## Workflow
This model is trained using implementation based on the approach in Accounting for Dependencies in Deep Learning Based Multiple Instance Learning for Whole Slide Imaging [1].

![workflow](https://github.com/Project-MONAI/tutorials/blob/main/pathology/multiple_instance_learning/mil_network.jpg?raw=true)

## Data
The dataset is from [Prostate cANcer graDe Assessment (PANDA) Challenge - 2020](https://www.kaggle.com/c/prostate-cancer-grade-assessment/) for cancer grade classification from prostate histology WSIs.

- Target: ISUP tumor grade (0-5)
- Task: Classification
- Modality: WSI / hystopathology
- Size: 11052 2D RGB images 

Please download the dataset into the /data/PandaChallenge2020 folder. If your dataset location is different please update the dataroot path in config.yaml, 
or append a commandline option --config#dataroot=/another/location to the end of your training commands.

The provided labelled data was partitioned (in datalist_panda_0.json), based on our own split, into training (8490), validation (1024) and testing (1100) datasets.

# Training configuration
This model utilized a similar approach described in [1]. The training was performed with the following:

- GPU: 4x16GB of GPU memory. 
- AMP: True

## Input
Input: 2D RGB WSI 
![mil_patches](https://github.com/Project-MONAI/tutorials/blob/main/pathology/multiple_instance_learning/mil_patches.jpg?raw=true)

## Output
Output: ISUP grade class label


# Model Performance
The model was trained with our own split, as shown in the datalist json file in config folder. 
The achieved QWK scores on the testing data are: 0.9071


## Training Performance
A graph showing the training loss over 50 epochs.  
![mil_train_loss](https://github.com/Project-MONAI/tutorials/blob/main/pathology/multiple_instance_learning/mil_train_loss.png?raw=true)

## Validation Performance
A graph showing the validation QWK over 50 epochs.  
![mil_val_loss](https://github.com/Project-MONAI/tutorials/blob/main/pathology/multiple_instance_learning/mil_val_loss.png?raw=true)
![mil_val_qw](https://github.com/Project-MONAI/tutorials/blob/main/pathology/multiple_instance_learning/mil_val_qwk.png?raw=true)


## MONAI Bundle Commands
In addition to the Pythonic APIs, a few command line interfaces (CLI) are provided to interact with the bundle. The CLI supports flexible use cases, such as overriding configs at runtime and predefining arguments in a file.

For more details usage instructions, visit the [MONAI Bundle Configuration Page](https://docs.monai.io/en/latest/config_syntax.html).

#### Execute training:
```
python -m monai.bundle run training --config_file configs/config.yaml
```

#### Multi-GPU training (recommended on 4 GPUs or more):
```
torchrun --nproc_per_node=gpu -m monai.bundle run training --config_file configs/config.yaml
```

Please note that the distributed training-related options depend on the actual running environment; For more details, please refer to [pytorch's official tutorial](https://pytorch.org/docs/stable/elastic/run.html).

#### Evaluate the trained model (Multi-GPU):
```
torchrun --nproc_per_node=gpu -m monai.bundle run evaluating --config_file configs/config.yaml
```

#### Execute inference (Multi-GPU):
```
python -m monai.bundle run inference --config_file configs/config.yaml
```

# References
[1] Andriy Myronenko, Ziyue Xu, Dong Yang, Holger Roth, Daguang Xu: "Accounting for Dependencies in Deep Learning Based Multiple Instance Learning for Whole Slide Imaging". In MICCAI (2021). https://arxiv.org/abs/2111.01556

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