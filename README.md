# Leveraging Self-Supervised Training for Unintentional Action Recognition (ECCVW 2022)

This is the official representation of [Leveraging Self-Supervised Training for Unintentional Action Recognition](https://arxiv.org/pdf/2209.11870.pdf).

[Project](https://dukaenea.github.io/ssl_uar/)

Please cite as below in case you use the code.
```
@inproceedings{duka2022leveraging,
  title={Leveraging Self-Supervised Training for Unintentional Action Recognition},
  author={Duka, Enea and Kukleva, Anna and Schiele, Bernt},
  booktitle={European Conference on Computer Vision Workshop SSLWIN (ECCVW)},
  year={2022},
  organization={Springer}
}
```

## Running the code
Follow the steps below to run the code.

### Dataset download.
Download the [Oops!](https://oops.cs.columbia.edu/data/) dataset from the official site. Note that the dataset is about 45GB in size. Unzip the 
dataset resources in the ```datasets/oops/``` folder of the project.

### Extracting features.
You can extract features using different pretrained image or video models. Run the following scripts to extract features using the respective models.
- ViT pretrained on ImageNet 21K: ```feat_extract/vit_feat_extract```
- ResNet50 pretrained on ImageNet 1K: ```feat_extract/resnet_feat_extract```
- R(2+1)D pretrained on Kinetics400: ```feat_extract/r2plus1d_feat_extract```

The features extracted will be saved in ```resources/data/features/{model_name}```

### Representation learning.
There are two stages of representation learning to be performed.

#### Frame2Clip (F2C)
In this stage we learn local features in a self supervised way. Run the script ```rep_learning/main.py``` using the config file ```configs/default_rep_learning.yml```
Make sure the ```multi_scale``` option in the config file is set to ```false```.

#### Frame2Clip2Video (F2C2V)
In this stage we further refine the features by adding global video information. This stage builds on top of F2C by fine tuning the model from that stage.
Run the script ```rep_learning/main.py``` using the config file ```configs/default_rep_learning.yml```. Change ```multi_scale``` to ```true``` and provide the
path of the model saved in the last stage as the ```vtn_ptr_path``` option.

### Downstream Tasks
The downstream tasks the model is finetuned on are as follow.
#### Unintentional Action Classification
We use the representations learned during F2C or F2C2V to learn unintentional action classification. To run with single scale, run ```action_classification/main.py``` using the config
file ```configs/default.yml```. Set ```vtn_ptr_path``` to the path of the saved model during F2C and ```multi_scale``` to ```false```.
To run with multiple scales, run ```transformer2x/main.py```. Set ```vtn_ptr_path``` to the path of the saved model during F2C2V and 
```multi_scale``` to ```true```.

#### Unintentional Action Localization
We use the classification model and validate it on localization. Run ```action_localization/main_al.py``` using the config
file ```configs/default.yml```. Set ```vtn_ptr_path``` to the path of the saved model during unintentional action classification.

#### Unintentional Action Anticipation
This task is identical to unintentional action classification with the sole difference being the value for the ```anticipate_label``` option which should
be set to the number of seconds in the future when we anticipate the action.
