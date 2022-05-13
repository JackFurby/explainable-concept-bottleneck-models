# Explainable Concept Bottleneck Models

This repository contains code and scripts for the paper: Explaining Concept Bottleneck Models with Layer-wise Relevance Propagation.


## Setup

### Dataset

Experiments detailed in this reporitory use the dataset [Caltech-UCSD Birds-200-2011 (CUB)](http://www.vision.caltech.edu/datasets/cub_200_2011/).

1. Download the dataset from [http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/)
2. Unzip and place in the directory `dataset/CUB/data`
3. Reorganise the dataset structure to be as follows inside the data directory:
```
  |
  |-attributes
  |  |-attributes.txt
  |  |-certainties.txt
  |  |-class_attribute_labels_continuous.txt
  |  |-image_attribute_labels.txt
  |-images
  |  |-001.Black_footed_Albatross
  |  |  |-Black_Footed_Albatross_0001_796111.jpg
  |  ...
  |-parts
  |  |-part_click_locs.txt
  |  |-part_locs.txt
  |  |-parts.txt
  |-segmentations
  |  |-001.Black_footed_Albatross
  |  |  |-Black_Footed_Albatross_0001_796111.png
  |  ...
  |-bounding_boxes.txt
  |-classes.txt
  |-image_class_labels.txt
  |-images.txt
  |-README
  |-train_test_split.txt
```
4. Add `attributes.txt`, `test.pkl`, `train.pkl` and `val.pkl` to the directory `dataset/CUB/dataset_splits/CBM_dataset_split`. These files can be found [here]().


### Python and packages

It is recommended you use an environment management. Conda was used for these experiments but this repository does not limit to any one option.

Python version: `3.8.12`

1. run `pip install -r requirements.txt`


### Models

This repository does not train models. To reproduce the models from the paper please see [https://github.com/JackFurby/VGG-Concept-Bottleneck](https://github.com/JackFurby/VGG-Concept-Bottleneck). The models can also be downloaded from [here](). Models should be saved as a state_dict.

1. Place the models in the directory `models/state_dict`


## Usage

Each of the results have been added to individual Jupyter Lab files. Please refer to each file for a full breakdown of the function.

The files `generate_results.py` and `pointing_game.py` will perform an operation on the entire dataset split given. `generate_results.py` repeat the same operations found in the Jupyter lab files and `pointing_game.py` runs a modification to the pointing game, caculating the average distance in pixels from the most salient point and the ground truth point from the dataset.


### generate_results.py

```
arguments
=========
  -h, --help            show this help message and exit
  --model MODEL         XtoC or joint model path
  --rules RULES [RULES ...]
                        LRP rules defined in TorchLRP
  --use_sigmoid         Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model
  --use_relu            Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model
  --use_torchexplain    use the package and convert model to work with torchexplain
  --expand_dim EXPAND_DIM
                        Size of middle layer in CtoY model. Default: 0
  --n_class_attr N_CLASS_ATTR
                        whether attr prediction is a binary or triary classification. Default: 2
  --n_concepts N_CONCEPTS
                        number of concepts. Default: 112
  --n_classes N_CLASSES
                        number of concepts. Default: 200
  --output OUTPUT       output directory path
  --mode {composite,singlar,IG,CtoY,conceptImportance}
                        mode to run script. Default: composite
  --samples_per_class SAMPLES_PER_CLASS
                        Max number of samples to generate results for per class in the dataset. Default: 10
```

Generate composite LRP saliency maps for input to concept vector  
`python generate_results.py --model ./models/state_dict/Joint0.01Model__Seed1.pth --output ./results/ --mode composite --rules alpha1beta0 epsilon gradient`

Generate singlar LRP saliency maps for input to concept vector  
`python generate_results.py --model ./models/state_dict/Joint0.01Model__Seed1.pth --output ./results/ --mode singlar --rules alpha1beta0`

Generate IG saliency maps for input to concept vector  
`python generate_results.py --model ./models/state_dict/Joint0.01Model__Seed1.pth --output ./results/ --mode IG`

Generate concept importance results  
`TO DO`


### pointing_game.py

```
arguments
=========
  -h, --help            show this help message and exit
  --model MODEL         XtoC or joint model path
  --rules RULES [RULES ...]
                        LRP rules defined in TorchLRP
  --use_sigmoid         Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model
  --use_relu            Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model
  --use_torchexplain    use the package and convert model to work with torchexplain
  --expand_dim EXPAND_DIM
                        Size of middle layer in CtoY model. Default: 0
  --n_class_attr N_CLASS_ATTR
                        whether attr prediction is a binary or triary classification. Default: 2
  --n_concepts N_CONCEPTS
                        number of concepts. Default: 112
  --n_classes N_CLASSES
                        number of concepts. Default: 200
  --output OUTPUT       output directory path
  --mode {composite,singlar,IG,CtoY,conceptImportance}
                        mode to run script. Default: composite
  --samples_per_class SAMPLES_PER_CLASS
                        Max number of samples to generate results for per classification. -1 to run all. Default: 10
```

LRP pointing game  
``python pointing_game.py --model ./saves/CBM_paper/state_dict/Joint0.01SigmoidModel__Seed1.pth --rules alpha1beta0 epsilon gradient --output ./results/ --mode LRP --samples_per_class -1``

IG pointing game  
``python pointing_game.py --model ./saves/CBM_paper/state_dict/Joint0.01SigmoidModel__Seed1.pth --output ./results/ --mode IG --samples_per_class -1``