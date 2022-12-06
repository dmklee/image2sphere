Image to Sphere: Learning Equivariant Features for Efficient Pose Prediction
---------------------------------------------------------------------
[Paper](https://github.com/dmklee/image2sphere/blob/gh-pages/assets/paper.pdf) | [Project Page](https://dmklee.github.io/image2sphere/)

---------------------------------------------------------------------

## Installation
This code was tested with python 3.8.  You can install all necessary requirements with pip:
```
pip install -r requirements.txt
```
You may get lots of warnings from e3nn about deprecated functions. If so, run commands as `python -W ignore -m src.train ...`

## Dataset preparation
Follow instruction in `datasets/README.md`.  Make sure to run commands from 
within the `datasets` folder.

## Train on ModelNet10-SO(3)
```
python -m src.train --dataset_name=modelnet10 --encoder=resnet50_pretrained --seed=0
```
Rotation error (in radians) on the test set will be stored in `results/pascal3d-warp-synth_resnet101-pretrained_seed0/eval.npy`

To train on the limited training set (20 views per instance), run:
```
python -m src.train --dataset_name=modelnet10-limited --encoder=resnet50_pretrained --seed=0
```

## Train on SYMSOL
Here is an example for training on SYMSOL I with 50k views per instance
```
python -m src.train --dataset_name=symsolI-50000 --encoder=resnet50_pretrained --seed=0
```
Average log likelihood on the test set will be stored in `results/symsolI-50000_resnet50-pretrained_seed0/eval_log_likelihood.npy`

You can adjust the number of views (`--dataset_name=symsolI-10000` will use 10k views per instance) or
train on SYMSOL II objects (`--dataset_name=symsolII-50000` will train on sphX; `--dataset_name=symsolIII-50000` will train on cylO; `--dataset_name=symsolIIII-50000` will train on tetX).  We train a single model on all of SYMSOL I, but separate models for each object from SYMSOL II.

## Train on PASCAL3D+
```
python -m src.train --dataset_name=pascal3d-warp-synth --encoder=resnet101_pretrained --seed=0
```
Rotation error (in radians) on the test set will be stored in `results/pascal3d-warp-synth_resnet101-pretrained_seed0/eval.npy`
