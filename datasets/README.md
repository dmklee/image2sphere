Run all commands in the `/datasets` folder.

## ModelNet10-SO(3)
Run python script to download and convert to npz files
```
python prepare_modelnet10.py
```

## PASCAL3D+
Download Pascal3d release 1.1
```
wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
```
Download RenderForCNN (requires ImageNet account). This is only needed for training, not 
evaluation.
```
https://shapenet.cs.stanford.edu/media/syn_images_cropped_bkg_overlaid.tar
```

## SYMSOL
Download dataset with this command:
```
wget 'https://storage.googleapis.com/gresearch/implicit-pdf/symsol_dataset.zip'
```
