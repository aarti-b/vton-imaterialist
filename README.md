# vton-imaterialist
A python package for Binary Segmentation DataSet ( vton_plus and imaterialist [topwear])


# Install package 

## Installation with pypi
```
pip3 install <Inprocess >
```
## Installation from source

```
git clone https://github.com/aarti-b/vton-imaterialist
python3 setup.py install
```

# Usage Guide

```
from vtonimat import SegData
images, labels = SegData().load_training()
```

Load by batches. Following command returns list of batches. Batch size is input parameter in method `load_training_in_batches`. 

```
from vton import SegData
images, labels = SegData().load_training_in_batches(1000)
```


