# Dual-Pathway DenseNets with Fully Lateral Connections for Multimodal Brain Tumor Segmentation
> This is used for the first paper. It's been reimplemented in Pytorch and the original source is implemented in Tensorflow.

## Requirements
* The code has been written in Python 3.6.3 and Pytorch 1.0.1
* Make sure to install all the libraries given in requirement.txt, and you can do so by the 
following command

```pip install -r requirement.txt```

## Dataset
[BRATS 2018](https://www.med.upenn.edu/sbia/brats2018/data.html) dataset was chosen to substantiate
our proposed method. It contains the 3D multimodal brain MRI data of 285 labeled training subjects and 
66 testing subjects which have to be submitted for online validation. We randomly split the 285 labeled
training data into training and testing set with the rate of 4:1.

## How to preprocess the dataset?
* Download the BRATS 2018 data and place it in data folder.(Visit [this](https://www.med.upenn.edu/sbia/brats2018/data.html) link to downlaod the data. You
need to register for the challenge)
* To perform bias field correction, you have to perform correction for each directory of dataset.

```
$ cd data_preprocessing
$ python n4correction Brats2018/MICCAI_BraTS_2018_Data_Training/HGG
```
## How to train and evaluate our proposed DP-DenseNets?

```$ python train_valid_score.py --id=FullyHierarchical```








