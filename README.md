# ECE740 Final Project

## UniAED: Unified Adversarial Examples Detection Using Anomaly Detection for Multiple Classes

A project that aims to detect adversarial examples in CIFAR-10 and CIFAR-100 datasets using anomaly detection techniques.

Gourp member: Juxin Fa 1786677 Tong Liu 1584195 Zihao Huang 1779842

## Purpose

The purpose of the project is to find ResNet-18 classification accuracy on both CIFAR-10 and CIFAR-100, AA attack rate on the pre-trained ResNet-18 and AA adversarial samples detection rate. The anomaly detection model is based on Official PyTorch Implementation of [A Unified Model for Multi-class Anomaly Detection](https://arxiv.org/abs/2206.03687), Accepted by NeurIPS 2022 Spotlight.

## Auto Attack adversarial samples detection rate Process

We decided to use the model as the UniAD could be implemented under n-vs-rest datasets, which is more efficient and visual to train and test the model. During training, we will use the clean CIFAR-10 trainingset with labeled 0-9. During testing, we will use the combination of clean CIFAR-10 testingset with labeled 0-9 and adversarial sample labeled 10. In this way, the model can detect the Auto Attack adversarial samples. Same as CIFAR-100.

## Directory Structure

### 0. AA samples
* `10_model`: pretrained ResNet18 on CIFAR-10 and AA samples. 
* `100_model`: ResNet18 on CIFAR-100 and AA samples. 

Please check `readme_cifar10` and `readme_cifar100` first. The trained ResNet18 models are also available [here](https://www.kaggle.com/datasets/jaxonlaw/resnet18-on-cifar)

### 1.experiment

eval_torch and train_torch are used to run the model

### 2.data

it is the repository of CIFAR-10 and CIFAR-100 dataset

### 3.models

folder with the main structure of the model

### 4.tools

train_val(UniAD\tools) implement training and testing code

### 5.utils

evaluation helper

### 6.dataset

cifar-10-batches-py(\UniAD\data\CIFAR-10\cifar-10-batches-py) is used to combine the clean dataset and adversarial sample.

cifar_dataset(\UniAD\datasets\custom_dataset)is used to load CIFAR-10 in the training and testing

custom_dataset(\UniAD\dataset\custom_datasets)is used to load CIFAR-100 in the training and testing

## Instruction

## 1. CIFAR-10 dataset directory


Download the CIFAR-10 and CIFAR-100 dataset from [here](http://www.cs.toronto.edu/~kriz/cifar.html). Unzip the file and move some to `./data/CIFAR-10/` and `./data/CIFAR-100/` seperately. 

## 2. cd experiment 

`cd ./experiments/CIFAR-10/01234/` is the original folder which means the author regards label 0,1,2,3,4 as the normal label and others are anormaly.

If you want to do anomaly detection on the auto attack adversarial sample of CIFAR-10, you need to use the instruction `cd ./experiments/CIFAR-10/02468/`.Then used combined_method2.py under `./data/CIFAR-10/cifar-10-batches-py/` to combine the clean CIFAR-10 testset and adversarial sample generated from step2 and generate a new dataset. Fiannly, implement the name into `test_list` of CIFAR-10 dataset where `ECE740/datasets/cifar_dataset.py/`

If you want to do anomaly detection on the auto attack adversarial sample of CIFAR-100, you need to use the instruction `cd ./experiments/CIFAR-10/13579/`.Then used combined_method2.py under `./data/CIFAR-100/cifar-100-python/` to combine the clean CIFAR-100 testset and adversarial sample generated from step2 and generate a new dataset. Fiannly, implement the name into `test_list` of CIFAR-10 dataset where `ECE740/datasets/custom_dataset.py/`

## 3. training and testing

For torch.distributed.launch:  `sh train_torch.sh #NUM_GPUS #GPU_IDS` or `sh eval_torch.sh #NUM_GPUS #GPU_IDS`.
