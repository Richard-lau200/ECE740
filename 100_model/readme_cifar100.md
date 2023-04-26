# ResNet18 on CIFAR-100

This document is based on `README.md` in the repository mentioned below. 

The model is from https://github.com/weiaicunzai/pytorch-cifar100.git with the ccuracy aounrd 76%

1. If you want train the model by yourself, use the command below to train the model.
```bash
python train.py -net resnet18 -gpu
```
The checkpoints are stored in `checkpoint`. Normally, the weights file with the best accuracy would be written to the disk with name suffix 'best'. 
My default weights file is `18_on_100.pth`

You can est the model using test.py
```bash
$ python test.py -net resnet18 -weights path_to_weights_file
```

2. `100_aa.py` is used to generate AA samples, which are stored in `aa_results`. The attack evaluations are in `log_file.txt`

3. After generating AA samples, use `filter_aa_100.ipynb` to remvoe the failed AA samples. 
The filtered outputs are stored in `real_cifar100_{norm}_5000.pth`, where 5000 indicates the original size of AA samples
