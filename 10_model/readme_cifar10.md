# ResNet18 on CIFAR-10
1. The model is from https://github.com/fra31/auto-attack.git with the accuracy aounrd 94%.

    Path: `autoattack/examples/model_test.pt` Please download `model_test.pt` file first

2. `10_aa.py`is used to generate AA samples, which are stored in `aa_results`. The attack evaluations  are in `log_file.txt`

3. After generating AA samples, use `filter_aa_10.ipynb` to remvoe the failed AA samples. The filtered outputs are stored in `real_cifar10_{norm}_5000.pth`, where 5000 indicates the original size of AA samples
