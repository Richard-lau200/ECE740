import torch
import numpy as np
import pickle

# 加载干净的CIFAR-10测试集
with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-10/cifar-10-batches-py/test_batch", "rb") as f:
    cifar10_test = pickle.load(f, encoding="bytes")

# 加载被autoattack之后的adversarial样本
aa_data = torch.load("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-10/cifar-10-batches-py/aa_Linf.pth")

print("aa_data keys:", aa_data.keys())

# 将adversarial样本数据转换为所需的格式
aa_images = np.concatenate([img.numpy() for img in aa_data['adv_complete']]).reshape(-1, 3, 32, 32)

# 为对抗性样本分配标签 10
aa_labels = [10] * len(aa_images)

# 调整CIFAR-10测试集中的图像数据的形状以匹配对抗性样本
test_images = cifar10_test[b"data"].reshape(-1, 3, 32, 32)

# 将adversarial样本添加到干净的CIFAR-10测试集
cifar10_test[b"data"] = np.concatenate((test_images, aa_images), axis=0)
cifar10_test[b"labels"].extend(aa_labels)

# 保存合并后的数据集为新的test_batch文件
with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-10/cifar-10-batches-py/FirstLinf_test_batch", "wb") as f:
    pickle.dump(cifar10_test, f)
