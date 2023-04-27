# import torch
# import numpy as np
# import pickle

# # load clean CIFAR-10
# with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/test", "rb") as f:
#     cifar10_test = pickle.load(f, encoding="bytes")

# # load adversarialsample
# aa_data = torch.load("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/real_cifar100_L2_5000.pth")

# print("aa_data keys:", aa_data.keys())

# # transfer
# aa_images = np.concatenate([img[0].numpy() for img in aa_data['adv_complete']]).reshape(-1, 3, 32, 32)

# # label 10
# aa_labels = [10] * len(aa_images)

# # adjust
# test_images = cifar10_test[b"data"].reshape(-1, 3, 32, 32)

# # add
# cifar10_test[b"data"] = np.concatenate((test_images, aa_images), axis=0)
# cifar10_test[b"labels"].extend(aa_labels)

# # save
# with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/caifar100_L2_test_batch", "wb") as f:
#     pickle.dump(cifar10_test, f)

#     UniAD/data/CIFAR-100/cifar-100-python/test
#     UniAD/data/CIFAR-100/cifar-100-python/real_cifar100_L2_5000.pth
import torch
import numpy as np
import pickle

# load clean CIFAR-100
with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/test", "rb") as f:
    cifar100_test = pickle.load(f, encoding="bytes")

# load adversarialsample
aa_data = torch.load("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/real_cifar100_Linf_5000.pth")

print("aa_data keys:", aa_data.keys())

# transfer
aa_images = np.concatenate([img[0].numpy() for img in aa_data['adv_complete']]).reshape(-1, 3, 32, 32)

# label 10
aa_labels = [100] * len(aa_images)

# adjust
test_images = cifar100_test[b"data"].reshape(-1, 3, 32, 32)

# add
cifar100_test[b"data"] = np.concatenate((test_images, aa_images), axis=0)
cifar100_test[b"fine_labels"].extend(aa_labels)  # Change "labels" to "fine_labels" or "coarse_labels"

# save
with open("/home/jupyter-tliu13@ualberta.ca-0beb7/UniAD/data/CIFAR-100/cifar-100-python/caifar100_L2_test_batch_1", "wb") as f:
    pickle.dump(cifar100_test, f)
