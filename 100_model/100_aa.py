import os
import argparse
from pathlib import Path
import warnings
from conf import settings
from utils import get_test_dataloader

import sys
sys.path.insert(0,'..')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from autoattack import AutoAttack
from models.resnet import resnet18

######################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-weights', type=str, default='18_on_100.pth')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--n_ex', type=int, default=5000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./aa_results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    
    args = parser.parse_args()
 
    model = resnet18()
    ckpt = torch.load(args.weights)
    model.load_state_dict(ckpt)
    model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Set the model to evaluation mode
    model.eval()

    # Load the CIFAR-100 dataset
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TEST_MEAN,
        settings.CIFAR100_TEST_STD,
        num_workers=2,
        batch_size=args.batch_size,
    )
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack       
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in cifar100_test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in cifar100_test_loader]
    y_test = torch.cat(l, 0)


    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(
                x_test[:args.n_ex], y_test[:args.n_ex],
                return_labels=True,
                bs=args.batch_size, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_norm_{}.pth'.format(
                args.save_dir, 'aa_cifar100', args.version, args.n_ex, args.norm))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))

                
