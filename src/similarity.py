import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import utils
from model import Model

def sim(net_q, net_k, memory_data_loader, test_data_loader):
    net_q.eval()
    net_k.eval()
    similarity = torch.nn.CosineSimilarity(dim=1)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    test_feature_bank, train_feature_bank = [], []
    test_feature_bank_k, train_feature_bank_k = [], []
    test_var, train_var = [], []
    idx_09 = []
    counter = 0
    with torch.no_grad():
        # generate feature bank
        for x, target in tqdm(memory_data_loader, desc='Feature extracting'):
            if counter > 10000:
                break
            feature_list = []
            feature_list_k = []
            for data in x:
                f, _ = net_q(data.cuda(non_blocking=True))
                f_k, _ = net_k(data.cuda(non_blocking=True))
                feature_list.append(f)
                feature_list_k.append(f_k)

            cos_list = []
            cos_list_k = []
            for i in range(len(x)-1):
                for j in range(len(x)-1):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))
                    cos_list_k.append(similarity(feature_list_k[i], feature_list_k[j]))

            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            train_var.append(var)
            result = cos_list[0]
            for i in range(1,len(cos_list)):
                result += cos_list[i]
            result /= len(cos_list)

            
            result_k = cos_list_k[0]
            for i in range(1,len(cos_list_k)):
                result_k += cos_list_k[i]
            result_k /= len(cos_list_k)
                    
            if result[i] >= 0.94:
                idx_09.append(i+counter)
            # if diff[i] > 0.2 and result[i] >= 0.94:
                # idx.append(i+counter)

            train_feature_bank.append(result)
            train_feature_bank_k.append(result_k)
            counter += len(result)

        # [D, N]
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        train_feature_bank_k = torch.cat(train_feature_bank_k, dim=0).contiguous()
        train_var = torch.cat(train_var, dim=0)
        print(train_feature_bank)

        test_bar = tqdm(test_data_loader)
        counter = 0
        for x, target in test_bar:
            if counter > 10000:
                break
            feature_list = []
            for data in x:
                f, _ = net_q(data.cuda(non_blocking=True))
                feature_list.append(f)
            cos_list = []
            for i in range(len(x)-1):
                for j in range(len(x)-1):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))

            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            test_var.append(var)
            result = cos_list[0]
            for i in range(1,len(cos_list)):
                result += cos_list[i]
            result /= len(cos_list)
            test_feature_bank.append(result)
            counter += len(result)

        # [D, N]
        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()
        test_var = torch.cat(test_var, dim=0)

        if len(train_feature_bank) <= len(test_feature_bank):
            test_feature_bank = test_feature_bank[:len(train_feature_bank)]
            test_var = test_var[:len(train_var)]
        else:
            train_feature_bank = train_feature_bank[:len(test_feature_bank)]
            train_feature_bank_k = train_feature_bank_k[:len(test_feature_bank)]
            train_var = train_var[:len(test_var)]

    plt.title('var')
    labels = ['test', 'train']
    data = [test_var.to('cpu').detach().numpy().copy(), train_var.to('cpu').detach().numpy().copy()]
    plt.hist(data, 20, label=labels, stacked=False)
    plt.savefig("results/var_train.png")
    plt.close()
    plt.title('Cosine Similarity')
    labels = ['train', 'test', 'other']
    data = [train_feature_bank.to('cpu').detach().numpy().copy(), test_feature_bank.to('cpu').detach().numpy().copy(), train_feature_bank_k.to('cpu').detach().numpy().copy()]
    # plt.hist(train_feature_bank.to('cpu').detach().numpy().copy(), 40)
    # plt.hist(test_feature_bank.to('cpu').detach().numpy().copy(), 40)
    # plt.hist(train_feature_bank_k.to('cpu').detach().numpy().copy(), 40)
    plt.hist(data, 40, label=labels, stacked=False)
    plt.savefig("results/sim_train.png")
    plt.close()
    plt.title('Diff Cosine Similarity')
    plt.hist((train_feature_bank - train_feature_bank_k).to('cpu').detach().numpy().copy(), 40)
    plt.savefig("results/simdiff_train.png")
    plt.close()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate at the training start')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # wandb init
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "arch": "resnet50",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "temperature": args.temperature,
        "feature_dim": args.feature_dim,
        "queue_size": args.m
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    train_data = utils.available_dataset[args.dataset](root='data', split='train+unlabeled', transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils.available_dataset[args.dataset](root='data', split='train', transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_data = utils.available_dataset[args.dataset](root='data', split='test', transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    model_q.load_state_dict(torch.load('moco_cifar_1000.pth'))
    model_k.load_state_dict(torch.load('moco_cifar.pth'))
    # model_k.load_state_dict(torch.load('moco_stl.pth'))

    sim(model_q, model_k, memory_loader, test_loader)
    # sim(model_q, memory_loader, memory_loader)

    wandb.finish()

