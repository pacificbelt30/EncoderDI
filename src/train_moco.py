import argparse

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os

import utils
from model import Model


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _ in train_bar:
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        _, query = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key = encoder_k(x_k[idx])
        key = key[torch.argsort(idx)]

        score_pos = torch.bmm(query.unsqueeze(dim=1), key.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg = torch.mm(query, memory_queue.t().contiguous())
        # [B, 1+M]
        out = torch.cat([score_pos, score_neg], dim=-1)
        # compute loss
        loss = F.cross_entropy(out / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate (epochs/2) model if True')
    parser.add_argument('--per_epochs', default=100, type=int, help='Number of models to be saved every what epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning Rate at the training start')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='Weight Decay')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # create output directory
    if not os.path.exists('results'):
        os.mkdir('results')

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
    if args.dataset == 'stl10':
        # train_data = utils.STL10Pair(root='data', split='train+unlabeled', transform=utils.train_transform, download=True)
        train_data = utils.STL10Pair(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True)
        memory_data = utils.STL10Pair(root='data', split='train', transform=utils.stl_test_transform, download=True)
        test_data = utils.STL10Pair(root='data', split='test', transform=utils.stl_test_transform, download=True)
    elif args.dataset == 'cifar10':
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        train_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.train_transform, download=True)
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.test_transform, download=True)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.test_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                  drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


    # model setup and optimizer config
    model_q = Model(feature_dim).cuda()
    model_k = Model(feature_dim).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # c as num of train class
    c = len(memory_data.classes)
    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs)
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test(model_q, memory_loader, test_loader)
        test_acc_1, test_acc_5 = utils.test_encoder(model_q, memory_loader, test_loader, epoch, epochs, k, temperature, c)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics to wandb
        wandb.log({'train_loss': train_loss, 'test_acc@1': test_acc_1, 'test_acc@5': test_acc_5})
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model_q.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if (epoch%args.per_epochs) == 0 and args.save_intermediate:
            save_name_intermediate = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epoch)
            torch.save(model_q.state_dict(), 'results/{}_model.pth'.format(save_name_intermediate))
            wandb.save('results/{}_model.pth'.format(save_name_intermediate))

    # wandb.save('results/{}_model.pth'.format(save_name_intermediate))
    wandb.save('results/{}_model.pth'.format(save_name_pre))
    wandb.save('results/{}_statistics.csv'.format(save_name_pre))
    wandb.finish()

