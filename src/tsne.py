import argparse
import os
import random
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10, CIFAR100, STL10
import utils
from model import Model, Classifier, StudentModel
from quantize import load_quantize_model

def seed_check(img, is_train: bool = True):
    fig, axes = plt.subplots(3, 3, tight_layout=True)
    # fig, axes = plt.subplots(2, 5, tight_layout=True)
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")
    axes[2, 0].axis("off")
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    axes[0, 0].imshow(utils.inv_transform(img[0]).permute(1, 2, 0))
    axes[0, 1].imshow(utils.inv_transform(img[1]).permute(1, 2, 0))
    axes[0, 2].imshow(utils.inv_transform(img[2]).permute(1, 2, 0))
    axes[1, 0].imshow(utils.inv_transform(img[3]).permute(1, 2, 0))
    axes[1, 1].imshow(utils.inv_transform(img[4]).permute(1, 2, 0))
    axes[1, 2].imshow(utils.inv_transform(img[5]).permute(1, 2, 0))
    axes[2, 0].imshow(utils.inv_transform(img[6]).permute(1, 2, 0))
    axes[2, 1].imshow(utils.inv_transform(img[7]).permute(1, 2, 0))
    axes[2, 2].imshow(utils.inv_transform(img[8]).permute(1, 2, 0))
    if is_train:
        fig.savefig('results/seed_check_train.png')
    else:
        fig.savefig('results/seed_check_test.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--classes', default=10, type=int, help='the number of classes')
    parser.add_argument('--cls_dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--enc_dataset', default='stl10', type=str, help='Pre-Training Dataset (e.g. CIFAR10, STL10)')
    parser.add_argument('--model', default='classifier', type=str, help='model name')
    parser.add_argument('--ssl_method', default='moco', type=str, help='ssl method name')
    parser.add_argument('--seed', default=42, type=int, help='specify static random seed')
    parser.add_argument('--model_path', type=str, default='results/128_4096_0.5_0.999_200_256_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--wandb_model_runpath', default='', type=str, help='the runpath if using a model stored in WandB')
    parser.add_argument('--wandb_project', default='default_project', type=str, help='WandB Project name')
    parser.add_argument('--wandb_run', default='default_run', type=str, help='WandB run name')
    parser.add_argument('--is_encoder', action='store_true', help='is model encoder?')
    parser.add_argument('--is_modification', action='store_true', help='is model modificated?')
    parser.add_argument('--model_modification_method', default='', type=str, help='is is_modification, specify model modification method(e.g. quant, prune, distill)')
    parser.add_argument('--use_thop', action='store_true', help='is loaded model using thop?')
    parser.add_argument('--is_argument', action='store_true', help='is test image argumented?')

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size

    # create output directory
    if not os.path.exists('results'):
        os.mkdir('results')

    # wandb init
    config = {
        "arch": "resnet34",
        "cls_dataset": args.cls_dataset,
        "enc_dataset": args.enc_dataset,
        "model": args.model,
        "ssl_method": args.ssl_method,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "is_encoder": args.is_encoder,
        "is_modification": args.is_modification,
        "modification_method": args.model_modification_method,
        "is_argument": args.is_argument,
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    query_number = 10
    if args.cls_dataset == 'stl10':
        train_data = utils.STL10NAug(root='data', split='train', transform=utils.stl_train_transform, download=True, n=query_number)
        train_data.set_mia_train_dataset_flag(True)
        test_data = utils.STL10NAug(root='data', split='test', transform=utils.stl_train_transform, download=True, n=query_number)
        test_data.set_mia_train_dataset_flag(True)
        no_arg_train_data = STL10(root='data', split='train', transform=utils.stl_test_ds_transform, download=True)
        no_arg_test_data = STL10(root='data', split='test', transform=utils.stl_test_ds_transform, download=True)
    elif args.cls_dataset == 'cifar10':
        train_data = utils.CIFAR10NAug(root='data', train=True, transform=utils.train_transform, download=True, n=query_number)
        test_data = utils.CIFAR10NAug(root='data', train=False, transform=utils.train_transform, download=True, n=query_number)
        no_arg_train_data = CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
        no_arg_test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
    else:
        train_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=query_number)
        test_data = utils.CIFAR100NAug(root='data', train=False, transform=utils.train_transform, download=True, n=query_number)
        no_arg_train_data = CIFAR100(root='data', train=False, transform=utils.train_transform, download=True)
        no_arg_test_data = CIFAR100(root='data', train=False, transform=utils.test_transform, download=True)
    shuffle=False
    arg_loader = DataLoader(arg_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    no_arg_loader = DataLoader(no_arg_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

    # model setup and optimizer config
    model_path = ''
    if args.wandb_model_runpath != '':
        import os
        if os.path.exists(args.model_path):
            os.remove(args.model_path)
        base_model = wandb.restore(args.model_path, run_path=args.wandb_model_runpath)
        model_path = base_model.name

    device = 'cuda'
    # model setup and optimizer config
    if args.is_encoder:
        model = Model(args.feature_dim).cuda()
        model.load_state_dict(torch.load(model_path), strict=not args.use_thop)
    elif args.is_modification and args.model_modification_method == 'quant':
        model = Classifier(args.classes).cpu()
        model = load_quantize_model(model, model_path, torch.randn((3,4,32,32)))
        device = 'cpu'
    elif args.is_modification and args.model_modification_method == 'distill':
        model = StudentModel(args.classes, 'mobilenet_v2').cuda()
        model.load_state_dict(torch.load(model_path))
    elif args.is_modification and args.model_modification_method == 'prune':
        model = Classifier(args.classes).cuda()
        model.load_state_dict(torch.load(model_path))
    else:
        model = Classifier(args.classes).cuda()
        model.load_state_dict(torch.load(model_path), strict=not args.use_thop)

    # initialize random seed
    # Seed is set after the model is defined because random initialization of the model weights is entered
    utils.set_random_seed(args.seed)

    loss_criterion = torch.nn.CrossEntropyLoss()
    atest_loss, atest_acc, atest_acc_5 = utils.train_val(model, arg_loader, None, loss_criterion, 0, 1, device)
    ntest_loss, ntest_acc, ntest_acc_5 = utils.train_val(model, no_arg_loader, None, loss_criterion, 0, 1, device)

    # save kstest result
    wandb.log({
        "arg_test_loss": atest_loss,
        "arg_test_acc": atest_acc,
        "arg_test_acc_5": atest_acc_5,
        "no_arg_test_loss": ntest_loss,
        "no_arg_test_acc": ntest_acc,
        "no_arg_test_acc_5": ntest_acc_5,
        "diff_loss": ntest_loss - atest_loss,
        "diff_acc": ntest_acc - atest_acc,
        "diff_acc_5": ntest_acc_5 - atest_acc_5,
    })

    # wandb finish
    wandb.finish()


class T_SNE():
    def __init__(self,model:PyTorchClassifier):
        self.classifier = model

    def tsne(self,x:np.ndarray,y:np.ndarray,y_wm,layers:list):
        batch = 50
        # y = y[:len(x)//batch*batch]
        activation = np.array([])
        newy = []
        for i in tqdm(range(len(x)//batch)):
            tempx = x[i*batch:(i+1)*batch]
            all_activations = self.classifier.get_all_activations(tempx)
            if len(layers) == 0:
                layers = [i for i in range(len(all_activations))]
            activations = []
            for j,l in enumerate(layers):
                activations.append(all_activations[l])
            # activation = activations[-1].to('cpu').detach().numpy().copy()
            temp = []
            bias = 2
            for j in range(3):
                if j == 0:
                    temp = torch.flatten(activations[-j-bias],1).to('cpu').detach().numpy().copy()
                    continue
                temp = np.append(temp,torch.flatten(activations[-j-bias],1).to('cpu').detach().numpy().copy(),axis=1)
            if i == 0:
                activation = torch.flatten(activations[-bias],1).to('cpu').detach().numpy().copy()
                newy = torch.argmax(activations[-1],dim=1).to('cpu').detach().numpy().copy()
                # print(activation)
                activation = temp
                continue
            # print(torch.flatten(activations[-j-2],1).shape)
            # activation = np.append(activation,torch.flatten(activations[-bias],1).to('cpu').detach().numpy().copy(),axis=0)
            newy = np.append(newy,activations[-1].argmax(1).to('cpu').detach().numpy().copy(),axis=0)
            activation = np.append(activation,temp,axis=0)
        print(type(activation))
        print(newy[len(x)-len(y_wm):])
        print(y_wm==newy[len(x)-len(y_wm):])
        print('wm_acc:',sum(y_wm==newy[len(x)-len(y_wm):])/len(y_wm))
        
        tsne = TSNE(n_components=2, random_state=41)
        X_reduced = tsne.fit_transform(activation)
        plt.figure(figsize=(13, 7))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    c=y, cmap='Paired',
                    s=30, alpha=1.0)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.savefig('test.png')
        print('FINISH')

