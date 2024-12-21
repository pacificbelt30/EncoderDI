import argparse
import os
import random
import csv
from scipy.stats import ttest_ind, kstest

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import numpy as np

from tqdm import tqdm
import wandb
import utils
from model import Model, Classifier, StudentModel
from quantize import load_quantize_model

def verify_model(model, trainloader, valloader, testloader, n_components: int=50, covariance: str='diag', encoder_flag: bool = True, device='cuda'):
    """
    Dataset Inference
    Returns:
        t-statistic: t-test statistic
        p_value: t-test p-value
        effect_size: effect size
    """

    model.eval()

    train_representations = []
    val_representations = []
    test_representations = []

    with torch.no_grad():
        for data in tqdm(trainloader):
            images, _ = data
            # representations = model(images)
            if encoder_flag:
                representations, _ = model(images.to(device, non_blocking=True))
            else:
                representations = model(images.to(device, non_blocking=True))
            train_representations.append(representations)

        for data in tqdm(valloader):
            images, _ = data
            # representations = model(images)
            if encoder_flag:
                representations, _ = model(images.to(device, non_blocking=True))
            else:
                representations = model(images.to(device, non_blocking=True))
            val_representations.append(representations)

        for data in tqdm(testloader):
            images, _ = data
            # representations = model(images)
            if encoder_flag:
                representations, _ = model(images.to(device, non_blocking=True))
            else:
                representations = model(images.to(device, non_blocking=True))
            test_representations.append(representations)

    train_representations = torch.cat(train_representations).contiguous().cpu().numpy()
    val_representations = torch.cat(val_representations).contiguous().cpu().numpy()
    test_representations = torch.cat(test_representations).contiguous().cpu().numpy()

    # create density estimator (Gaussian Mixture Model)
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance)
    gmm.fit(train_representations)

    # calculate log-likehood
    val_log_likelihood = gmm.score_samples(val_representations)
    test_log_likelihood = gmm.score_samples(test_representations)

    # verify by t-test
    t_statistic, p_value = ttest_ind(val_log_likelihood, test_log_likelihood)
    effect_size = (np.mean(val_log_likelihood) - np.mean(test_log_likelihood)) / np.std(test_log_likelihood)

    return p_value, t_statistic, effect_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DI for SSL')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--classes', default=10, type=int, help='the number of classes')
    parser.add_argument('--num_of_samples', default=2500, type=int, help='num of samples')
    parser.add_argument('--dataset', default='stl10', type=str, help='Training Dataset (e.g. CIFAR10, STL10)')
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
    parser.add_argument('--num_of_components', default=50, type=int, help='number of gmm components')
    parser.add_argument('--covariance_type', default='diag', type=str, help='covariance type of gmm')

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size

    # initialize random seed
    utils.set_random_seed(args.seed)

    if not os.path.exists('results'):
        os.mkdir('results')

    # wandb init
    config = {
        "arch": "resnet34",
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "num_of_samples": args.num_of_samples,
        "seed": args.seed,
        "is_encoder": args.is_encoder,
        "is_modification": args.is_modification,
        "modification_method": args.model_modification_method,
        "components": args.num_of_components,
        "covariance": args.covariance_type,
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    if args.dataset == 'stl10':
        train_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True)
        train_data.set_mia_train_dataset_flag(True)
        test_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True)
        test_data.set_mia_train_dataset_flag(False)
    elif args.dataset == 'cifar10':
        # train_data = utils.CIFAR10NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        # test_data = utils.CIFAR10NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
        train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=utils.train_transform, download=True)
        test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=utils.train_transform, download=True)
    else:
        # train_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=10)
        # test_data = utils.CIFAR100NAug(root='data', train=False, transform=utils.train_transform, download=True, n=10)
        train_data = torchvision.datasets.CIFAR100(root='data', train=True, transform=utils.train_transform, download=True)
        test_data = torchvision.datasets.CIFAR100(root='data', train=False, transform=utils.train_transform, download=True)

    # random split train and test
    train_size = int(0.5 * len(train_data))
    val_size = len(train_data) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size], generator)

    shuffle=True
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=8)

    # model setup and optimizer config
    model_path = ''
    if args.wandb_model_runpath != '':
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


    # Verify model by GMM
    n_components = args.num_of_components
    covariance_type = args.covariance_type
    """
    Classifier:
        n_components: 10
        covariance: full
    Encoder:
        n_components: 50
        covariance: diag
    """
    pvalue, statistic, effect_size = verify_model(model, train_loader, val_loader, test_loader, n_components, covariance_type, args.is_encoder, device)

    # 結果の出力
    print(f'p-value: {pvalue}')
    print(f'statistic: {statistic}')
    print(f'effect-size: {effect_size}')

    # save kstest result
    wandb.log({'pvalue': pvalue, 'statistic': statistic, 'effect_size': effect_size})
    # wandb finish
    os.remove(os.path.join(wandb.run.dir, args.model_path))
    wandb.finish()

