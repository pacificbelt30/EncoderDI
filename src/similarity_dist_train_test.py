import argparse
import os
import random
import csv
from scipy.stats import kstest, mannwhitneyu

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

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

def sim(model, memory_data_loader, test_data_loader, num_of_samples:int=500, encoder_flag:bool=True, device:str='cuda'):
    model.eval()
    test_method = kstest
    # test_method = mannwhitneyu
    similarity = torch.nn.CosineSimilarity(dim=1)
    test_feature_bank, train_feature_bank = [], []
    test_cos_list, train_cos_list = [], []
    test_mean, train_mean = [], []
    test_var, train_var = [], []
    test_median, train_median = [], []
    test_max, train_max = [], []
    test_min, train_min = [], []
    counter = 0

    with torch.no_grad():
        # generate feature bank
        for x, target in tqdm(memory_data_loader, desc='Feature extracting'):
            target = target.to(device)
            if counter > num_of_samples:
                break
            if counter == 0:
                img = [img[1] for img in x]
                seed_check(img, True)
            feature_list = []
            for data in x:
                if encoder_flag:
                    f, _ = model(data.to(device, non_blocking=True))
                else:
                    f = model(data.to(device, non_blocking=True))
                feature_list.append(f)

            cos_list = []
            for j in range(len(x)):
                for i in range(len(x)):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))

            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            train_var.append(var)
            median = torch.median(torch.stack(cos_list, dim=1), dim=1)[0] # return median_type known as named_tuple
            train_median.append(median)
            max = torch.max(torch.stack(cos_list, dim=1), dim=1)[0]
            train_max.append(max)
            min = torch.min(torch.stack(cos_list, dim=1), dim=1)[0]
            train_min.append(min)

            # mean = cos_list[0]
            # for i in range(1,len(cos_list)):
                # mean += cos_list[i]
            # mean /= len(cos_list)
            mean = torch.mean(torch.stack(cos_list, dim=1), dim=1)
            train_mean.append(mean)

            train_feature_bank.append(torch.stack(feature_list, dim=1))
            train_cos_list.append(torch.stack(cos_list, dim=1))
            counter += len(mean)

        counter=0
        for x, target in tqdm(test_data_loader, desc='Feature extracting'):
            target = target.to(device)
            if counter > num_of_samples:
                break
            if counter == 0:
                img = [img[1] for img in x]
                seed_check(img, False)
            feature_list = []
            for data in x:
                if encoder_flag:
                    f, _ = model(data.to(device, non_blocking=True))
                else:
                    f = model(data.to(device, non_blocking=True))
                feature_list.append(f)

            cos_list = []
            for j in range(len(x)):
                for i in range(len(x)):
                    if i >= j:
                        continue
                    cos_list.append(similarity(feature_list[i], feature_list[j]))
            var = torch.var(torch.stack(cos_list, dim=1), dim=1)
            test_var.append(var)
            median = torch.median(torch.stack(cos_list, dim=1), dim=1)[0] # return median_type known as named_tuple
            test_median.append(median)
            max = torch.max(torch.stack(cos_list, dim=1), dim=1)[0]
            test_max.append(max)
            min = torch.min(torch.stack(cos_list, dim=1), dim=1)[0]
            test_min.append(min)

            # result = cos_list[0]
            # for i in range(1,len(cos_list)):
                # result += cos_list[i]
            # result /= len(cos_list)
            mean = torch.mean(torch.stack(cos_list, dim=1), dim=1)
            test_mean.append(mean)

            test_feature_bank.append(torch.stack(feature_list, dim=1))
            test_cos_list.append(torch.stack(cos_list, dim=1))
            counter += len(mean)

        # [D, N]
        train_cos_list = torch.cat(train_cos_list, dim=0).contiguous()
        test_cos_list = torch.cat(test_cos_list, dim=0).contiguous()
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()
        # train_mean = torch.cat(train_mean, dim=0).contiguous()
        # test_mean = torch.cat(test_mean, dim=0).contiguous()
        # train_var = torch.cat(train_var, dim=0)
        # test_var = torch.cat(test_var, dim=0)
        # train_median = torch.cat(train_median, dim=0)
        # test_median = torch.cat(test_median, dim=0)
        # train_max = torch.cat(train_max, dim=0)
        # test_max = torch.cat(test_max, dim=0)
        # train_min = torch.cat(train_min, dim=0)
        # test_min = torch.cat(test_min, dim=0)

        train_mean = torch.mean(train_cos_list, dim=1)
        test_mean = torch.mean(test_cos_list, dim=1)
        train_var = torch.var(train_cos_list, dim=1)
        test_var = torch.var(test_cos_list, dim=1)
        train_median = torch.median(train_cos_list, dim=1)[0]
        test_median = torch.median(test_cos_list, dim=1)[0]
        train_min = torch.min(train_cos_list, dim=1)[0]
        test_min = torch.min(test_cos_list, dim=1)[0]
        train_max = torch.max(train_cos_list, dim=1)[0]
        test_max = torch.max(test_cos_list, dim=1)[0]

    color = ['tab:blue', 'tab:orange', 'tab:green']

    train_random_sampling = random.sample(range(0, len(train_mean)), num_of_samples)
    test_random_sampling = random.sample(range(0, len(test_mean)), num_of_samples)
    olabels = ['train', 'test']
    # data = [train_mean[train_random_sampling].to('cpu').detach().numpy().copy(),test_mean[test_random_sampling].to('cpu').detach().numpy().copy()]
    data = [train_mean[:num_of_samples].to('cpu').detach().numpy().copy(),test_mean[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig(f"results/sim_test_train_model.png")
    plt.close()

    n = 2
    data = [
            torch.mean(train_cos_list[:, :n*(n-1)//2], dim=1)[:num_of_samples].to('cpu').detach().numpy().copy(),
            torch.mean(test_cos_list[:, :n*(n-1)//2])[:num_of_samples].to('cpu').detach().numpy().copy()
    ]
    ks_result = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig(f"results/sim_test_train_model_(n={n}).png")
    plt.close()

    n = 5
    data = [
            torch.mean(train_cos_list[:, :n*(n-1)//2], dim=1)[:num_of_samples].to('cpu').detach().numpy().copy(),
            torch.mean(test_cos_list[:, :n*(n-1)//2])[:num_of_samples].to('cpu').detach().numpy().copy()
    ]
    ks_result = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig(f"results/sim_test_train_model_(n={n}).png")
    plt.close()

    # data = [train_mean[train_random_sampling].to('cpu').detach().numpy().copy(),test_mean[test_random_sampling].to('cpu').detach().numpy().copy()]
    data = [train_median[:num_of_samples].to('cpu').detach().numpy().copy(),test_median[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_median = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Median Cosine Similarity')
    plt.savefig(f"results/median_test_train_model.png")
    plt.close()

    data = [train_var[:num_of_samples].to('cpu').detach().numpy().copy(),test_var[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_var = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test Varriance distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.0, 0.1), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.0, 0.1), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Variance')
    plt.savefig(f"results/var_test_train_model.png")
    plt.close()

    data = [train_min[:num_of_samples].to('cpu').detach().numpy().copy(),test_min[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_min = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test Varriance distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Minimum')
    plt.savefig(f"results/min_test_train_model.png")
    plt.close()

    data = [train_max[:num_of_samples].to('cpu').detach().numpy().copy(),test_max[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_max = test_method(data[0], data[1], alternative='two-sided', method='auto')
    # plt.title(f'{num_of_samples}_{ks_result.pvalue}')
    plt.title(f'train & test Varriance distribution, {num_of_samples} samples')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Maximum')
    plt.savefig(f"results/max_test_train_model.png")
    plt.close()

    data = [train_mean.to('cpu').detach().numpy().copy(),test_mean.to('cpu').detach().numpy().copy()]
    ks_result_all = test_method(train_mean.to('cpu').detach().numpy().copy(),test_mean.to('cpu').detach().numpy().copy(), alternative='two-sided', method='auto')
    # plt.title(f'all_{ks_result_all.pvalue}')
    plt.title(f'train & test similarity distribution, {num_of_samples} samples, {ks_result_all.pvalue}')
    plt.hist(data[0], 30, alpha=0.6, density=False, label=olabels[0], stacked=False, range=(0.4, 1.0), color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=olabels[1], stacked=False, range=(0.4, 1.0), color=color[1])
    plt.legend()
    plt.ylabel('The number of samples')
    plt.xlabel('Mean Cosine Similarity')
    plt.savefig("results/sim_test_train_model_all.png")
    plt.close()

    try:
        train_data = [
            train_mean[:num_of_samples].to('cpu').detach().numpy().copy(),
            train_var[:num_of_samples].to('cpu').detach().numpy().copy(),
            train_median[:num_of_samples].to('cpu').detach().numpy().copy(),
            train_min[:num_of_samples].to('cpu').detach().numpy().copy(),
            train_max[:num_of_samples].to('cpu').detach().numpy().copy(),
            train_feature_bank[:num_of_samples].to('cpu').detach().numpy().copy()
        ]
        test_data = [
            test_mean[:num_of_samples].to('cpu').detach().numpy().copy(),
            test_var[:num_of_samples].to('cpu').detach().numpy().copy(),
            test_median[:num_of_samples].to('cpu').detach().numpy().copy(),
            test_min[:num_of_samples].to('cpu').detach().numpy().copy(),
            test_max[:num_of_samples].to('cpu').detach().numpy().copy(),
            test_feature_bank[:num_of_samples].to('cpu').detach().numpy().copy()
        ]
        with open(f'results/sim_train.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['mean', 'variance', 'median', 'min', 'max'])
            # for d1, d2, d3, d4, d5, d6 in zip(train_data[0], train_data[1], train_data[2], train_data[3], train_data[4], train_data[5]):
            for d1, d2, d3, d4, d5, d6 in zip(*train_data):
                writer.writerow([d1, d2, d3, d4, d5])
        with open(f'results/sim_test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['mean', 'variance', 'median', 'min', 'max'])
            # for d1, d2, d3, d4, d5, d6 in zip(test_data[0], test_data[1], test_data[2], test_data[3], test_data[4], test_data[5]):
            for d1, d2, d3, d4, d5, d6 in zip(*test_data):
                writer.writerow([d1, d2, d3, d4, d5])
    except:
        import traceback
        traceback.print_exc()

    return ks_result.pvalue, ks_result.statistic, ks_result_var.pvalue, ks_result_var.statistic, ks_result_median.pvalue, ks_result_median.statistic, ks_result_min.pvalue, ks_result_min.statistic, ks_result_max.pvalue, ks_result_max.statistic

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
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

    # args parse
    args = parser.parse_args()
    batch_size = args.batch_size

    # create output directory
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
    }
    wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)

    # data prepare
    query_number = 10
    if args.dataset == 'stl10':
        memory_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True, n=query_number)
        memory_data.set_mia_train_dataset_flag(True)
        test_data = utils.STL10NAug(root='data', split='unlabeled', transform=utils.stl_train_transform, download=True, n=query_number)
        test_data.set_mia_train_dataset_flag(False)
    elif args.dataset == 'cifar10':
        memory_data = utils.CIFAR10NAug(root='data', train=True, transform=utils.train_transform, download=True, n=query_number)
        test_data = utils.CIFAR10NAug(root='data', train=False, transform=utils.train_transform, download=True, n=query_number)
    else:
        memory_data = utils.CIFAR100NAug(root='data', train=True, transform=utils.train_transform, download=True, n=query_number)
        test_data = utils.CIFAR100NAug(root='data', train=False, transform=utils.train_transform, download=True, n=query_number)
    shuffle=True
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

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

    pvalue, statistic, var_pvalue, var_statistic, med_pvalue, med_statistic, min_pvalue, min_statistic, max_pvalue, max_statistic = sim(model, memory_loader, test_loader, num_of_samples=args.num_of_samples, encoder_flag=args.is_encoder, device=device)
    # sim(model_q, memory_loader, memory_loader)

    # save kstest result
    wandb.log({'pvalue': pvalue, 'statistic': statistic, 'var_pvalue': var_pvalue, 'var_statistic': var_statistic, 'med_pvalue': med_pvalue, 'med_statistic': med_statistic, 'min_pvalue': min_pvalue, 'min_statistic': min_statistic, 'max_pvalue': max_pvalue, 'max_statistic': max_statistic})

    # wandb finish
    os.remove(os.path.join(wandb.run.dir, args.model_path))
    # wandb.save("results/seed_check.png")
    wandb.save("results/seed_check_train.png")
    wandb.save("results/seed_check_test.png")
    wandb.save("results/sim_test_train_model.png")
    wandb.save("results/min_test_train_model.png")
    wandb.save("results/max_test_train_model.png")
    wandb.save("results/median_test_train_model.png")
    wandb.save("results/var_test_train_model.png")
    wandb.save("results/sim_test_train_model_all.png")
    wandb.save("results/sim_train.csv")
    wandb.save("results/sim_test.csv")
    wandb.finish()


