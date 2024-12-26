import argparse
import os
import random
import csv
from scipy.stats import kstest, mannwhitneyu, anderson_ksamp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

import utils
from model import Model, Classifier, StudentModel
from quantize import load_quantize_model

def seed_check(imgs, is_train: bool = True):
    fig, axes = plt.subplots(3, 3, tight_layout=True)
    # fig, axes = plt.subplots(2, 5, tight_layout=True)
    for i in range(3):
        for j in range(3):
            axes[i, j].axis("off")
            # axes[i, j].imshow(utils.inv_transform(img[i * 3 + j]).permute(1, 2, 0))
            axes[i, j].imshow(utils.inv_transform(imgs[i][j]).permute(1, 2, 0))
    # axes[0, 0].axis("off")
    # axes[0, 1].axis("off")
    # axes[0, 2].axis("off")
    # axes[1, 0].axis("off")
    # axes[1, 1].axis("off")
    # axes[1, 2].axis("off")
    # axes[2, 0].axis("off")
    # axes[2, 1].axis("off")
    # axes[2, 2].axis("off")
    # axes[0, 0].imshow(utils.inv_transform(img[0]).permute(1, 2, 0))
    # axes[0, 1].imshow(utils.inv_transform(img[1]).permute(1, 2, 0))
    # axes[0, 2].imshow(utils.inv_transform(img[2]).permute(1, 2, 0))
    # axes[1, 0].imshow(utils.inv_transform(img[3]).permute(1, 2, 0))
    # axes[1, 1].imshow(utils.inv_transform(img[4]).permute(1, 2, 0))
    # axes[1, 2].imshow(utils.inv_transform(img[5]).permute(1, 2, 0))
    # axes[2, 0].imshow(utils.inv_transform(img[6]).permute(1, 2, 0))
    # axes[2, 1].imshow(utils.inv_transform(img[7]).permute(1, 2, 0))
    # axes[2, 2].imshow(utils.inv_transform(img[8]).permute(1, 2, 0))
    if is_train:
        fig.savefig('results/seed_check_train.png')
    else:
        fig.savefig('results/seed_check_test.png')
    plt.close()

def extract_features(model, data_loader, encoder_flag:bool=True, is_train:bool=True, device:str='cuda'):
    model.eval()
    feature_bank = []
    cos_list_bank = []
    similarity = torch.nn.CosineSimilarity(dim=1)
    is_first_iter = True

    with torch.no_grad():
        for x, target in tqdm(data_loader, desc='Feature extracting'):
            target = target.to(device)
            if is_first_iter:
                imgs = [img[0] for img in x], [img[1] for img in x], [img[2] for img in x]
                seed_check(imgs, is_train)
                is_first_iter = False

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

            feature_bank.append(torch.stack(feature_list, dim=1))
            cos_list_bank.append(torch.stack(cos_list, dim=1))

    return feature_bank, cos_list_bank

def plot_histogram(data, labels, xlabel, ylabel, title, filename, color, range):
    plt.hist(data[0], 30, alpha=0.6, density=False, label=labels[0], stacked=False, range=range, color=color[0])
    plt.hist(data[1], 30, alpha=0.6, density=False, label=labels[1], stacked=False, range=range, color=color[1])
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def get_similarity_data(n, train_data, test_data, num_of_samples):
    return [
        torch.mean(train_data[:, :n*(n-1)//2], dim=1)[:num_of_samples].to('cpu').detach().numpy().copy(),
        torch.mean(test_data[:, :n*(n-1)//2], dim=1)[:num_of_samples].to('cpu').detach().numpy().copy()
    ]

def sim(model, memory_data_loader, test_data_loader, num_of_samples:int=500, encoder_flag:bool=True, device:str='cuda', method: str='kstest'):
    model.eval()
    if method == 'mannwhitneyu':
        test_method = lambda x, y: mannwhitneyu(x, y, alternative='two-sided')
    elif method == 'anderson':
        test_method = lambda x, y: anderson_ksamp([x,y], midrank=True, method=None)
    else:
        test_method = lambda x, y: kstest(x, y, alternative='two-sided', method='auto')
    # similarity = torch.nn.CosineSimilarity(dim=1)
    test_feature_bank, train_feature_bank = [], []
    test_cos_list, train_cos_list = [], []
    test_mean, train_mean = [], []
    test_var, train_var = [], []
    test_median, train_median = [], []
    test_max, train_max = [], []
    test_min, train_min = [], []

    with torch.no_grad():
        # generate feature bank
        train_feature_bank, train_cos_list = extract_features(model, memory_data_loader, encoder_flag, True, device)
        test_feature_bank, test_cos_list = extract_features(model, test_data_loader, encoder_flag, False, device)

        # [D, N]
        train_cos_list = torch.cat(train_cos_list, dim=0).contiguous()
        test_cos_list = torch.cat(test_cos_list, dim=0).contiguous()
        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        test_feature_bank = torch.cat(test_feature_bank, dim=0).contiguous()

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
    ks_result = test_method(data[0], data[1])
    plot_histogram(data, olabels, 'Mean Cosine Similarity', 'The number of samples', f'train & test similarity distribution, {num_of_samples} samples', "results/sim_test_train_model.png", color, (0.4, 1.0))

    # Different n values
    for n in [2, 3, 5]:
        data = get_similarity_data(n, train_cos_list, test_cos_list, num_of_samples)
        plot_histogram(data, olabels, 'Mean Cosine Similarity', 'The number of samples', f'train & test similarity distribution, {num_of_samples} samples', f"results/sim_test_train_model_(n={n}).png", color, (0.4, 1.0))

    # Median Cosine Similarity
    data = [train_median[:num_of_samples].to('cpu').detach().numpy().copy(),test_median[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_median = test_method(data[0], data[1])
    data = [train_median[:num_of_samples].to('cpu').detach().numpy().copy(), test_median[:num_of_samples].to('cpu').detach().numpy().copy()]
    plot_histogram(data, olabels, 'Median Cosine Similarity', 'The number of samples', f'train & test similarity distribution, {num_of_samples} samples', "results/median_test_train_model.png", color, (0.4, 1.0))

    # Variance
    data = [train_var[:num_of_samples].to('cpu').detach().numpy().copy(),test_var[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_var = test_method(data[0], data[1])
    plot_histogram(data, olabels, 'Variance', 'The number of samples', f'train & test Variance distribution, {num_of_samples} samples', "results/var_test_train_model.png", color, (0.0, 0.1))

    # Minimum Cosine Similarity
    data = [train_min[:num_of_samples].to('cpu').detach().numpy().copy(), test_min[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_min = test_method(data[0], data[1])
    plot_histogram(data, olabels, 'Minimum', 'The number of samples', f'train & test Minimum distribution, {num_of_samples} samples', "results/min_test_train_model.png", color, (0.4, 1.0))

    # Maximum Cosine Similarity
    data = [train_max[:num_of_samples].to('cpu').detach().numpy().copy(), test_max[:num_of_samples].to('cpu').detach().numpy().copy()]
    ks_result_max = test_method(data[0], data[1])
    plot_histogram(data, olabels, 'Maximum', 'The number of samples', f'train & test Maximum distribution, {num_of_samples} samples', "results/max_test_train_model.png", color, (0.4, 1.0))

    # All data
    data = [train_mean.to('cpu').detach().numpy().copy(), test_mean.to('cpu').detach().numpy().copy()]
    ks_result_all = test_method(data[0], data[1])
    plot_histogram(data, olabels, 'Mean Cosine Similarity', 'The number of samples', f'train & test similarity distribution, {num_of_samples} samples, {ks_result_all.pvalue}', "results/sim_test_train_model_all.png", color, (0.4, 1.0))

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
        with open(f'results/sim_train_raw.csv', 'w') as f:
            # datas = train_cos_list[:num_of_samples].to('cpu').detach().numpy().copy()
            datas = train_cos_list.to('cpu').detach().numpy().copy()
            writer = csv.writer(f)
            writer.writerow([d for d in range(len(data[0]))])
            for data in datas:
                writer.writerow([d for d in data])
        with open(f'results/sim_test_raw.csv', 'w') as f:
            # datas = test_cos_list[:num_of_samples].to('cpu').detach().numpy().copy()
            datas = test_cos_list.to('cpu').detach().numpy().copy()
            writer = csv.writer(f)
            writer.writerow([d for d in range(len(data[0]))])
            for data in datas:
                writer.writerow([d for d in data])
    except:
        import traceback
        traceback.print_exc()

    return ks_result.pvalue, ks_result.statistic, ks_result_var.pvalue, ks_result_var.statistic, ks_result_median.pvalue, ks_result_median.statistic, ks_result_min.pvalue, ks_result_min.statistic, ks_result_max.pvalue, ks_result_max.statistic, ks_result_2.pvalue, ks_result_2.statistic, ks_result_3.pvalue, ks_result_3.statistic, ks_result_5.pvalue, ks_result_5.statistic

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

    pvalue, statistic, var_pvalue, var_statistic, med_pvalue, med_statistic, min_pvalue, min_statistic, max_pvalue, max_statistic, pvalue_2, statistic_2, pvalue_3, statistic_3, pvalue_5, statistic_5 = sim(model, memory_loader, test_loader, num_of_samples=args.num_of_samples, encoder_flag=args.is_encoder, device=device)
    # sim(model_q, memory_loader, memory_loader)

    # save kstest result
    wandb.log({
        'pvalue': pvalue, 'statistic': statistic,
        'var_pvalue': var_pvalue, 'var_statistic': var_statistic,
        'med_pvalue': med_pvalue, 'med_statistic': med_statistic,
        'min_pvalue': min_pvalue, 'min_statistic': min_statistic,
        'max_pvalue': max_pvalue, 'max_statistic': max_statistic,
        'mean_pvalue_2': pvalue_2, 'mean_statistic_2': statistic_2,
        'mean_pvalue_3': pvalue_3, 'mean_statistic_3': statistic_3,
        'mean_pvalue_5': pvalue_5, 'mean_statistic_5': statistic_5
    })

    # wandb finish
    os.remove(os.path.join(wandb.run.dir, args.model_path))
    # wandb.save("results/seed_check.png")
    wandb.save("results/seed_check_train.png")
    wandb.save("results/seed_check_test.png")
    wandb.save("results/sim_test_train_model.png")
    wandb.save("results/sim_test_train_model_(n=2).png")
    wandb.save("results/sim_test_train_model_(n=3).png")
    wandb.save("results/sim_test_train_model_(n=5).png")
    wandb.save("results/min_test_train_model.png")
    wandb.save("results/max_test_train_model.png")
    wandb.save("results/median_test_train_model.png")
    wandb.save("results/var_test_train_model.png")
    wandb.save("results/sim_test_train_model_all.png")
    wandb.save("results/sim_train.csv")
    wandb.save("results/sim_test.csv")
    wandb.save("results/sim_train_raw.csv")
    wandb.save("results/sim_test_raw.csv")
    wandb.finish()


