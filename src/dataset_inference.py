import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from scipy.stats import ttest_ind
import numpy as np


def verify_model(model, trainloader, valloader, testloader, n_components: int=50, covariance: str='diag'):
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
        for data in trainloader:
            images, _ = data
            representations = model(images)
            train_representations.append(representations)

        for data in valloader:
            images, _ = data
            representations = model(images)
            val_representations.append(representations)

        for data in testloader:
            images, _ = data
            representations = model(images)
            test_representations.append(representations)

    train_representations = torch.cat(train_representations).numpy()
    val_representations = torch.cat(val_representations).numpy()
    test_representations = torch.cat(test_representations).numpy()

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
    # load CIFAR-10
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # random split train and test
    train_size = int(0.5 * len(trainset))
    val_size = len(trainset) - train_size
    generator = torch.Generator().manual_seed(42)
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size], generator)

    # create dataloader
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # load model (resnet18)
    model = torchvision.models.resnet18(pretrained=True)

    # Verify model by GMM
    n_components = 50
    covariance_type = 'diag'
    """
    Classifier:
        n_components: 10
        covariance: full
    Encoder:
        n_components: 50
        covariance: diag
    """
    p_value, statistic, effect_size = verify_model(model, trainloader, valloader, testloader, n_components, covariance_type)

    # 結果の出力
    print(f'p-value: {p_value}')
    print(f'statistic: {statistic}')
    print(f'effect-size: {effect_size}')

