import torch
import pickle
import torchvision
import torchvision.transforms as transforms


def mnist():
    batch_size = 60000
    transform = transforms.Compose(
        [transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data", train=False, transform=transform
    )
    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    for i, (images, labels) in enumerate(train_loader):
        images = images.flatten(start_dim=1)
        labels = labels

    x = images
    y = labels

    for i, (images_test, labels_test) in enumerate(test_loader):
        images_test = images_test.flatten(start_dim=1)
        labels_test = labels_test

    x_test = images_test
    y_test = labels_test

    with open('data/mnist_', 'wb+') as f:
        pickle.dump([x, y, x_test, y_test], f)

    return x, y, x_test, y_test


def get_mnist(type='reg'):
    if type == 'reg':
        data_file = 'data/mnist_'
        with open(data_file, 'rb+') as f:
            x, y, x_test, y_test = pickle.load(f)
    return x, y, x_test, y_test


if __name__ == '__main__':
    """
    Generates all the required data
    """
    mnist()
