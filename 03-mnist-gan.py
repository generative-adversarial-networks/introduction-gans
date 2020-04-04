import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms
import sys


# Data load and transformation
batch_size=100

data_transforms = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True,
                           download=True, transform=data_transforms)
dataloader_mnist_train = t_data.DataLoader(mnist_trainset,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )


class Generator(nn.Module):
    """
    Class that defines the the Generator Neural Network
    """
    def __init__(self, inp, out):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(inp, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, out),
                nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    """
    Class that defines the the Discriminator Neural Network
    """
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


def plot_loss_evolution(discriminator_loss, generator_loss):
    x = range(len(discriminator_loss)) if len(discriminator_loss) > 0 else range(len(generator_loss))
    if len(discriminator_loss) > 0: plt.plot(x, discriminator_loss, '-b', label='Discriminator loss')
    if len(generator_loss) > 0: plt.plot(x, generator_loss, ':r', label='Generator loss')
    plt.legend()
    plt.show()

def plot_mnist_data(data, label=None):
    """
    Shows a data sample of MNIST database
    :param data: Data sample of MNIST.
    :param label: Label of the data sample of MNIST.
    """
    data = data.detach().reshape(28, 28)
    plt.imshow(data, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if not label is None:
        plt.xlabel(label, fontsize='x-large')
    plt.show()


def read_latent_space(size):
    """
    Creates a tensor with random values fro latent space  with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with random values (z) with shape = size
    """
    z = torch.rand(size,100)
    if torch.cuda.is_available(): return z.cuda()
    return z


def real_data_target(size):
    """
    Creates a tensor with the target for real data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with real label value (ones) with shape = size
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    """
    Creates a tensor with the target for fake data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with fake label value (zeros) with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def main():

    # Creating the GAN generator
    generator = Generator(100, 784)
    generator_learning_rate = 0.001
    generator_loss = nn.BCELoss()
    generator_optimizer = optim.SGD(generator.parameters(), lr=generator_learning_rate, momentum=0.9)

    # Creating the GAN discriminator
    discriminator = Discriminator(784, 1)
    discriminator_learning_rate = 0.001
    discriminator_loss = nn.BCELoss()
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=discriminator_learning_rate, momentum=0.9)

    # Training epochs
    epochs = 1

    noise_for_plot = read_latent_space(batch_size)
    discriminator_loss_storage, generator_loss_storage = [], []

    print('MNIST dataset loaded...')
    print('Starting adversarial GAN training for {} epochs.'.format(epochs))
    for epoch in range(epochs):

        data_iterator = iter(dataloader_mnist_train)
        batch_number = 0

        # training discriminator
        while batch_number < len(data_iterator):

            # 1. Train the discriminator
            discriminator.zero_grad()
            # 1.1 Train discriminator on real data
            input_real, _ = next(data_iterator)
            discriminator_real_out = discriminator(input_real.reshape(batch_size, 784))
            discriminator_real_loss = discriminator_loss(discriminator_real_out, real_data_target(batch_size))
            discriminator_real_loss.backward()
            # 1.2 Train the discriminator on data produced by the generator
            input_fake = read_latent_space(batch_size)
            generator_fake_out = generator(input_fake).detach()
            discriminator_fake_out = discriminator(generator_fake_out)
            discriminator_fake_loss = discriminator_loss(discriminator_fake_out, fake_data_target(batch_size))
            discriminator_fake_loss.backward()
            # 1.3 Optimizing the discriminator weights
            discriminator_optimizer.step()

            discriminator_loss_storage.append(discriminator_fake_loss + discriminator_real_loss)

            # 2. Train the generator
            generator.zero_grad()
            # 2.1 Create fake data
            input_fake = read_latent_space(batch_size)
            generator_fake_out = generator(input_fake)
            # 2.2 Try to fool the discriminator with fake data
            discriminator_out_to_train_generator = discriminator(generator_fake_out)
            discriminator_loss_to_train_generator = generator_loss(discriminator_out_to_train_generator,
                                                                   real_data_target(batch_size))
            discriminator_loss_to_train_generator.backward()
            # 2.3 Optimizing the generator weights
            generator_optimizer.step()
            generator_loss_storage.append(discriminator_loss_to_train_generator)

            batch_number += 1

        print('Epoch={}, Discriminator loss={}, Generator loss={}'.format(epoch, discriminator_loss_storage[-1], generator_loss_storage[-1]))

        generator_output = generator(noise_for_plot)
        plot_mnist_data(generator_output[0])
        #if epoch % printing_steps == 0:
    plot_loss_evolution(discriminator_loss_storage, generator_loss_storage)

main()
