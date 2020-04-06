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
batch_size=30


def plot_samples(real_data, fake_data):
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    fake_data_to_show = fake_data.permute(-1, 0).detach().numpy()
    real_data_to_show = real_data.permute(-1, 0).detach().numpy()
    plt.plot(real_data_to_show[0], real_data_to_show[1], 'o', color='blue')
    plt.plot(fake_data_to_show[0], fake_data_to_show[1], 'x', color='red')
    plt.show()


def get_data_points(mu_x, mu_y, sigma, chunk_size):
    x = torch.empty(chunk_size).normal_(mu_x, sigma)
    y = torch.empty(chunk_size).normal_(mu_y, sigma)
    chunk_of_data = torch.cat((x, y), 0) # Concat two tensors
    chunk_of_data = chunk_of_data.view(2, chunk_size) # Change dimesions
    chunk_of_data = chunk_of_data.permute(-1, 0) # Transpose via permute

    return chunk_of_data

def get_data_samples(batch_size):
    points_x = np.linspace(2.5, 7.5, batch_size)
    points = np.array([[x, x] for x in points_x])
    return torch.from_numpy(points).float()


class Generator(nn.Module):
    """
    Class that defines the the Generator Neural Network
    """
    def __init__(self, inp, out):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, 2),
            nn.SELU()
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
    generator = Generator(100, 2)
    generator_learning_rate = 0.001
    generator_loss = nn.BCELoss()
    generator_optimizer = optim.SGD(generator.parameters(), lr=generator_learning_rate, momentum=0.9)

    # Creating the GAN discriminator
    discriminator = Discriminator(2, 1)
    discriminator_learning_rate = 0.001
    discriminator_loss = nn.BCELoss()
    discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=discriminator_learning_rate, momentum=0.9)

    # Training epochs
    epochs = 30

    list_of_centers = [(2.5, 2.5), (7.5, 7.5)]

    noise_for_plot = read_latent_space(batch_size)
    discriminator_loss_storage, generator_loss_storage = [], []

    print('MNIST dataset loaded...')
    print('Starting adversarial GAN training for {} epochs.'.format(epochs))

    # Plot a little bit of trash
    input_real = get_data_samples(list_of_centers, 0.05, batch_size)
    generator_output = generator(noise_for_plot)
    plot_samples(input_real, generator_output)

    for epoch in range(epochs):

        batch_number = 0

        # training discriminator
        while batch_number < 100: #len(data_iterator):

            # 1. Train the discriminator
            discriminator.zero_grad()
            # 1.1 Train discriminator on real data
            input_real = get_data_samples(list_of_centers, 0.05, batch_size)
            discriminator_real_out = discriminator(input_real.reshape(batch_size, 2))
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

        if epoch % 2 == 0:
            generator_output = generator(noise_for_plot)
            plot_samples(input_real, generator_output)

    plot_loss_evolution(discriminator_loss_storage, generator_loss_storage)

main()
