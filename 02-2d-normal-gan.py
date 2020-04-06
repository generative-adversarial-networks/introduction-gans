import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

data_mean = 3.0
data_stddev = 0.2

Series_Length = 30

g_input_size = 20
g_hidden_size = 150
g_output_size = Series_Length

d_input_size = Series_Length
d_hidden_size = 75
d_output_size = 1


d_minibatch_size = 15
g_minibatch_size = 10

num_epochs = 5000
print_interval = 1000

d_learning_rate = 3e-3
g_learning_rate = 8e-3

def get_real_sampler(mu, sigma):
    dist = Normal( mu, sigma )
    return lambda m, n: dist.sample( (m, n) ).requires_grad_()

def get_noise_sampler():
    return lambda m, n: torch.rand(m, n).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian

actual_data = get_real_sampler( data_mean, data_stddev )
noise_data  = get_noise_sampler()

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()

    def forward(self, x):
        x = self.xfer( self.map1(x) )
        x = self.xfer( self.map2(x) )
        return self.xfer( self.map3( x ) )


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def draw(data):
    x = np.linspace(1,7, 100)
    to_draw = data.tolist()
    mu, std = norm.fit(to_draw)
    p1 = norm.pdf(x, mu, std)
    p2 = norm.pdf(x, 3.0, 0.2)
    plt.figure()
    plt.plot(x, p1, 'k', linewidth=2)
    plt.plot(x, p2, 'r', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    # d = data.tolist() if isinstance(data, torch.Tensor ) else data
    # print(d)
    # plt.plot( d )
    plt.show()

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.map1(x))
        x = self.elu(self.map2(x))
        return torch.sigmoid( self.map3(x) )

G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size)

criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate ) #, betas=optim_betas)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate ) #, betas=optim_betas)

def train_D_on_actual() :
    real_data = actual_data( d_minibatch_size, d_input_size )
    decision = D( real_data )
    error = criterion( decision, torch.ones( d_minibatch_size, 1 ))  # ones = true
    error.backward()


def train_D_on_generated() :
    noise = noise_data( d_minibatch_size, g_input_size )
    fake_data = G( noise )
    decision = D( fake_data )
    error = criterion( decision, torch.zeros( d_minibatch_size, 1 ))  # zeros = fake
    error.backward()


def train_G():
    noise = noise_data( g_minibatch_size, g_input_size )
    fake_data = G( noise )
    fake_decision = D( fake_data )
    error = criterion( fake_decision, torch.ones( g_minibatch_size, 1 ) )  # we want to fool, so pretend it's all genuine

    error.backward()
    return error.item(), fake_data


losses = []

for epoch in range(num_epochs):
    D.zero_grad()

    train_D_on_actual()
    train_D_on_generated()
    d_optimizer.step()

    G.zero_grad()
    loss, generated = train_G()
    g_optimizer.step()

    losses.append(loss)
    if (epoch % print_interval) == (print_interval - 1):
        print("Epoch %6d. Loss %5.3f" % (epoch + 1, loss))

    if epoch % 500 == 0:
        draw(generated)

print("Training complete")








# d = torch.empty( generated.size(0), 53 )
# print(generated)
# print(generated.size(0))
# print(d)
# for i in range( 0, d.size(0) ) :
#     print(generated[i])
#     data = generated.tolist()
#     print(data)
#     d[i] = torch.histc( generated[i], min=0, max=6, bins=53 )
# draw( d.t() )