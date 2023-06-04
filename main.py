import torch
from torch import nn
from torchvision import datasets, transforms
from torch.autograd.variable import Variable
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory to save images
os.makedirs('generated_images', exist_ok=True)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(
    root='./mnist_data/',
    train=True, 
    transform=transform,
    download=True)

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)



class VirtualBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        # Reference batch
        self.ref_batch = torch.zeros(num_features)
        self.ref_batch_var = torch.ones(num_features)

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, input):
        batch_size, num_channels, height, width = input.size()
        input_view = input.view(batch_size, -1)

        batch_mean = torch.mean(input_view, dim=0)
        batch_var = torch.var(input_view, dim=0)

        # Batch mean and variance are calculated by averaging over both the input batch and a reference batch
        mean = (self.ref_batch.detach() + batch_mean) / 2
        var = (self.ref_batch_var.detach() + batch_var) / 2

        # Update stored batch statistics
        self.ref_batch = self.ref_batch.clone().detach().requires_grad_(True)
        self.ref_batch_var = self.ref_batch_var.clone().detach().requires_grad_(True)

        input_view = (input_view - mean) / torch.sqrt(var + self.eps)
        input_view = input_view * self.gamma + self.beta
        output = input_view.view(batch_size, num_channels, height, width)

        return output


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False, device=torch.device('cpu')):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims).to(device))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # Matrix multiplication
        # [batch, in_features] x [in_features, out_features*kernel_dims] -> [batch, out_features, kernel_dims]
        matrices = x.mm(self.T.view(self.in_features, -1)).view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # [1, batch, out_features, kernel_dims]
        M_T = M.permute(1, 0, 2, 3)  # [batch, 1, out_features, kernel_dims]
        norm = torch.abs(M - M_T).sum(3)  # [batch, batch, out_features]
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1) if self.mean else expnorm.sum(0)  # [batch, out_features]

        x = torch.cat([x, o_b], 1)
        return x


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        self.linear = nn.Linear(128*7*7, 128)  # Adjusted this line: input size is 128*7*7
        self.minibatch_layer = MinibatchDiscrimination(128, 128, 8, device=device)
        self.final = nn.Sequential(
            nn.Linear(256, 1),  # Adjusted to match the concatenated size of x
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass input through main part
        x = self.main(x)
        x = self.linear(x)
        x = self.minibatch_layer(x)
        x = self.final(x)

        return x

        
    
# Define Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256*7*7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

# Initialize models and optimizers
discriminator = Discriminator().to(device)
generator = Generator().to(device)
historical_generator = Generator().to(device)

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define losses
loss = nn.BCELoss().to(device)

# Training
def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()

    real_output = discriminator(real_data)
    real_labels = Variable(torch.ones(real_data.size(0), 1)).to(device) * 0.9  # Smoothing the real labels
    error_real = loss(real_output, real_labels)
    error_real.backward()

    fake_output = discriminator(fake_data)
    fake_labels = Variable(torch.zeros(real_data.size(0), 1)).to(device)
    error_fake = loss(fake_output, fake_labels)
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()

    fake_output = discriminator(fake_data)

    error = loss(fake_output, Variable(torch.ones(fake_data.size(0), 1)).to(device))
    
    error.backward()

    optimizer.step()

    return error


d_steps = 1
# Number of epochs
num_epochs = 200

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):

        real_data = Variable(real_batch).to(device)

        # Generate fake data
        fake_data = generator(Variable(torch.randn(real_data.size(0), 100)).view(real_data.size(0), 100).to(device))

        fake_data = fake_data.detach()

        # Train discriminator
        d_error = train_discriminator(d_optimizer, real_data, fake_data)

        # Generate new fake data for generator
        fake_data = generator(Variable(torch.randn(real_data.size(0), 100)).view(real_data.size(0), 100).to(device))


        # Train generator
        g_error = train_generator(g_optimizer, fake_data)

        # Update historical average generator
        alpha = 0.999  # Historical averaging coefficient
        for historical_param, generator_param in zip(historical_generator.parameters(), generator.parameters()):
            historical_param.data.mul_(alpha).add_((1 - alpha) * generator_param.data)

        if (n_batch) % 100 == 0:
            print('Epoch: {}, Batch: {}, D Loss: {}, G Loss: {}'.format(epoch, n_batch, d_error, g_error))

            
    imgs_numpy = fake_data.data.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.reshape(imgs_numpy[i], (28, 28)), cmap='gray')
        plt.axis('off')
        
    fig.savefig('generated_images/generated_mnist_%d.png' % (epoch))
    
    display.display(plt.gcf())