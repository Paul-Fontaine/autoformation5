from torchvision.datasets import MNIST
from torchvision import transforms
from DCGAN_pytorch import DCGAN
from torch.utils.data import DataLoader

# Hyperparameters
N_EPOCHS = 20
BATCH_SIZE = 256
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999


# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist = MNIST(root='data', download=True, train=True, transform=transform)
# mnist.data = mnist.data[mnist.targets == 0]
# mnist.targets = mnist.targets[mnist.targets == 0]
train_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

# Create a DCGAN model
dcgan = DCGAN()
dcgan.train_(train_loader, N_EPOCHS, LR, BETA1, BETA2, use_tensorboard=False)
# dcgan.load_from_checkpoint()
# dcgan.generate(16, plot=True)
