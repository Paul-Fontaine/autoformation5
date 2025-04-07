from torchvision.datasets import MNIST
from torchvision import transforms

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist = MNIST(root='data', download=True, train=True, transform=transform)

# Select only the digit '0' for training
# mnist.data = mnist.data[mnist.targets == 0]
# mnist.targets = mnist.targets[mnist.targets == 0]

# select 5000 samples
# mnist.data = mnist.data[:5000]
# mnist.targets = mnist.targets[:5000]
