from torchvision.datasets import MNIST
from torchvision import transforms

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
mnist = MNIST(root='data', download=True, train=True, transform=transform)
