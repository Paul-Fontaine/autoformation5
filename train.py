from DCGAN_pytorch import DCGAN
from torch.utils.data import DataLoader
from dataset import mnist
import time

# Hyperparameters
N_EPOCHS = 20
BATCH_SIZE = 64
LR = 0.0002
BETA1 = 0.9
BETA2 = 0.999
DISCRIMINATOR_UPDATE_PERIOD = 5


def run_name_from_hyperparams():
    # get the current time in hours, minutes and seconds
    current_time = time.localtime()
    hours = current_time.tm_hour
    minutes = current_time.tm_min
    seconds = current_time.tm_sec
    return f"runs/{hours}-{minutes}-{seconds}__epochs_{N_EPOCHS}_batch_{BATCH_SIZE}_lr_{LR}_beta1_{BETA1}_beta2_{BETA2}_d_update_{DISCRIMINATOR_UPDATE_PERIOD}"


train_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

# Create a DCGAN model
dcgan = DCGAN()
dcgan.train_(train_loader, N_EPOCHS, LR, BETA1, BETA2, DISCRIMINATOR_UPDATE_PERIOD, use_tensorboard=run_name_from_hyperparams())
