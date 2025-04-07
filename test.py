from DCGAN_pytorch import DCGAN

dcgan = DCGAN()
dcgan.load_from_checkpoint()
dcgan.generate(16, plot=True)
