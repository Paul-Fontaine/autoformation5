from DCGAN_pytorch import DCGAN

dcgan = DCGAN()
dcgan.load_from_checkpoint()
dcgan.generate(49, plot=True, save="generated_images.png")
