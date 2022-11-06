import torch
from torch import nn
from model import Discriminator,Generator
from model import train_loader
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config",config_name="config")
class TrainGanModel:
    def __init__(self,config:DictConfig):
        self.lr = config.learning_rate
        self.num_epochs = config.num_epochs
        self.loss_function = nn.BCELoss()
        self.discriminator=Discriminator()
        self.generator=Generator()
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

    def train(self,config:DictConfig):
        for epoch in range(self.num_epochs):
            for n, (real_samples, _) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples_labels = torch.ones((config.batch_size, 1))
                latent_space_samples = torch.randn((config.batch_size, 2))
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((config.batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                loss_discriminator = self.loss_function(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((config.batch_size, 2))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = self.loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                self.optimizer_generator.step()

                # Show loss
                if epoch % 10 == 0 and n == config.batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

if __name__=="__main__":
    TrainGanModel().train()