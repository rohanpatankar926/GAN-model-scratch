from data_generate.DataGen import DataGen
from torch.utils.data import DataLoader
import torch
from torch import nn
import hydra
from omegaconf import DictConfig

data_gen = DataGen()
train_set = data_gen.generate()

@hydra.main(config_path="config",config_name="config")
def train_loader(config:DictConfig):
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    return train_loader

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 556),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(556, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

if __name__=="__main__":
    discriminator = Discriminator()
    generator=Generator()