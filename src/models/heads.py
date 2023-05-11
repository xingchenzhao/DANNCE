import torch
import torchvision.models as torchmodels
from .caffenet.models import caffenet


class CaffeNetDiscriminator(torch.nn.Module):
    def __init__(self, num_classes, size=1, depth=1, conv_input=False):

        super().__init__()

        size = int(1024 * size)

        blocks = []
        for d in range(1, depth + 1):
            blocks.append(
                torch.nn.Sequential(
                    torch.nn.Linear(size // d, size // (d + 1)),
                    torch.nn.ReLU(inplace=True), torch.nn.Dropout()))

        if conv_input:
            input_processing = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(256 * 6 * 6,
                                4096), torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(), torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(inplace=True), torch.nn.Dropout(),
                torch.nn.Linear(4096, size), torch.nn.ReLU(),
                torch.nn.Dropout())

        else:
            input_processing = torch.nn.Sequential(torch.nn.Linear(4096, size),
                                                   torch.nn.ReLU(),
                                                   torch.nn.Dropout())

        self.layers = torch.nn.Sequential(
            input_processing, *blocks,
            torch.nn.Linear(size // (depth + 1), num_classes))

        # disc head get's default initialization
        print(self.layers)

    def forward(self, x):
        return self.layers(x)


class ResNetDiscriminator(torch.nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        self.layers = torch.nn.Sequential(torch.nn.Linear(512, 1024),
                                          torch.nn.ReLU(), torch.nn.Dropout(),
                                          torch.nn.Linear(1024, 1024),
                                          torch.nn.ReLU(), torch.nn.Dropout(),
                                          torch.nn.Linear(1024, num_classes))

        # disc head get's default initialization

    def forward(self, x):
        return self.layers(x)
