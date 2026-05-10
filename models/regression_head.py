from torch import nn

class Decoder_model(nn.Module):
    def __init__(self, in_channels, num_layers=1, kernel_size=3):
        super(Decoder_model, self).__init__()

        self.layer = []
        for _ in range(num_layers):
            self.layer.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2))
            self.layer.append(nn.LeakyReLU())
        self.layer = nn.Sequential(*self.layer)

        self.out_channels = in_channels
        self.reset_parameters()

    def forward(self, x):
        return self.layer(x)
    
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class ObjectnessHead(nn.Module):
    def __init__(self, in_channels):
        super(ObjectnessHead, self).__init__()

        self.head = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1)
            )    
        self.reset_parameters()

    def forward(self, x):
        return self.head(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class BboxesHead(nn.Module):
    def __init__(self, in_channels):
        super(BboxesHead, self).__init__()

        self.head = nn.Sequential(
                nn.Conv2d(in_channels, 4, kernel_size=1),
            )
        self.reset_parameters()
    
    def forward(self, x):
        return self.head(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)