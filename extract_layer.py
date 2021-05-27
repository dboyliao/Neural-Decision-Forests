from torch import nn


class MNISTFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, shallow=False):
        super().__init__()
        self.shallow = shallow
        if shallow:
            self.add_module(
                "conv1", nn.Conv2d(1, 64, kernel_size=15, padding=1, stride=5)
            )
        else:
            self.add_module("conv1", nn.Conv2d(1, 32, kernel_size=3, padding=1))
            self.add_module("relu1", nn.ReLU())
            self.add_module("pool1", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop1", nn.Dropout(dropout_rate))
            self.add_module("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.add_module("relu2", nn.ReLU())
            self.add_module("pool2", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop2", nn.Dropout(dropout_rate))
            self.add_module("conv3", nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.add_module("relu3", nn.ReLU())
            self.add_module("pool3", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop3", nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        if self.shallow:
            return 64 * 4 * 4
        else:
            return 128 * 3 * 3


class UCIAdultFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0.0, shallow=True):
        super().__init__()
        self.shallow = shallow
        if shallow:
            self.add_module("linear", nn.Linear(113, 1024))
        else:
            raise NotImplementedError

    def get_out_feature_size(self):
        return 1024


class UCILetterFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0.0, shallow=True):
        super().__init__()
        self.shallow = shallow
        if shallow:
            self.add_module("linear", nn.Linear(16, 1024))
        else:
            raise NotImplementedError

    def get_out_feature_size(self):
        return 1024


class UCIYeastFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate=0.0, shallow=True):
        super().__init__()
        self.shallow = shallow
        if shallow:
            self.add_module("linear", nn.Linear(8, 1024))
        else:
            raise NotImplementedError

    def get_out_feature_size(self):
        return 1024
