import torch.nn as nn
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 64, 7), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(64, 128, 5), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(128, 128, 3), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(128, 256, 3), nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(256 * 27, 384), 
                                nn.Sigmoid(),
                                nn.Linear(384, 2) 
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)