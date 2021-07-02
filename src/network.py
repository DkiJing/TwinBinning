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
        # # network 1
        # self.convnet = nn.Sequential(nn.Conv1d(1, 32, 3), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2),
        #                              nn.Conv1d(32, 64, 3), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2))
        #  self.fc = nn.Sequential(nn.Linear(64 * 14, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 2)
        #                         )
        # # network 2
        # self.convnet = nn.Sequential(nn.Conv1d(1, 16, kernel_size=5, padding=2),
        #                              nn.MaxPool1d(2), nn.PReLU(),

        #                              nn.Conv1d(16, 32, kernel_size=5, padding=2),
        #                              nn.MaxPool1d(2), nn.PReLU(),

        #                              nn.Conv1d(32, 48, kernel_size=5, padding=2),
        #                              nn.MaxPool1d(2), nn.PReLU())

        # self.fc = nn.Sequential(nn.Linear(48 * 20, 512), # 原来是48*44, 512
        #                         nn.PReLU(),
        #                         nn.Linear(512, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 2)
        #                         )

        # # network 3
        # self.convnet = nn.Sequential( nn.Conv1d(1, 32, 5), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2),
        #                              nn.Conv1d(32, 64, 5), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2),
        #                              nn.Conv1d(64, 128, 5), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2))
        # self.fc = nn.Sequential(nn.Linear(128 * 17, 256), # 原来是128*124, 256
        #                         nn.PReLU(),
        #                         nn.Linear(256, 128),
        #                         nn.PReLU(),
        #                         nn.Linear(128, 2)
        #                         )
        # 
        # # network 4
        # self.convnet = nn.Sequential(nn.Conv1d(1, 16, 3), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2),
        #                              nn.Conv1d(16, 32, 3), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2),
        #                              nn.Conv1d(32, 64, 3), nn.PReLU(),
        #                              nn.MaxPool1d(2, stride=2))
        # self.fc = nn.Sequential(nn.Linear(64 * 18, 1024), # 原来是64*626, 1024
        #                         nn.PReLU(),
        #                         nn.Linear(1024, 512),
        #                         nn.PReLU(),
        #                         nn.Linear(512, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 128),
        #                         nn.PReLU(),
        #                         nn.Linear(128, 2)
        #                         )

        # network 5
        self.convnet = nn.Sequential(nn.Conv1d(1, 64, 7), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(64, 128, 5), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(128, 128, 3), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(128, 256, 3), nn.PReLU())
        self.fc = nn.Sequential(nn.Linear(256 * 3, 384), # 原来是256*3, 384
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