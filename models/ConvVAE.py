# @Author: Enea Duka
# @Date: 4/29/21
import torch
import torch.nn as nn
from utils.arg_parse import opt

class ConvVAE(nn.Module):
    def __init__(self, in_channels):
        super(ConvVAE, self).__init__()

        #self.encoder = nn.Sequential(
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=(11, 11), stride=4)
        self.sig = nn.Sigmoid()
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=(5, 5), stride=1, padding=2)
        # nn.Sigmoid(),
        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=1, padding=1)
        # nn.Sigmoid(),
        self.lin1 = nn.Linear(13 * 13 * 128, 2000)
        #)

        # self.decoder = nn.Sequential(
        self.lin2 = nn.Linear(2000, 13 * 13 * 128)
        # nn.Sigmoid(),
        self.mup1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.convt1 = nn.ConvTranspose2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        # nn.Sigmoid(),
        self.mup2 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.convt2 = nn.ConvTranspose2d(256, 512, kernel_size=(5, 5), stride=1, padding=2)
        # nn.Sigmoid(),
        self.convt3 = nn.ConvTranspose2d(512, in_channels, kernel_size=(11, 11), stride=4)
        # )

    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x)

        x = self.conv1(x)
        x, idx1 = self.mp1(x)
        x = self.conv2(x)
        x, idx2 = self.mp2(x)
        x = self.conv3(x)
        x = x.reshape(-1, 128*13*13)
        x = self.lin1(x)

        x = self.lin2(x)
        x = x.reshape(-1, 128, 13, 13)
        x = self.convt1(x)
        x = self.mup1(x, idx2)
        x = self.convt2(x)
        x = self.mup2(x, idx1)
        x = self.convt3(x)

        return x



def create_model(in_channels, lr, optim='adam', weight_decay=0, momentum=0, pretrained=False):
    model = ConvVAE(in_channels)
    model.cuda()
    model = nn.DataParallel(model)

    if pretrained:
        model_dict = torch.load(opt.ptr_tmpreg_model_path)['state_dict']
        model.load_state_dict(model_dict)

    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss = nn.MSELoss(reduction='sum')

    return model, optimizer, loss