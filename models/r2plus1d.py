# code from torchvision and https://github.com/facebookresearch/VMZ
# @Author: Enea Duka
# @Date: 4/22/21
import torch
import torch.nn as nn
from torchvision.models.video.resnet import R2Plus1dStem, BasicBlock, Bottleneck
from utils.arg_parse import opt
from utils.logging_setup import logger


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=359,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()
        self.block = block
        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if not opt.act_backbone == 'r2plus1d' and not opt.unint_act_backbone == 'r2plus1d':
            return x

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        if opt.unint_act_backbone == 'r2plus1d':
            return x

        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def reinit_fc(self, num_classes):
        self.fc = nn.Sequential(
            nn.Linear(512 * self.block.expansion, 256 * self.block.expansion),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256 * self.block.expansion, num_classes)
        )

    def freeze(self):
        # freeze all layers except the final block
        for name, child in self.named_children():
            if name not in ['layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = False
            for m in child.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()


class R2Plus1dStem_Pool(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """

    def __init__(self):
        super(R2Plus1dStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
                in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


def _generic_resnet(arch, pretrained=False, **kwargs):
    model = VideoResNet(**kwargs)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.load(arch)
        model.load_state_dict(state_dict)

    return model


def build_r2plus1d(pretrain=True,
                   use_pool1=True,
                   pretrain_type='ig65m',
                   replace_fc=False,
                   num_classes=0,
                   **kwargs):
    if pretrain_type == 'sp1m':
        pretrained_model_path = ''
    elif pretrain_type == 'ig65m':
        pretrained_model_path = '/BS/unintentional_actions/nobackup/pretrained_models/r2plus1d/r2plus1d_152_ig65m_from_scratch_f106380637.pth'

    model = _generic_resnet(
        pretrained_model_path,
        pretrain,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrain:
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
    if replace_fc:
        model.freeze()
        model.reinit_fc(num_classes)
    model = model.cuda()
    model = nn.DataParallel(model)

    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    loss = nn.CrossEntropyLoss()

    logger.debug(str(model))
    logger.debug(str(optimizer))
    logger.debug(str(loss))

    return model, optimizer, loss
