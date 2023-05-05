from torch import nn
import torch
from torch.nn import Module, Conv2d, Parameter

def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.BatchNorm2d(out_planes),
    )


class FeatureExtractionModule(Module):
    def __init__(self, in_places):
        super(FeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_places, in_places, (1, 1))
        self.attention = Attention(in_places)

        self.conv2 = conv3otherRelu(in_places, in_places, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.attention(conv1)
        conv2 = self.conv2(x)
        return conv1 + conv2


class Attention(Module):
    def __init__(self, in_places, scale=4, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=(1, 1))

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, width, height)

        return (self.gamma * weight_value).contiguous()


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = conv3otherRelu(in_chan, out_chan, 1, 1, 0)
        self.conv_atten = Attention(out_chan)

    def forward(self, x):
        fcat = torch.cat(x, dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class SFNet(nn.Module):
    def __init__(self, band_num=1, resolution=None):
        super(SFNet, self).__init__()
        if resolution is None:
            resolution = [30, 50]
        self.band_num = band_num
        self.name = 'SFNet'
        self.resolution = resolution

        channels = 32

        self.conv_input = nn.Sequential(
            conv3otherRelu(self.band_num, channels),
            conv3otherRelu(channels, channels),
        )

        self.FEM1 = FeatureExtractionModule(channels)
        self.FEM2 = FeatureExtractionModule(channels)
        self.FEM3 = FeatureExtractionModule(channels)

        self.FFM = FeatureFusionModule(channels * 3, channels * 3)

        self.conv_dim = nn.Sequential(
            conv3otherRelu(channels * 3, channels),
        )

        self.conv_output = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(channels // 4, self.band_num, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=1500,
                      out_features=1500)
        )

    def forward(self, x):
        batch, channel, height, width = x.shape

        conv = self.conv_input(x)
        fem1 = self.FEM1(conv)
        fem2 = self.FEM2(fem1)
        fem3 = self.FEM3(fem2)

        ffm = self.FFM([fem1, fem2, fem3])
        ffm = self.conv_dim(ffm)
        output = self.conv_output(ffm)
        output = output.reshape(batch, channel, -1)
        output = self.linear(output)

        return output.reshape(batch, channel, height, width)


if __name__ == '__main__':
    input = torch.randn(8, 1, 30, 50)
    yaw = torch.randn(8, 1, 1)
    net = SFNet().eval()
    y = net(input).squeeze()
    print(y.shape)
