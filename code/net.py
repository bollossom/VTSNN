import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate


####################################################################
# --------------------  1 VTSNN-IF  ------------------------------ #
####################################################################


# ====================  1-1) VTSNN-IF Conv Layer   ================ #


class Conv_Layer_3x3_with_IF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(Conv_Layer_3x3_with_IF, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=padding,
                               bias=True)

        self.IFNode1 = neuron.IFNode(v_threshold=1.0, v_reset=None,
                                     surrogate_function=surrogate.ATan())

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=padding,
                               bias=True)

        self.IFNode2 = neuron.IFNode(v_threshold=1.0, v_reset=None,
                                     surrogate_function=surrogate.ATan())

    def forward(self, x):

        out = self.conv1(x)
        out = self.IFNode1(out)
        out = self.conv2(out)
        out = self.IFNode2(out)

        return out


# ================   1-2) VTSNN-IF Upsample Layer  ================= #

class Upsample_Layer_with_IF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(Upsample_Layer_with_IF, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides,
                                        bias=True)

        self.IFNode = neuron.IFNode(v_threshold=1.0, v_reset=None,
                                    surrogate_function=surrogate.ATan())

    def forward(self, x):
        out = self.conv1(x)
        out = self.IFNode(out)
        return out


# ============= 1-3) The Overall VTSNN-IF Net ============== #

class VTSNN_IF(nn.Module):
    def __init__(self, last_layer_threshold):
        super(VTSNN_IF, self).__init__()
        self.last_layer_threshold = last_layer_threshold
        self.conv_layer_3x3_1_2 = Conv_Layer_3x3_with_IF(1, 16)
        self.conv_layer_3x3_3_4 = Conv_Layer_3x3_with_IF(16, 32)
        self.conv_layer_3x3_5_6 = Conv_Layer_3x3_with_IF(32, 64)
        self.conv_layer_3x3_7_8 = Conv_Layer_3x3_with_IF(64, 32)
        self.conv_layer_3x3_9_10 = Conv_Layer_3x3_with_IF(32, 16)
        self.conv_layer_1x1_conv = nn.Sequential(nn.Conv2d(16, 1,
                                                           kernel_size=1, stride=1, padding=0),
                            )
        self.upsample_2x2_1 = Upsample_Layer_with_IF(64, 32)
        self.upsample_2x2_2 = Upsample_Layer_with_IF(32, 16)
        self.conv_layer_1x1_IF = neuron.IFNode(v_threshold=last_layer_threshold,
                                               v_reset=None, surrogate_function=surrogate.ATan())

    def forward(self, x):
        x1 = self.conv_layer_3x3_1_2(x)
        p1 = F.max_pool2d(x1, 2)

        x2 = self.conv_layer_3x3_3_4(p1)
        p2 = F.max_pool2d(x2, 2)

        x4 = self.conv_layer_3x3_5_6(p2)

        up1 = self.upsample_2x2_1(x4)
        concat1 = torch.cat([up1, x2], dim=1)
        x5 = self.conv_layer_3x3_7_8(concat1)

        up2 = self.upsample_2x2_2(x5)
        concat2 = torch.cat([up2, x1], dim=1)
        x6 = self.conv_layer_3x3_9_10(concat2)

        x7 = self.conv_layer_1x1_conv(x6)
        out = self.conv_layer_1x1_IF(x7)

        return out


####################################################################
# ---------------------  2 VTSNN-LIF ----------------------------- #
####################################################################


# ===================== 2-1) VTSNN-LIF Conv Layer ================ #

class Conv_Layer_3x3_with_LIF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(Conv_Layer_3x3_with_LIF, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=padding,
                               bias=True)

        self.LIFNode1 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=1.1,
                                       surrogate_function=surrogate.ATan())

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=padding,
                               bias=True)

        self.LIFNode2 = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=1.1,
                                       surrogate_function=surrogate.ATan())

    def forward(self, x):

        out = self.conv1(x)
        out = self.LIFNode1(out)
        out = self.conv2(out)
        out = self.LIFNode2(out)

        return out


# =================== 2-2) VTSNN-LIF Upsample Layer ================ #


class Upsample_Layer_with_LIF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(Upsample_Layer_with_LIF, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides,
                                        bias=True)

        self.LIFNode = neuron.LIFNode(v_threshold=1.0, v_reset=None, tau=1.1,
                                      surrogate_function=surrogate.ATan())

    def forward(self, x):
        out = self.conv1(x)
        # print(out)
        out = self.LIFNode(out)
        return out


# ============ 2-3) The Overall VTSNN-LIF Net ============== #

class VTSNN_LIF(nn.Module):
    def __init__(self, last_layer_threshold):
        super(VTSNN_LIF, self).__init__()
        self.conv_layer_3x3_1_2 = Conv_Layer_3x3_with_LIF(1, 16)
        self.conv_layer_3x3_3_4 = Conv_Layer_3x3_with_LIF(16, 32)
        self.conv_layer_3x3_5_6 = Conv_Layer_3x3_with_LIF(32, 64)
        self.conv_layer_3x3_7_8 = Conv_Layer_3x3_with_LIF(64, 32)
        self.conv_layer_3x3_9_10 = Conv_Layer_3x3_with_LIF(32, 16)
        self.conv_layer_1x1_conv = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
                                      )
        self.upsample_2x2_1 = Upsample_Layer_with_LIF(64, 32)
        self.upsample_2x2_2 = Upsample_Layer_with_LIF(32, 16)
        self.conv_layer_1x1_LIF = neuron.LIFNode(v_threshold=last_layer_threshold, v_reset=None, tau=1.1, surrogate_function=surrogate.ATan())

    def forward(self, x):
        x1 = self.conv_layer_3x3_1_2(x)
        p1 = F.max_pool2d(x1, 2)

        x2 = self.conv_layer_3x3_3_4(p1)
        p2 = F.max_pool2d(x2, 2)

        x4 = self.conv_layer_3x3_5_6(p2)

        up1 = self.upsample_2x2_1(x4)
        concat1 = torch.cat([up1, x2], dim=1)
        x5 = self.conv_layer_3x3_7_8(concat1)

        up2 = self.upsample_2x2_2(x5)
        concat2 = torch.cat([up2, x1], dim=1)
        x6 = self.conv_layer_3x3_9_10(concat2)

        x7 = self.conv_layer_1x1_conv(x6)
        out = self.conv_layer_1x1_LIF(x7)

        return out


#######################################################################

model = VTSNN_LIF(last_layer_threshold=0.077)
x = torch.randn(1,1,32,32)
print(model(x).shape)
