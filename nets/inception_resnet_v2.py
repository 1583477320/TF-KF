import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 基础组件：Conv + BN + ReLU
# ----------------------------------------------------------------------------
class Conv2dNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, bias=bias
        )
        # 与 TF-Slim 对齐
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.0003)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ----------------------------------------------------------------------------
# Block35 (repeat)
# ----------------------------------------------------------------------------
class Block35(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super().__init__()
        self.scale = scale

        self.branch0 = Conv2dNormActivation(in_channels, 32, 1)

        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 32, 1),
            Conv2dNormActivation(32, 32, 3, padding=1),
        )

        self.branch2 = nn.Sequential(
            Conv2dNormActivation(in_channels, 32, 1),
            Conv2dNormActivation(32, 48, 3, padding=1),
            Conv2dNormActivation(48, 64, 3, padding=1),
        )

        # 注意命名必须叫 conv2d
        self.conv2d = nn.Conv2d(128, in_channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat(
            [self.branch0(x), self.branch1(x), self.branch2(x)], dim=1
        )
        out = self.conv2d(out)
        x = x + self.scale * out
        return self.relu(x)


# ----------------------------------------------------------------------------
# Block17 (repeat_1)
# ----------------------------------------------------------------------------
class Block17(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super().__init__()
        self.scale = scale

        self.branch0 = Conv2dNormActivation(in_channels, 192, 1)

        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 128, 1),
            Conv2dNormActivation(128, 160, (1, 7), padding=(0, 3)),
            Conv2dNormActivation(160, 192, (7, 1), padding=(3, 0)),
        )

        self.conv2d = nn.Conv2d(384, in_channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x)], dim=1)
        out = self.conv2d(out)
        x = x + self.scale * out
        return self.relu(x)


# ----------------------------------------------------------------------------
# Block8 (repeat_2 + block8)
# ----------------------------------------------------------------------------
class Block8(nn.Module):
    def __init__(self, in_channels, scale=1.0, no_relu=False):
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu

        self.branch0 = Conv2dNormActivation(in_channels, 192, 1)

        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, 1),
            Conv2dNormActivation(192, 224, (1, 3), padding=(0, 1)),
            Conv2dNormActivation(224, 256, (3, 1), padding=(1, 0)),
        )

        self.conv2d = nn.Conv2d(448, in_channels, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x)], dim=1)
        out = self.conv2d(out)
        x = x + self.scale * out
        if not self.no_relu:
            x = self.relu(x)
        return x


# ----------------------------------------------------------------------------
# Inception-ResNet-V2 (strict 对齐版)
# ----------------------------------------------------------------------------
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1001, create_aux_logits=True,
                 dropout_keep_prob=0.8):
        super().__init__()
        self.create_aux_logits = create_aux_logits

        # ---------------- Stem ----------------
        self.conv2d_1a = Conv2dNormActivation(3, 32, 3, stride=2, padding=0)   # 149x149
        self.conv2d_2a = Conv2dNormActivation(32, 32, 3, padding=0)           # 147x147
        self.conv2d_2b = Conv2dNormActivation(32, 64, 3, padding=1)           # 147x147
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=0)                # 73x73
        self.conv2d_3b = Conv2dNormActivation(64, 80, 1, padding=0)           # 73x73
        self.conv2d_4a = Conv2dNormActivation(80, 192, 3, padding=0)          # 71x71
        self.maxpool_5a = nn.MaxPool2d(3, stride=2, padding=0)                # 35x35

        # ---------------- mixed_5b ----------------
        self.mixed_5b = nn.ModuleDict({
            "branch0": Conv2dNormActivation(192, 96, 1),
            "branch1": nn.Sequential(
                Conv2dNormActivation(192, 48, 1),
                Conv2dNormActivation(48, 64, 5, padding=2),
            ),
            "branch2": nn.Sequential(
                Conv2dNormActivation(192, 64, 1),
                Conv2dNormActivation(64, 96, 3, padding=1),
                Conv2dNormActivation(96, 96, 3, padding=1),
            ),
            "branch3": nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1),
                Conv2dNormActivation(192, 64, 1),
            ),
        })

        # ---------------- repeat (Block35 x10) ----------------
        self.repeat = nn.Sequential(*[
            Block35(320, scale=0.17) for _ in range(10)
        ])

        # ---------------- mixed_6a (Reduction-A) ----------------
        self.mixed_6a = nn.ModuleDict({
            "branch0": Conv2dNormActivation(320, 384, 3, stride=2, padding=0),
            "branch1": nn.Sequential(
                Conv2dNormActivation(320, 256, 1),
                Conv2dNormActivation(256, 256, 3, padding=1),
                Conv2dNormActivation(256, 384, 3, stride=2, padding=0),
            ),
            "branch2": nn.MaxPool2d(3, stride=2, padding=0),
        })

        # ---------------- repeat_1 (Block17 x20) ----------------
        self.repeat_1 = nn.Sequential(*[
            Block17(1088, scale=0.10) for _ in range(20)
        ])

        # ---------------- Auxiliary logits ----------------
        if create_aux_logits:
            self.aux_logits = nn.Sequential(
                nn.AvgPool2d(5, stride=3, padding=0),
                Conv2dNormActivation(1088, 128, 1),
                Conv2dNormActivation(128, 768, 5, padding=0),
                nn.Flatten(),
                nn.Linear(768, num_classes),
            )

        # ---------------- mixed_7a (Reduction-B) ----------------
        self.mixed_7a = nn.ModuleDict({
            "branch0": nn.Sequential(
                Conv2dNormActivation(1088, 256, 1),
                Conv2dNormActivation(256, 384, 3, stride=2, padding=0),
            ),
            "branch1": nn.Sequential(
                Conv2dNormActivation(1088, 256, 1),
                Conv2dNormActivation(256, 288, 3, stride=2, padding=0),
            ),
            "branch2": nn.Sequential(
                Conv2dNormActivation(1088, 256, 1),
                Conv2dNormActivation(256, 288, 3, padding=1),
                Conv2dNormActivation(288, 320, 3, stride=2, padding=0),
            ),
            "branch3": nn.MaxPool2d(3, stride=2, padding=0),
        })

        # ---------------- repeat_2 (Block8 x9) ----------------
        self.repeat_2 = nn.Sequential(*[
            Block8(2080, scale=0.20) for _ in range(9)
        ])

        # final block8
        self.block8 = Block8(2080, scale=1.0, no_relu=True)

        # ---------------- Final conv ----------------
        self.conv2d_7b = Conv2dNormActivation(2080, 1536, 1)

        # ---------------- Classifier ----------------
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        self.logits = nn.Linear(1536, num_classes)

    # ------------------------------------------------------------------
    def forward(self, x):
        # Stem
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)

        # mixed_5b
        b0 = self.mixed_5b["branch0"](x)
        b1 = self.mixed_5b["branch1"](x)
        b2 = self.mixed_5b["branch2"](x)
        b3 = self.mixed_5b["branch3"](x)
        x = torch.cat([b0, b1, b2, b3], dim=1)

        # repeat
        x = self.repeat(x)

        # mixed_6a
        b0 = self.mixed_6a["branch0"](x)
        b1 = self.mixed_6a["branch1"](x)
        b2 = self.mixed_6a["branch2"](x)
        x = torch.cat([b0, b1, b2], dim=1)

        # repeat_1
        x = self.repeat_1(x)

        aux = None
        if self.training and self.create_aux_logits:
            aux = self.aux_logits(x)

        # mixed_7a
        b0 = self.mixed_7a["branch0"](x)
        b1 = self.mixed_7a["branch1"](x)
        b2 = self.mixed_7a["branch2"](x)
        b3 = self.mixed_7a["branch3"](x)
        x = torch.cat([b0, b1, b2, b3], dim=1)

        # repeat_2 + block8
        x = self.repeat_2(x)
        x = self.block8(x)

        # final conv + logits
        x = self.conv2d_7b(x)
        x = self.avgpool_1a(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.logits(x)

        return logits, aux
