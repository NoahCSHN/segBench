import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    SegFormer 里的 MLP 层其实就是：Conv1x1 -> ReLU
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    def __init__(self, in_channels_list, num_classes, embedding_dim=256, dropout_ratio=0.1):
        """
        Args:
            in_channels_list: Backbone 4层特征的通道数，例如 [64, 128, 320, 512]
            num_classes: 类别数
            embedding_dim: 统一映射到的维度 (Decoder channel)
        """
        super().__init__()
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels_list

        # 1. 线性层 (Linear Layers)
        # 负责把不同通道数的特征 (64, 128...) 全部投影到 embedding_dim (256)
        self.linear_c4 = MLP(c4_in_channels, embedding_dim)
        self.linear_c3 = MLP(c3_in_channels, embedding_dim)
        self.linear_c2 = MLP(c2_in_channels, embedding_dim)
        self.linear_c1 = MLP(c1_in_channels, embedding_dim)

        # 2. 融合层 (Linear Fuse)
        # 拼接后通道变成 4 * 256 = 1024，再融合回 256
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 分类层 (Dropout + Conv1x1)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: [c1, c2, c3, c4] 来自 Backbone 的特征列表
        """
        c1, c2, c3, c4 = features

        ############## MLP layer for C4 ##############
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # 上采样到 c1 的尺寸 (即 1/4 原图大小)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        ############## MLP layer for C3 ##############
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        ############## MLP layer for C2 ##############
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        ############## MLP layer for C1 ##############
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 拼接融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 预测
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
