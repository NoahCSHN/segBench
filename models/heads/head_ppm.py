import torch
import torch.nn as nn
import torch.nn.functional as F

class PPM(nn.Module):
    """Pyramid Pooling Module: 聚合全局上下文信息"""
    def __init__(self, in_channels, reduction_dim, bins):
        super().__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channels, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            # 池化 -> 卷积 -> 双线性插值回原尺寸
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=False))
        # 拼接所有尺度的特征
        return torch.cat(out, 1)

class ContextSegHead(nn.Module):
    """
    增加一个Pyramid Pooling Module层，将特征图分成 1x1, 2x2, 3x3, 6x6 四个尺度，强行融合全局信息（解决“像水又像天”的问题）
    FPN融合，将高层语义特征与底层细节特征融合
    """
    def __init__(self, in_channels, num_classes, embedding_dim=512):
        super().__init__()
        
        # 1. 降维层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. PPM 模块 (1x1, 2x2, 3x3, 6x6)
        # 输入 dim, 每个分支输出 dim/4, 总输出 = dim + 4*(dim/4) = 2*dim
        self.ppm = PPM(embedding_dim, int(embedding_dim / 4), bins=[1, 2, 3, 6])
        
        # 3. 解码融合层
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim * 2, embedding_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), # 防止过拟合
            nn.Conv2d(embedding_dim, num_classes, 1)
        )

    def forward(self, features):
        # DINOv3 返回的是列表 [x]，我们取第一个
        x = features[-1]
        
        x = self.bottleneck(x)
        x = self.ppm(x)
        logits = self.decoder(x)
        
        return logits

class PPMHead_ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, embedding_dim=512):
        """
        Args:
            in_channels: Backbone 最后一层的通道数 (ResNet50 是 2048)
            num_classes: 类别数
            embedding_dim: 中间层维度
        """
        super().__init__()
        
        # PPM 模块：包含 1x1, 2x2, 3x3, 6x6 四个尺度
        # 降维后的通道数通常是输入通道的 1/4 或固定值
        reduction_dim = int(in_channels / 4) if in_channels >= 512 else 128
        self.ppm = PPM(in_channels, reduction_dim, bins=[1, 2, 3, 6])
        
        # PPM 输出总通道数 = 输入 + 4 * reduction_dim
        ppm_out_channels = in_channels + 4 * reduction_dim
        
        # 解码层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ppm_out_channels, embedding_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        )

    def forward(self, features):
        """
        Args:
            features: Backbone 返回的列表 [c1, c2, c3, c4]
        """
        # PPM 只需要利用语义信息最强的最后一层 (c4)
        x = features[-1]
        
        # 1. 金字塔池化融合
        x = self.ppm(x)
        
        # 2. 最终分类
        logits = self.bottleneck(x)
        
        return logits
