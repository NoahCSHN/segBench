import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=512):
        """
        Args:
            in_channels: Backbone 输出的通道数 (例如 1024)
            num_classes: 你的类别数 (例如 9)
            embed_dim: 中间层的通道数
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            # 1. 降维/特征融合 (Conv 3x3 + BN + ReLU)
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            
            # 2. 最终分类 (Conv 1x1)
            nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        )

    def forward(self, features):
        """
        输入: features 是一个列表，我们取最后一个元素
        """
        # 取出 Backbone 的输出
        x = features[-1] 
        
        # 通过分类层
        logits = self.layers(x)
        
        return logits
