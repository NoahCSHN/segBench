import torch
import torch.nn as nn
import timm

class ResNetBackbone(nn.Module):
    def __init__(self, model_type='resnet50', pretrained=True):
        """
        Args:
            model_type: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
            pretrained: 是否加载 ImageNet 预训练权重 (强烈建议 True)
        """
        super().__init__()
        
        print(f"初始化 ResNet ({model_type})...")
        
        # === 核心魔法：features_only=True ===
        # 这会让 timm 自动返回中间层的特征列表，而不是最后的分类结果
        # out_indices=(0, 1, 2, 3) 对应 ResNet 的 layer1, layer2, layer3, layer4
        # 也就是下采样倍率为 4x, 8x, 16x, 32x 的特征图
        self.model = timm.create_model(
            model_type,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # 自动获取每个阶段的通道数
        # ResNet18/34: [64, 128, 256, 512]
        # ResNet50/101: [256, 512, 1024, 2048]
        self.embed_dims = self.model.feature_info.channels()
        
        # 为了兼容我们之前的 Head (SimpleHead/ContextHead)，
        # 我们把最后一层的通道数记录为 embed_dim
        self.embed_dim = self.embed_dims[-1]

    def forward(self, x):
        # timm 的 forward 已经返回了列表 [c1, c2, c3, c4]
        features = self.model(x)
        return features
