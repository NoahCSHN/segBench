import torch
import torch.nn as nn
import timm

class UniversalBackbone(nn.Module):
    def __init__(self, model_name='convnext_tiny', pretrained=True):
        """
        万能 Backbone 包装器，支持：
        - ConvNeXt: 'convnext_tiny', 'convnext_small'
        - Swin: 'swin_tiny_patch4_window7_224'
        - MiT (SegFormer): 'mit_b0', 'mit_b1', 'mit_b2'
        - ResNet: 'resnet18', 'resnet50'
        """
        super().__init__()
        print(f"初始化 Backbone: {model_name} ...")
        
        # timm 的 features_only=True 接口对这些模型完全统一
        # 它们都会自动返回 4 层特征列表
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3) # 提取 1/4, 1/8, 1/16, 1/32
        )
        
        # 自动获取通道数
        self.channels_list = self.model.feature_info.channels()
        self.embed_dim = self.channels_list[-1]
        
        print(f"特征通道: {self.channels_list}")

    def forward(self, x):
        # 返回列表 [c1, c2, c3, c4]
        return self.model(x)
