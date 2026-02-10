import torch
import torch.nn as nn
import timm

class MobileNetBackbone(nn.Module):
    def __init__(self, model_name='mobilenetv3_large_100', pretrained=True):
        """
        Args:
            model_name: 
                - 'mobilenetv3_large_100': 标准版 (推荐)
                - 'mobilenetv3_small_100': 极速版 (精度稍低)
                - 'mobilenetv2_100': 老版本
            pretrained: 是否加载 ImageNet 权重
        """
        super().__init__()
        print(f"初始化 MobileNet ({model_name})...")

        # features_only=True 会自动找到 strides=2, 4, 8, 16, 32 的层
        # 对于 MobileNetV3，我们通常需要 indices=(1, 2, 3, 4) 对应 4x, 8x, 16x, 32x 下采样
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3, 4) 
            )
        except RuntimeError:
            # 万一你的 timm 版本太老，不支持 indices，尝试默认
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True
            )

        # 自动获取通道数
        self.channels_list = self.model.feature_info.channels()
        self.embed_dim = self.channels_list[-1]
        
        print(f"MobileNet 特征通道: {self.channels_list}")
        # MobileNetV3-Large 通常是: [24, 40, 112, 960]

    def forward(self, x):
        # 返回列表 [c1, c2, c3, c4]
        return self.model(x)
