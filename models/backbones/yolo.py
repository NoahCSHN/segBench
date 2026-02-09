import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8Backbone(nn.Module):
    def __init__(self, model_type='yolov8n.pt', pretrained=True):
        """
        Args:
            model_type: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'
            pretrained: 是否加载 COCO 预训练权重
        """
        super().__init__()
        print(f"初始化 YOLOv8 Backbone ({model_type})...")
        
        # 1. 加载官方模型
        # 我们只加载检测模型即可，因为我们只需要它的骨干
        yolo_wrapper = YOLO(model_type)
        
        # 2. 提取核心网络 (nn.Module)
        self.model = yolo_wrapper.model
        
        # 3. 确定要提取的层
        # YOLOv8 标准架构中：
        # Layer 4  -> P3 (8x 下采样, 细节)
        # Layer 6  -> P4 (16x 下采样, 结构)
        # Layer 9  -> P5 (32x 下采样, 语义) - 这是 SPPF 层
        self.return_layers = {4: 'p3', 6: 'p4', 9: 'p5'}
        
        # 4. 自动获取通道数
        # 我们跑一次伪数据来获取通道数，这样最稳
        dummy_input = torch.randn(1, 3, 640, 640)
        self.feature_info = {}
        
        # 注册 hook 来自动获取输出维度
        hooks = []
        def get_activation(name):
            def hook(model, input, output):
                self.feature_info[name] = output.shape[1] # 记录通道数
            return hook

        # 注册 hook 到指定层
        for idx, layer in enumerate(self.model.model):
            if idx in self.return_layers:
                hooks.append(layer.register_forward_hook(get_activation(self.return_layers[idx])))
        
        # 跑一次前向传播
        with torch.no_grad():
            self.model(dummy_input)
            
        # 清理 hooks
        for h in hooks:
            h.remove()
            
        print(f"YOLOv8 特征通道: {self.feature_info}")
        # 按顺序排列通道数: [P3, P4, P5]
        self.embed_dims = [self.feature_info['p3'], self.feature_info['p4'], self.feature_info['p5']]
        self.embed_dim = self.feature_info['p5'] # 最后一层给 PPM Head 用

    def forward(self, x):
        outputs = {}
        
        # 手动遍历 YOLO 的层，提取中间结果
        # YOLOv8 的 model.model 是一个 nn.Sequential list
        for i, layer in enumerate(self.model.model):
            x = layer(x)
            if i in self.return_layers:
                outputs[self.return_layers[i]] = x
            
            # 到了第 9 层 (SPPF) 就可以停了，后面的 Head 我们不要
            if i == 9:
                break
        
        # 返回列表: [P3, P4, P5]
        return [outputs['p3'], outputs['p4'], outputs['p5']]
