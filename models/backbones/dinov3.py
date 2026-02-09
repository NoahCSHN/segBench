import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os,sys

# ================= 配置区域 =================
# 请修改为你存放 dinov3 源码的实际文件夹路径
# 假设你的源码文件夹叫 'dinov3_source'，并且就在当前目录下
DINOV3_SOURCE_PATH = os.path.join('/home/wayrobo/0_code/dinov3')
# ===========================================

# 1. 动态添加源码路径到系统路径，这样 Python 才能找到它
if DINOV3_SOURCE_PATH not in sys.path:
    sys.path.append(DINOV3_SOURCE_PATH)

# 2. 尝试导入 DINOv3 的模型定义
# 注意：你需要打开你的 dinov3 源码文件夹，找到定义模型的文件（通常叫 vision_transformer.py）
# 确认里面的类名或函数名。这里假设是标准的 vit_large 或 vit_base
try:
    # 这里的导入语句可能需要根据你下载的源码实际情况微调
    # 例如：from vision_transformer import vit_large, vit_base
    from dinov3.hub.backbones import dinov3_vits16, dinov3_convnext_tiny 
except ImportError as e:
    print(f"【错误】无法导入 DINOv3 源码。请检查路径: {DINOV3_SOURCE_PATH}")
    print(f"具体报错: {e}")
    # 为了防止代码崩溃，定义一个假的函数，实际运行时会报错
    vit_large = None

class Dinov3TransformerBackbone(nn.Module):
    def __init__(self, weight_path, model_type='vit', img_size=512):
        """
        Args:
            weight_path: .pth 权重文件的路径
            model_type: 'vit_large' 或 'vit_base' (根据你下载的权重决定)
            img_size: 输入图片大小 (DINOv3 需要固定尺寸，如 512)
            patch_size: 切片大小 (通常是 14)
        """
        super().__init__()
        
        print(f"正在初始化 DINOv3 ({model_type})...")
        
        # 1. 初始化模型结构
        if model_type == 'vit':
            self.model = dinov3_vits16()
        elif model_type == 'cnn':
            self.model = dinov3_convnext_tiny()
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        # 2. 加载本地权重
        print(f"正在加载权重: {weight_path} ...")
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            
            # 处理权重字典的 key（移除 'module.', 'backbone.' 等前缀，移除 head）
            # 不同的权重文件结构可能不同，这里做一个通用的清洗
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 过滤掉不需要的 head 层权重
            clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if "head" not in k}
            
            # 加载权重 (strict=False 允许忽略一些不匹配的层，比如位置编码微小的差异)
            msg = self.model.load_state_dict(clean_state_dict, strict=False)
            print(f"权重加载报告: {msg}")
            
        except Exception as e:
            print(f"【严重错误】权重加载失败: {e}")
            raise e

        self.patch_size = 16 
        self.embed_dim = self.model.embed_dim # 自动获取通道数 (Large是1024, Base是768)

    def forward(self, x):
        """
        输入: (Batch, 3, H, W)
        输出: List [ (Batch, Embed_Dim, H/14, W/14) ]
        """
        B, C, H, W = x.shape
        
        # 1. 前向传播提取特征 (输出 shape 通常是 B, N, D)
        # 注意：这里调用的是 forward_features，不是 forward
        outputs = self.model.forward_features(x)

        # 2. 【关键修复】处理 DINOv2 的字典输出
        if isinstance(outputs, dict):
            # DINOv2 官方代码通常把 Patch 特征放在这个 key 里
            # 注意：x_norm_patchtokens 已经自动去掉了 CLS token，是纯粹的 Patch
            x = outputs['x_norm_patchtokens']
        else:
            # 以防万一某些版本返回的是 Tensor
            x = outputs

        # 3. 处理输出序列
        # DINOv3 可能有 1 个 CLS token + 4 个 Register tokens
        # 我们需要计算出 patch 的数量，然后只取 patch 部分
        h_grid = H // self.patch_size
        w_grid = W // self.patch_size
        num_patches = h_grid * w_grid
        
        # 如果取出来的特征比 patch 数还多 (比如某些版本没去 CLS)，则手动切片
        if x.shape[1] > num_patches:
             x = x[:, -num_patches:, :]

        # 4. 序列重塑为图片 (Reshape & Permute)
        # (B, N, D) -> (B, H_grid, W_grid, D) -> (B, D, H_grid, W_grid)
        x = x.reshape(B, h_grid, w_grid, self.embed_dim).permute(0, 3, 1, 2)

        # 为了兼容后续可能的金字塔结构，我们返回一个列表
        return [x]
