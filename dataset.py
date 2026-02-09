import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random

class NuScenesSegDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=512, ignore_index=255):
        """
        Args:
            data_root: 数据集根目录 (包含 samples 和 gt_masks 的那一级)
            split: 'train' 或 'val' (这里我们简单按比例划分)
            img_size: 训练图片大小 (需要是 14 的倍数，推荐 518 或 512)
        """
        self.data_root = data_root
        self.img_size = img_size
        self.ignore_index = ignore_index
        
        # 1. 定义路径
        self.img_dir = os.path.join(data_root, 'samples', 'CAM_FRONT')
        self.mask_dir = os.path.join(data_root, 'gt_masks')
        
        # 2. 扫描所有文件
        # 我们假设 mask 文件名和 image 文件名是一一对应的（除了后缀）
        self.items = []
        if not os.path.exists(self.img_dir):
            raise ValueError(f"找不到图片目录: {self.img_dir}")

        all_files = sorted(os.listdir(self.img_dir))
        
        # 3. 简单的划分 Train/Val (8:2)
        # 为了保证实验可复现，我们用固定种子打乱
        random.seed(42)
        random.shuffle(all_files)
        
        split_idx = int(len(all_files) * 0.8)
        if split == 'train':
            file_list = all_files[:split_idx]
        else:
            file_list = all_files[split_idx:]
            
        # 4. 建立索引
        for file_name in file_list:
            if not file_name.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            # 构造对应的 mask 文件名
            # 你的生成脚本里，image是 .jpg, mask是 .png，且前缀相同
            base_name = os.path.splitext(file_name)[0]
            mask_name = base_name + ".png"
            
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            # 只有当 mask 存在时才加入训练列表
            if os.path.exists(mask_path):
                self.items.append({
                    'img_path': os.path.join(self.img_dir, file_name),
                    'mask_path': mask_path
                })
        
        print(f"[{split}] 加载完成，共 {len(self.items)} 张图片。")

        # 5. 定义预处理 (ImageNet 标准化)
        # DINOv3 预训练时使用的是 ImageNet 的均值和方差
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 1. 读取图片和掩码
        image = Image.open(item['img_path']).convert('RGB')
        mask = Image.open(item['mask_path']) # 这里的 mask 应该是单通道索引图 (0, 1, 2...)

        # 2. 调整大小 (Resize)
        # 注意：Mask 必须使用 Nearest 插值，不能破坏类别索引值
        image = F.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        # 3. 数据增强 (仅对训练集)
        # 这里演示最简单的随机翻转
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # 4. 转 Tensor
        img_tensor = self.normalize(image)
        mask_array = np.array(mask, dtype=np.int64) # 转成 numpy
        mask_tensor = torch.from_numpy(mask_array).long() # 转成 LongTensor

        return img_tensor, mask_tensor
