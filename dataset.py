import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch, cv2
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
        self.split = split
        
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

        # ================= 定义增强策略 =================
        if split == 'train':
            self.transform = A.Compose([
                # 修改点：
                # 1. 显式指定 size 参数
                # 2. scale 参数改为 (0.08, 1.0) 这是 ImageNet 标准，表示裁剪面积占原图的 8% 到 100%
                #    如果你希望物体看起来更大（拉近），可以使用 (0.5, 1.0)
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3),
                
                # 2. 光照与天气 (解决水坑/倒影问题)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(p=0.5),
                
                # 3. 模糊与噪声 (模拟真实摄像头)
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                
                # 4. 遮挡 (强迫学习上下文)
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, mask_fill_value=255, p=0.3),
                
                # 5. 归一化与转 Tensor
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # 验证集只需要 Resize 和 归一化
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 1. 读取图片 (OpenCV 读取更快，Albumentations 默认用 numpy)
        image = cv2.imread(item['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转 RGB
        
        # 2. 读取 Mask
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_GRAYSCALE) # 单通道
        
        # 3. 应用增强 (核心！)
        # albumentations 会自动处理 image 和 mask 的对应关系
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].long() # 转 LongTensor

        return img_tensor, mask_tensor
