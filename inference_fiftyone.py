import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import fiftyone as fo
from tqdm import tqdm

# === 引入你之前的模型定义 ===
# 确保 models 文件夹在当前目录下
from models.backbones.dinov3 import Dinov3TransformerBackbone
from models.heads.my_head import SimpleSegHead
from models.heads.head_ppm import ContextSegHead

# ================= 配置区域 =================
# 1. 路径配置
# 验证集图片的文件夹 (只放图片，不需要放标注)
VAL_IMAGE_DIR = "/home/wayrobo/0_code/segment-anything-2/sav_dataset/0_poly_DrivingRange/workflow" 
# 训练好的权重文件
CHECKPOINT_PATH = "checkpoints/VITS16_PPM_epoch_10.pth" # 替换为你实际的权重路径
# DINOv3 预训练权重路径 (Backbone 初始化还需要用到它)
DINO_WEIGHT_PATH = "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 
# fiftyone dataset name
FIFTYONE_DATASET_NAME = "DINOV3_VITS16_PPM_GOLF_WORKFLOW"

# 2. 模型参数 (必须与训练时一致)
IMG_SIZE = 512
NUM_CLASSES = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 类别映射 (用于 FiftyOne 显示图例)
# Key: 预测的 ID (0-8), Value: 类别名称
ID_TO_LABEL = {
    0: "static.field",
    1: "static.puddle",
    2: "static.structure",
    3: "static.vegatation",
    4: "dynamic.vehicle",
    5: "static.wayrobo",
    6: "dynamic.human",
    7: "dynamic.backet",
    8: "static.marker"
}
# ===========================================

def get_model():
    """重建模型结构并加载训练权重"""
    print("正在构建模型...")
    backbone = Dinov3TransformerBackbone(
        weight_path=DINO_WEIGHT_PATH,
        model_type='vit', # 确保与训练时一致
        img_size=IMG_SIZE
    )
    head = ContextSegHead(
        in_channels=backbone.embed_dim, 
        num_classes=NUM_CLASSES
    )
    
    # 定义简单的包装类 (与 trains.py 里的 SegModel 一致)
    class SegModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            feats = self.backbone(x)
            logits = self.head(feats)
            return logits

    model = SegModel(backbone, head)
    
    # 加载你训练好的 Checkpoint
    print(f"加载训练权重: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def main():
    # 1. 准备模型
    model = get_model()

    # 2. 定义预处理 (必须与训练时的 transform 一致)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 创建 FiftyOne 数据集
    dataset_name = FIFTYONE_DATASET_NAME
    
    # 如果数据集已存在，先删除 (方便重复运行)
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(dataset_name)
    
    # 4. 开始推理
    image_files = [f for f in os.listdir(VAL_IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort()
    
    samples = []
    print(f"开始推理 {len(image_files)} 张图片...")

    with torch.no_grad():
        for img_file in tqdm(image_files):
            img_path = os.path.join(VAL_IMAGE_DIR, img_file)
            
            # --- 读取与预处理 ---
            original_img = Image.open(img_path).convert('RGB')
            w, h = original_img.size # 记录原图尺寸
            
            input_tensor = transform(original_img).unsqueeze(0).to(DEVICE) # (1, 3, 512, 512)
            
            # --- 模型推理 ---
            logits = model(input_tensor) # (1, 9, 36, 36) (取决于 Head 输出尺寸)
            
            # 上采样回 **原图尺寸** (这一点对可视化很重要)
            # 我们直接插值到 (h, w)
            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            # 获取预测结果 (Argmax) -> (h, w)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            # --- 创建 FiftyOne 样本 ---
            sample = fo.Sample(filepath=img_path)
            
            # 添加预测结果
            # FiftyOne 只需要二维的 uint8 数组即可
            sample["prediction"] = fo.Segmentation(
                mask=pred_mask
            )
            
            samples.append(sample)

    # 5. 添加样本到数据集
    dataset.add_samples(samples)
    
    # 6. 设置可视化样式 (Mask Targets)
    # 让 FiftyOne 知道 ID 5 是 "wayrobo" 并自动分配颜色
    dataset.default_mask_targets = ID_TO_LABEL
    
    # 持久化保存
    dataset.save()

    print("推理完成！正在启动 FiftyOne App...")
    
    # 7. 启动 App
    session = fo.launch_app(dataset, port=5151, address="0.0.0.0", auto=False)
    
    # 8. 安全挂起
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭...")
    finally:
        session.close()
        print("✅ 服务已安全关闭。")

if __name__ == "__main__":
    main()
