import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import class_weights, get_parameter_groups
# 导入我们自己写的模块
from models.backbones.dinov3 import Dinov3TransformerBackbone
from models.backbones.resnet import ResNetBackbone
from models.backbones.yolo import YOLOv8Backbone
from models.backbones.universal import UniversalBackbone
from models.backbones.mit import mit_b0, mit_b1 
from models.backbones.mobilenet import MobileNetBackbone 
from models.heads.my_head import SimpleSegHead
from models.heads.head_ppm import ContextSegHead, PPMHead_ResNet
from models.heads.head_segformer import SegFormerHead 
from dataset import NuScenesSegDataset
from evaluator import Evaluator

# ================= 配置 =================
DATA_ROOT = "/home/wayrobo/0_code/segment-anything-2/nuScene_golf_dataset_v1p1" # 【修改这里】你的数据集路径
WEIGHT_PATH = "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"   # 【修改这里】你的 DINOv3 权重路径
#CHECKPOINT_NAME = "MOBILENET_LARGE_PPM_datasetv1p1"
CHECKPOINT_NAME = "VITS16_PPM_datasetv1p1"
#CHECKPOINT_NAME = "CONVNEXT_TINY_PPM_datasetv1p1"
#CHECKPOINT_NAME = "YOLOV8_PPM_datasetv1p1"
#CHECKPOINT_NAME = "RESNET18_PPM_datasetv1p1"
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
NUM_CLASSES = 10  # 你的类别数
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = [ 
    "static.field",
    "static.puddle",
    "static.structure",
    "static.vegatation",
    "dynamic.vehicle",
    "static.wayrobo",
    "dynamic.human",
    "dynamic.basket",
    "static.marker",
    "static.station"
]

# ================= 1. 准备数据 =================
print("正在准备数据...")
train_dataset = NuScenesSegDataset(DATA_ROOT, split='train', img_size=IMG_SIZE)
val_dataset = NuScenesSegDataset(DATA_ROOT, split='val', img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # 验证集 Batch=1 方便计算 IoU

# ================= 2. 准备模型 =================
print("正在构建模型...")
# Backbone
backbone = Dinov3TransformerBackbone(
    weight_path=WEIGHT_PATH,
    model_type='vit', # 确保和你下载的权重匹配
    img_size=IMG_SIZE
)
# Head
#head = SimpleSegHead(
#    in_channels=backbone.embed_dim, 
#    num_classes=NUM_CLASSES
#)

head = ContextSegHead(
    in_channels=backbone.embed_dim,
    num_classes=NUM_CLASSES
)

"""
# Backbone
backbone = ResNetBackbone(
    model_type='resnet18' # 确保和你下载的权重匹配
)
# Head
head = PPMHead_ResNet(
    in_channels=backbone.embed_dim,
    num_classes=NUM_CLASSES,
    embedding_dim=512
)

# Backbone
backbone = YOLOv8Backbone(
    model_type='yolov8s.pt', # 确保和你下载的权重匹配
    pretrained=True
)
# Head
head = PPMHead_ResNet(
    in_channels=backbone.embed_dim,
    num_classes=NUM_CLASSES,
    embedding_dim=256
)
# === 选项 A: ConvNeXt-Tiny (推荐作为主力) ===
# 精度比 ResNet50 高，速度差不多
backbone = UniversalBackbone('convnext_tiny')

# === 选项 B: SegFormer-B1 (推荐用于 Jetson) ===
# 速度极快，专门为分割设计
# backbone = UniversalBackbone('mit_b1')

# === 选项 C: Swin-Tiny (高精度) ===
# 如果显存够，可以试试这个
# backbone = UniversalBackbone('swin_tiny_patch4_window7_224')

# Head 依然使用 PPMHead 或 UPerHead (推荐 UPerHead 因为这些模型都输出4层特征)
# 如果用 PPMHead，它会自动取最后一层
head = PPMHead_ResNet(
    in_channels=backbone.embed_dim, 
    num_classes=NUM_CLASSES
)

# 1. 实例化 Backbone (使用 MiT-B1，官方权重会自动下载)
# mit_b1 的通道定义是: [64, 128, 320, 512]
backbone = mit_b1(pretrained=True)
# 2. 实例化 Head
# 注意：一定要把通道列表传给 Head，因为它需要对每一层做映射
head = SegFormerHead(
    in_channels_list=backbone.embed_dims, # [64, 128, 320, 512]
    num_classes=NUM_CLASSES,
    embedding_dim=256 # SegFormer 默认是 256 (B0/B1) 或 768 (B2-B5)
)
backbone = MobileNetBackbone('mobilenetv3_large_100', pretrained=True)
# 2. 实例化 Head
# 注意：一定要把通道列表传给 Head，因为它需要对每一层做映射
head = SegFormerHead(
    in_channels_list=backbone.channels_list, # [64, 128, 320, 512]
    num_classes=NUM_CLASSES,
    embedding_dim=256 # SegFormer 默认是 256 (B0/B1) 或 768 (B2-B5)
)
"""


# 组合模型
class SegModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        input_shape = x.shape[2:]
        feats = self.backbone(x)
        logits = self.head(feats)
        # 上采样回原图大小 (因为 Head 输出是 1/14)
        return torch.nn.functional.interpolate(
            logits, size=input_shape , mode='bilinear', align_corners=False
        )

model = SegModel(backbone, head).to(DEVICE)

# ================= 3. 定义优化器和损失 =================
#optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
#criterion = nn.CrossEntropyLoss(ignore_index=255) # 忽略背景或无效值

# ================= 修改优化器定义 =================
# 使用分层学习率：Backbone 学习率会自动按 0.75 衰减，Head 保持 1e-4
# 注意：我们把基础 LR 稍微调大一点到 2e-4，因为 Backbone 会被衰减得很小
optimizer = optim.AdamW(
    get_parameter_groups(model, lr=2e-4, weight_decay=0.05, layer_decay=0.8), 
    betas=(0.9, 0.999)
)
# 将权重传给 Loss
criterion = nn.CrossEntropyLoss(
    weight=class_weights, 
    ignore_index=255
)

# ================= 4. 训练循环 =================
print(f"开始训练，设备: {DEVICE}")
evaluator = Evaluator(NUM_CLASSES)
best_miou = 0.0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    train_steps = len(train_loader)
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(total=train_steps, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch")
    
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        pbar.update(1)
        
    pbar.close()
    avg_train_loss = epoch_loss / train_steps
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
    
    # --- Validation ---
    model.eval()
    evaluator.reset()
    val_loss = 0

    print(f"Epoch {epoch+1} 完成，平均 Loss: {epoch_loss / len(train_loader):.4f}")
    
    # 这里可以加代码跑 val_loader 计算 IoU
    print(f"正在验证 Epoch {epoch+1} ...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="[Val]"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            # 获取预测结果 (argmax)
            preds = torch.argmax(outputs, dim=1)
            
            # 转为 numpy 并累积混淆矩阵
            # 注意: 这里的 masks 可能包含 255，Evaluator 内部会处理
            evaluator.add_batch(masks.cpu().numpy(), preds.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    ious = evaluator.evaluate()
    miou = np.nanmean(ious)
    
    # --- 打印报表 ---
    print("-" * 50)
    print(f"Epoch {epoch+1} Validation Report:")
    print(f"Val Loss: {avg_val_loss:.4f}")
    print(f"mIoU:     {miou:.2%}")
    print("-" * 50)
    print(f"{'Class Name':<15} | {'IoU':<10}")
    print("-" * 30)
    for i, iou in enumerate(ious):
        print(f"{CLASS_NAMES[i]:<15} | {iou:.2%}")
    print("-" * 50)
    
    # 保存最佳模型
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), f"checkpoints/{CHECKPOINT_NAME}_best.pth")
        print(f"★ 新的最佳模型已保存 (mIoU: {best_miou:.2%})")
    
    # 保存最新模型
    torch.save(model.state_dict(), f"checkpoints/{CHECKPOINT_NAME}_last.pth")

print("训练结束！")
