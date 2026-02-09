import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# 导入我们自己写的模块
from models.backbones.dinov3 import Dinov3TransformerBackbone
from models.heads.my_head import SimpleSegHead
from dataset import NuScenesSegDataset

# ================= 配置 =================
DATA_ROOT = "/home/wayrobo/0_code/segment-anything-2/nuScene_golf_dataset" # 【修改这里】你的数据集路径
WEIGHT_PATH = "/home/wayrobo/0_code/dinov3/pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"   # 【修改这里】你的 DINOv3 权重路径
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 9  # 你的类别数
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 1. 准备数据 =================
print("正在准备数据...")
train_dataset = NuScenesSegDataset(DATA_ROOT, split='train', img_size=IMG_SIZE)
val_dataset = NuScenesSegDataset(DATA_ROOT, split='val', img_size=IMG_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
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
head = SimpleSegHead(
    in_channels=backbone.embed_dim, 
    num_classes=NUM_CLASSES
)

# 组合模型
class SegModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        # 上采样回原图大小 (因为 Head 输出是 1/14)
        return torch.nn.functional.interpolate(
            logits, size=x.shape[2:], mode='bilinear', align_corners=False
        )

model = SegModel(backbone, head).to(DEVICE)

# ================= 3. 定义优化器和损失 =================
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=255) # 忽略背景或无效值

# ================= 4. 训练循环 =================
print(f"开始训练，设备: {DEVICE}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
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
        
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} 完成，平均 Loss: {epoch_loss / len(train_loader):.4f}")
    
    # === 简单的验证 (可选) ===
    # 这里可以加代码跑 val_loader 计算 IoU
    
    # 保存模型
    torch.save(model.state_dict(), f"checkpoints/seg_model_epoch_{epoch+1}.pth")

print("训练结束！")
