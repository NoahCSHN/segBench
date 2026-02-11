import torch
import json
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ================= 修改 Loss 部分 =================

# 1. 定义类别权重 (根据你的 Class Name 顺序)
# 原则：像素越少的类别，权重越大；像素越多的类别，权重越小。
# 0:Field, 1:Puddle, 2:Structure, 3:Veg, 4:Vehicle, 5:Wayrobo, 6:Human, 7:Basket, 8:Marker, 9:Station
# Field 设为 0.1 (因为它太多了)，Puddle/Human 设为 5.0 或 10.0
class_weights = torch.tensor([
    0.5,   # static.field (背景，降低权重)
    10.0,  # static.puddle (难点，狠狠加权)
    1.0,   # static.structure
    10.0,  # static.vegatation (0% 的那个，强制关注)
    5.0,   # dynamic.vehicle
    1.0,   # static.wayrobo (已经很好了，保持 1.0)
    10.0,  # dynamic.human (小目标，加权)
    5.0,   # dynamic.basket
    5.0,   # static.marker
    2.0    # static.station
]).to(DEVICE)

# ================= 修正后的 get_parameter_groups =================
def get_parameter_groups(model, weight_decay=1e-4, skip_list=(), lr=1e-4, layer_decay=0.75):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 1. 基础分组：是否衰减权重
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # 2. 分层 LR 逻辑 (针对 ViT/DINO)
        layer_id = 12 # 默认给 Head (最高层)
        
        if "backbone" in name:
            if "patch_embed" in name:
                layer_id = 0
            elif "blocks" in name:
                # 解析 blocks.0, blocks.1 ...
                import re
                match = re.search(r"blocks\.(\d+)", name)
                if match:
                    layer_id = int(match.group(1)) + 1
        
        # 计算该层的 scale
        scale = layer_decay ** (12 - layer_id)
        
        group_key = (group_name, layer_id)
        
        # 初始化字典
        if group_key not in parameter_group_names:
            parameter_group_names[group_key] = {
                "weight_decay": this_weight_decay,
                "params": [], # 这里初始化是空的
                "lr": lr * scale
            }
            parameter_group_vars[group_key] = []
            
        # 收集参数
        parameter_group_vars[group_key].append(param)
    
    # 【核心修复】：把收集到的参数塞回配置字典里！
    ret = []
    for key, config in parameter_group_names.items():
        config["params"] = parameter_group_vars[key] # <--- 这一步你之前漏了
        ret.append(config)
        
    return ret

# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        valid_mask = (target != self.ignore_index)
        target = target * valid_mask.long()
        num_classes = pred.shape[1]
        target_one_hot = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        valid_mask = valid_mask.unsqueeze(1).expand_as(pred)
        intersection = (pred * target_one_hot * valid_mask).sum(dim=(2, 3))
        union = (pred * valid_mask).sum(dim=(2, 3)) + (target_one_hot * valid_mask).sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# 组合 Loss
# ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
# dice_loss = DiceLoss(ignore_index=255)

# 在训练循环里：
# loss = 0.5 * ce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
