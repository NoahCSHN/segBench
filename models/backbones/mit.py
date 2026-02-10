import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# ================= 基础组件 =================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1) # 使用Conv2d代替Linear，方便处理空间信息
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 空间缩减 (Spatial Reduction) - SegFormer 的核心
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, head_dim, qk_scale, attn_drop, drop, sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # Attention 部分
        x = x + self.attn(self.norm1(x), H, W)
        
        # MLP 部分 (需要 Reshape 成图片格式再变回来)
        B, N, C = x.shape
        x_img = x.transpose(1, 2).view(B, C, H, W)
        x_img = x_img + self.mlp(self.norm2(x).transpose(1, 2).view(B, C, H, W))
        x = x_img.flatten(2).transpose(1, 2)
        return x

class OverlapPatchEmbed(nn.Module):
    """ 重叠的 Patch Embedding，这是 SegFormer 能够处理密集预测的关键 """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ================= 主模型类 =================
class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], head_dim=32):
        super().__init__()
        self.depths = depths

        # Patch Embeddings (4个阶段)
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], head_dim=head_dim, mlp_ratio=mlp_ratios[0], qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], head_dim=head_dim, mlp_ratio=mlp_ratios[1], qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], head_dim=head_dim, mlp_ratio=mlp_ratios[2], qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], head_dim=head_dim, mlp_ratio=mlp_ratios[3], qk_scale=qk_scale,
                      drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # 记录每层的通道数，方便 Head 使用
        self.embed_dims = embed_dims
        self.embed_dim = embed_dims[-1]

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # 返回 [1/4, 1/8, 1/16, 1/32] 四层特征
        return outs

# ================= 预置配置 =================
# 这是 mit_b1 的标准配置
def mit_b1(pretrained=True):
    model = MixVisionTransformer(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1
    )
    
    if pretrained:
        # 自动下载 mit_b1 权重 (来自 SegFormer 官方)
        url = "https://github.com/bismex/SegFormer-PyTorch/releases/download/v1.0/mit_b1.pth"
        print(f"正在加载 mit_b1 预训练权重: {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print("mit_b1 权重加载成功！")
        
    return model

# 这是 mit_b0 (更小，适合极端边缘设备)
def mit_b0(pretrained=True):
    model = MixVisionTransformer(
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1
    )
    if pretrained:
        url = "https://github.com/bismex/SegFormer-PyTorch/releases/download/v1.0/mit_b0.pth"
        print(f"正在加载 mit_b0 预训练权重...")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model
