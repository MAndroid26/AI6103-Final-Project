import torch
import torch.nn as nn
from diff_aug import DiffAugment


def upsampling(x, H, W):
    B, N, C = x.size()  # imput mini-batch x of shape (batch size, pic len, hidden size), N=H*W
    x = x.permute(0, 2, 1)
    x = x.view(B, C, H, W)  # reshapes the 1D sequence of token embedding back to a 2D feature map
    x = nn.PixelShuffle(2)(x)  # Upscale
    B, C, H, W = x.size()
    x = x.view(B, C, H * W)  # reshapes 2D to 1D
    x = x.permute(0, 2, 1)
    return x


# Modified Normalization
class ModifiedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        norm = input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)
        return norm


# Feed-forward MLP with GELU non-linearity
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Multi-head #self-attention module
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = 1. / dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Basic block of transformer
# The first part is constructed by a multi-head self-attention module and the second part is a feed-forward MLP with GELU non-linearity.
# The normalization layer is applied before both of the two parts. Both parts employ residual connection.
class Trans_Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()

        self.normlayer1 = ModifiedNorm(dim)
        self.selfattention = Attention(dim, heads, drop_rate, drop_rate)
        self.normlayer2 = ModifiedNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, drop_rate)

    def forward(self, x):
        x = x + self.selfattention(self.normlayer1(x))
        x = x + self.mlp(self.normlayer2(x))
        return x


# Stage consist of multiple blocks
class Trans_Stage(nn.Module):

    def __init__(self, layer, dim, heads, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.BlockList = nn.ModuleList([Trans_Block(dim, heads, mlp_ratio, drop_rate) for i in range(layer)])

    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        return x


# Generator for CIFAR-10 dataset.
class Generator(nn.Module):

    def __init__(self, layer1=5, layer2=4, layer3=2, patch_size=8, dim=384, heads=4, mlp_ratio=4, drop_rate=0.):
        super(Generator, self).__init__()

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.patch_size = patch_size
        self.dim = dim
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.droprate_rate = drop_rate

        # Initial layer
        self.mlp = nn.Linear(1024, (self.patch_size ** 2) * self.dim)

        # Positional embedding
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, patch_size ** 2, self.dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (patch_size * 2) ** 2, self.dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (patch_size * 4) ** 2, self.dim // 16))

        # Transformer Stages
        self.TStage1 = Trans_Stage(layer=self.layer1, dim=self.dim, heads=self.heads, mlp_ratio=self.mlp_ratio,
                                   drop_rate=self.droprate_rate)
        self.TStage2 = Trans_Stage(layer=self.layer2, dim=self.dim // 4, heads=self.heads, mlp_ratio=self.mlp_ratio,
                                   drop_rate=self.droprate_rate)
        self.TStage3 = Trans_Stage(layer=self.layer3, dim=self.dim // 16, heads=self.heads, mlp_ratio=self.mlp_ratio,
                                   drop_rate=self.droprate_rate)

        # Final layer
        self.unflat = nn.Conv2d(in_channels=self.dim // 16, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, noise):
        # Stage 0, linear layer for input noise
        x = self.mlp(noise).view(-1, self.patch_size ** 2, self.dim)

        # Stage 1, 5x transformer blocks
        x = x + self.pos_embed_1
        x = self.TStage1(x)

        # Stage 2, pixel shuffle, upsampling and 4x transformer blocks
        x = UpSampling(x, self.patch_size, self.patch_size)
        x = x + self.pos_embed_2
        x = self.TStage2(x)

        # Stage 3, pixel shuffle, upsampling and 2x transformer blocks
        x = UpSampling(x, self.patch_size * 2, self.patch_size * 2)
        x = x + self.pos_embed_3
        x = self.TStage3(x)

        # Stage 4,
        x = self.unflat(x.permute(0, 2, 1).view(-1, self.dim // 16, self.patch_size * 4, self.patch_size * 4))

        return x


# Discriminator for CIFAR-10 dataset.
class Discriminator(nn.Module):
    def __init__(self, args, layer1=3, layer2=3, img_size=32, patch_size=2, input_channel=3, num_classes=1, dim=384,
                 heads=4, mlp_ratio=4, drop_rate=0.):
        super().__init__()
        self.args = args
        self.layer1 = layer1
        self.layer2 = layer2
        self.patch_size = patch_size
        self.drop_rate = nn.Dropout(p=drop_rate)

        # Patch embedding
        self.patch_embed_1 = nn.Conv2d(input_channel, dim // 2, kernel_size=patch_size, stride=patch_size)
        self.patch_embed_2 = nn.conv2d(input_channel, dim // 2, kernel_size=patch_size, stride=patch_size)

        # Patch numbers
        num_patches_1 = (img_size // patch_size) ** 2
        num_patches_2 = ((img_size // 2) // patch_size) ** 2

        # Positional embedding
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, dim // 2))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, dim))

        # Transformer Stages
        self.TStage1 = Trans_Stage(layer=self.layer1, dim=dim // 2, heads=heads, mlp_ratio=mlp_ratio,
                                   drop_rate=self.droprate_rate)
        self.TStage2 = Trans_Stage(layer=self.layer2, dim=dim, heads=heads, mlp_ratio=mlp_ratio,
                                   drop_rate=self.droprate_rate)

        self.last_block = Trans_Block(dim, heads=heads, mlp_ratio=mlp_ratio,
                                      drop_rate=self.droprate_rate)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed_1, std=0.2)
        nn.init.trunc_normal_(self.pos_embed_2, std=0.2)
        nn.init.trunc_normal_(self.cls_token, std=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.diff_aug:
            x = DiffAugment(x, self.diff_aug)
        B, _, H, W = x.size()
        H = W = H // self.patch_size

        # Stage 0, linear flatten
        x_1 = self.patch_embed_1(x).flatten(2).permute(0, 2, 1)
        x_2 = self.patch_embed_2(x).flatten(2).permute(0, 2, 1)

        # Stage 1, 3x transformer blocks + AvgPooling + Concatenate
        x = x_1 + self.pos_embed_1
        B, _, C = x.size()
        x = self.TStage1(x)
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        x = torch.cat([x, x_2], dim=-1)

        # Stage 2, 3x transformer blocks
        x = x + self.pos_embed_2
        x = self.TStage2(x)

        # Final stage
        # add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # one final block
        x = self.last_block(x)

        # Get CLS head (real / fake)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x
