from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops import repeat

from einops.layers.torch import Rearrange
########
# 1. for NystromAttention module
# ref: https://github.com/lucidrains/nystrom-attention/blob/main/nystrom_attention/nystrom_attention.py
#######

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
    
########
# 2. testing:
#######
class DeformableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, deformable_groups=4):
        super(DeformableMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.deformable_groups = deformable_groups

        # Assuming that embed_dim is divisible by num_heads
        self.head_dim = embed_dim // num_heads

        # Define the projection matrices
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Deformable groups could control how the offsets are applied.
        # You would need to implement the mechanism to calculate the offsets.
        # This is a placeholder for such a mechanism.
        self.offsets_calculator = nn.Linear(embed_dim, deformable_groups * 2)

    def forward(self, q, k, v):
        B, L, E = q.size()
        
        # Calculate queries, keys, values
        queries = self.query_proj(q).view(B, L, self.num_heads, self.head_dim)
        keys = self.key_proj(k).view(B, L, self.num_heads, self.head_dim)
        values = self.value_proj(v).view(B, L, self.num_heads, self.head_dim)
        
        # Calculate offsets using a learned function (to be implemented)
        offsets = self.offsets_calculator(q)  # This is a placeholder
        
        # Apply deformable attention mechanism here
        # The actual implementation details would depend on the specific
        # formulation of deformable attention you're following.
        
        # Placeholder for output after applying deformable attention
        output = torch.zeros_like(q)
        
        return output

# # Example usage:
# B, L, E = 32, 2500, 512  # Batch size B, sequence length L, embedding dimension E
# num_heads = 8  # Number of attention heads
# deformable_groups = 4  # Number of deformable groups

# fusion_feature = torch.rand(B, L, E)  # Replace with your actual fusion_feature tensor
# path_feature = torch.rand(B, L, E)    # Replace with your actual path_feature tensor

# # Create the deformable multi-head attention layer
# deformable_attention_layer = DeformableMultiheadAttention(embed_dim=E, num_heads=num_heads, deformable_groups=deformable_groups)

# # Forward pass through the deformable multi-head attention layer
# attn_output = deformable_attention_layer(fusion_feature, path_feature, path_feature)

# print(attn_output.shape)  # Should print torch.Size([B, L, E])

########
# 3. for DeformCrossAttention2D module
# ref: https://github.com/lucidrains/deformable-attention/blob/main/deformable_attention/deformable_attention_2d.py
#######

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

class DeformCrossAttention2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = 4,
        offset_kernel_size = 6,
        group_queries = False,
        group_key_values = False
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x1, x2, return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """
        # x1(B, 512, 1+2500), x2(B, 512, 1+2500)
        x1_cls_token, x1_feat_token = x1[:, :, 0], x1[:, :, 1:]
        # print('x1_cls_token.shape:', x1_cls_token.shape) #[B, 512]
        x1 = x1_feat_token
        x2_cls_token, x2_feat_token = x2[:, :, 0], x2[:, :, 1:]
        x2 = x2_feat_token
        
        x1 = x1.view(x1.shape[0], x1.shape[1], 50, 50)
        x2 = x2.view(x2.shape[0], x2.shape[1], 50, 50)
        
        x1 = x1.view(x1.shape[0], x1.shape[1], 50, 50) #(B, 512, 1+2500)
        x2 = x2.view(x2.shape[0], x2.shape[1], 50, 50) #(B, 512, 1+2500)
        heads, b, h, w, downsample_factor, device = self.heads, x2.shape[0], *x2.shape[-2:], self.downsample_factor, x2.device

        # queries
        q = self.to_q(x1) #(1, 1+512, 64, 64)

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        vgrid = grid + offsets

        vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group(x2),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries

        q = q * self.scale

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias

        grid = create_grid_like(x2)
        grid_scaled = normalize_grid(grid, dim = 0)
        rel_pos_bias = self.rel_pos_bias(grid_scaled, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        out = out.view(out.shape[0], out.shape[1], 2500)
        
        out = torch.cat((x1_cls_token.unsqueeze(2), out), dim=2) #[B,512,1]
        
        return out
    
# if __name__ == "__main__":
    
#     attn = DeformCrossAttention2D(
#         dim = 512,                   # feature dimensions
#         dim_head = 64,               # dimension per head
#         heads = 8,                   # attention heads
#         dropout = 0.,                # dropout
#         downsample_factor = 4,       # downsample factor (r in paper)
#         offset_scale = 4,            # scale of offset, maximum offset
#         offset_groups = None,        # number of offset groups, should be multiple of heads
#         offset_kernel_size = 6,      # offset kernel size
#     )

#     x = torch.randn(1, 512, 64, 64).   
#     attn(x) # (1, 512, 64, 64)

#     x = (1, 512, H, W)

########
# 4. for DeformCrossAttention1D module
# ref: https://github.com/lucidrains/deformable-attention/blob/main/deformable_attention/deformable_attention_2d.py
#######