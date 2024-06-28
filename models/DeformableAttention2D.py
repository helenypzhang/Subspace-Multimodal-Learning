from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops import repeat

from einops.layers.torch import Rearrange

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
    # def __init__(
    #     self,
    #     *,
    #     dim,
    #     dim_head = 64,
    #     heads = 8,
    #     dropout = 0.,
    #     downsample_factor = 4,
    #     offset_scale = None,
    #     offset_groups = 4,
    #     offset_kernel_size = 6,
    #     group_queries = False,
    #     group_key_values = False
    # ):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = 4,
        offset_groups = 8,
        offset_kernel_size = 6,
        group_queries = True,
        group_key_values = True
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
        # # x1(B, 512, 1+2500), x2(B, 512, 1+2500)
        # x1_cls_token, x1_feat_token = x1[:, :, 0], x1[:, :, 1:]
        # # print('x1_cls_token.shape:', x1_cls_token.shape) #[B, 512]
        # x1 = x1_feat_token
        # x2_cls_token, x2_feat_token = x2[:, :, 0], x2[:, :, 1:]
        # x2 = x2_feat_token
        
        x1 = x1.view(x1.shape[0], x1.shape[1], 50, 50)
        x2 = x2.view(x2.shape[0], x2.shape[1], 50, 50) 
        heads, b, h, w, downsample_factor, device = self.heads, x2.shape[0], *x2.shape[-2:], self.downsample_factor, x2.device
        # print('x1.shape:', x1.shape) #torch.Size([1, 128, 50, 50])
        # print('x2.shape:', x2.shape) #torch.Size([1, 128, 50, 50])

        # queries
        q = self.to_q(x1) 
        # print('q.shape:', q.shape) #q.shape: torch.Size([1, 512, 50, 50])

        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group(q) 
        # print('grouped_queries.shape:', grouped_queries.shape) #torch.Size([8, 64, 50, 50])
        offsets = self.to_offsets(grouped_queries) 
        # print('offsets.shape:', offsets.shape) #torch.Size([8, 2, 12, 12])

        # calculate grid + offsets

        grid =create_grid_like(offsets)
        # print('grid.shape:', grid.shape) #torch.Size([2, 12, 12])
        vgrid = grid + offsets
        # print('vgrid.shape:', vgrid.shape) #torch.Size([8, 2, 12, 12])

        vgrid_scaled = normalize_grid(vgrid)
        # print('vgrid_scaled.shape:', vgrid_scaled.shape) #torch.Size([8, 12, 12, 2])

        kv_feats = F.grid_sample(
            group(x2),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
        # print('kv_feats.shape1:', kv_feats.shape) # [8, 16, 12, 12]

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)
        # print('kv_feats.shape2:', kv_feats.shape) #[1, 128, 12, 12]

        # derive key / values

        k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        # print('k, v.shape:', k.shape, v.shape) #[2, 512, 12, 12]

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
        
        # print('out.shape:', out.shape) #[B, 128, 50, 50]

        out = out.view(out.shape[0], out.shape[1], 2500)
        
        if return_vgrid:
            return out, vgrid
        
        # out = torch.cat((x1_cls_token.unsqueeze(2), out), dim=2) #[B,512,1]
        
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