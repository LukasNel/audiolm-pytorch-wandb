from tqdm import tqdm
from mamba_ssm import Mamba
import torch
import torch.nn as nn
from torch.nn import  functional as F
dropout = 0.3
class FeedForward(nn.Module):
  def __init__(self, n_embed) -> None:
    super().__init__()
    self.ffn = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed, n_embed),
      nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.ffn(x)

class MambaBlock(nn.Module):
  def __init__(self, dim) -> None:
    super().__init__()
    self.sa_head = Mamba(
      # This module uses roughly 3 * expand * d_model^2 parameters
      d_model=dim, # Model dimension d_model
      d_state=16,  # SSM state expansion factor
      d_conv=4,    # Local convolution width
      expand=2,    # Block expansion factor
  ).to("cuda")
    self.ffn = FeedForward(dim)
    self.ln1 = nn.LayerNorm(dim)
    self.ln2 = nn.LayerNorm(dim)


  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ffn(self.ln2(x))

    return x
  
class MambaTransformer(nn.Module):
    def __init__(self,dim, depth):
        super().__init__()
        self.blocks = nn.Sequential(*[MambaBlock(dim) for _ in range(depth)])

    def forward(self, x, context = None, self_attn_mask = None, context_mask = None, kv_cache = None, return_kv_cache = True):
        x = self.blocks(x)
        if return_kv_cache:
            return x, kv_cache
        return x
