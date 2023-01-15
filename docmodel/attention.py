from torch import nn
import torch
from typing import Optional, Tuple
from einops import rearrange
from flash_attn.flash_attention import FlashMHA

class FlashAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.flash_mha = FlashMHA(
            config.hidden_size,
            config.num_attention_heads,
            bias=config.qkv_bias,
            batch_first=True,
            attention_dropout=config.dropout_prob,
            causal=config.causal_attention,
        )

        self.flash_mha.out_proj = None
        self.output_dim = config.hidden_size
        
    @torch.jit.ignore  # This jit.ignore call is ignored?
    def flash_inner(self, qkv):
        return self.flash_mha.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=self.flash_mha.causal)
       
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor]:
        
        # def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)

        Returns only the rearranged, unprojected output
        """
        qkv = self.flash_mha.Wqkv(hidden_states)
        if self.rotary_emb is not None:
            query, key, value = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads).unbind(dim=2)
            query, key = self.rotary_emb(query, key)
            qkv = torch.stack([query.type(qkv.dtype), key.type(qkv.dtype), value.type(qkv.dtype)], dim=2)
        else:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads)
        context, attn_weights = self.flash_inner(qkv)
        context_layer = rearrange(context, "b s h d -> b s (h d)")
        return (context_layer, )

        