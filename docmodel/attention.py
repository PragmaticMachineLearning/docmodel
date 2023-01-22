from torch import nn
import torch
from typing import Optional, Tuple
from einops import rearrange
from flash_attn.flash_attention import FlashAttention

class FlashSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.inner_attn = FlashAttention(attention_dropout=self.config.attention_probs_dropout_prob)
        
    @torch.jit.ignore  # This jit.ignore call is ignored?
    def flash_inner(self, qkv):
        return self.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=False)
       
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:        
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        qkv = torch.stack((q, k, v), dim=2)
        qkv = rearrange(qkv, "b s three (h d) -> b s three h d", h=self.config.num_attention_heads)
        context, attn_weights = self.flash_inner(qkv)
        context_layer = rearrange(context, "b s h d -> b s (h d)")
        return (context_layer, )

        