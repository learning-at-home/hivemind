# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT modules that do not hog your GPU memory """
import math
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.checkpoint import checkpoint, get_device_states, set_device_states
from transformers import AlbertConfig
from transformers.file_utils import add_start_docstrings
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.albert.modeling_albert import (
    ACT2FN,
    ALBERT_START_DOCSTRING,
    AlbertForPreTraining,
    AlbertLayerGroup,
    AlbertMLMHead,
    AlbertModel,
    AlbertSOPHead,
    AlbertTransformer,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LeanAlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


class LeanAlbertConfig(AlbertConfig):
    rotary_embedding_base: int = 10_000
    hidden_act_gated: bool = True

    def __hash__(self):
        return hash("\t".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")))


class LeanFFN(nn.Module):
    """
    A transformer FFN module that doesn't hog your GPU memory.
    Complete with pre-LayerNorm and residual connections.
    :param gated: use gated activations based on https://arxiv.org/abs/2002.05202 and https://arxiv.org/abs/2102.11972
      note: gated activations require 1.5x more parameters compared to their non-gated variants.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation=F.gelu,
        gated: bool = False,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.dense_i2h = nn.Linear(hidden_size, intermediate_size * 2 if gated else intermediate_size)
        self.dense_h2o = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.activation = activation

    def forward(self, input):
        return _LeanFFN.apply(
            input,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.dense_i2h.weight,
            self.dense_i2h.bias,
            self.dense_h2o.weight,
            self.dense_h2o.bias,
            self.activation,
            self.training,
            self.layer_norm.eps,
        )


class _LeanFFN(torch.autograd.Function):
    @staticmethod
    def _apply_activation(pre_activation: torch.Tensor, activation: callable, hid_size: int):
        if pre_activation.shape[-1] == hid_size:
            return activation(pre_activation)
        elif pre_activation.shape[-1] == 2 * hid_size:
            pre_gate, lin = pre_activation.split(pre_activation.shape[-1] // 2, dim=-1)
            return activation(pre_gate).mul_(lin)
        else:
            raise RuntimeError("The output size of FFN layer must be either 1x or 2x the intermediate_size.")

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        ln_weight,
        ln_bias,
        i2h_weight,
        i2h_bias,
        h2o_weight,
        h2o_bias,
        activation,
        training,
        ln_eps,
    ):
        ctx._activation, ctx._training, ctx._ln_eps = activation, training, ln_eps
        ctx._cpu_rng_state = torch.get_rng_state()
        ctx._device_rng_states = get_device_states(input)

        input_2d = input.view(-1, input.shape[-1])

        input_ln = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ln_eps)

        pre_activation = F.linear(input_ln, i2h_weight, i2h_bias)
        hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, h2o_weight.shape[1])

        out = F.linear(hid_act, h2o_weight, h2o_bias)
        out = out.add_(input_2d)
        ctx.save_for_backward(input, pre_activation, ln_weight, ln_bias, i2h_weight, h2o_weight)
        return out.view(*input.shape)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_ln_weight = grad_ln_bias = None
        grad_i2h_weight = grad_i2h_bias = grad_h2o_weight = grad_h2o_bias = None
        input, pre_activation, ln_weight, ln_bias, i2h_weight, h2o_weight = ctx.saved_tensors
        torch.set_rng_state(ctx._cpu_rng_state)
        set_device_states(*ctx._device_rng_states)

        input_2d = input.view(-1, input.shape[-1])
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        grad_hid_act = torch.mm(grad_output_2d, h2o_weight)

        with torch.enable_grad():
            # rematerialize activation
            pre_activation.requires_grad_(True)
            hid_act = _LeanFFN._apply_activation(pre_activation, ctx._activation, h2o_weight.shape[1])
            (grad_hid,) = torch.autograd.grad(hid_act, pre_activation, grad_hid_act)
            pre_activation.requires_grad_(False)

        grad_input_ln_2d = torch.mm(grad_hid, i2h_weight)

        with torch.enable_grad():
            # rematerialize input_ln
            input_2d.requires_grad_(True)
            input_ln_2d = F.layer_norm(input_2d, input.shape[-1:], ln_weight, ln_bias, ctx._ln_eps)

            if any(ctx.needs_input_grad[0:3]):
                grad_input_2d, grad_ln_weight, grad_ln_bias = torch.autograd.grad(
                    outputs=input_ln_2d, inputs=[input_2d, ln_weight, ln_bias], grad_outputs=grad_input_ln_2d
                )

            input_2d.requires_grad_(False)
            input_ln_2d = input_ln_2d.detach_()

        if ctx.needs_input_grad[0]:
            grad_input_2d = grad_input_2d.add_(grad_output_2d)
            grad_input = grad_input_2d.view(*grad_output.shape)
        if ctx.needs_input_grad[3]:
            grad_i2h_weight = grad_hid.t().mm(input_ln_2d)
        if ctx.needs_input_grad[4]:
            grad_i2h_bias = grad_hid.sum(0)
        if ctx.needs_input_grad[5]:
            grad_h2o_weight = grad_output_2d.t().mm(hid_act)
        if ctx.needs_input_grad[6]:
            grad_h2o_bias = grad_output_2d.sum(0)

        return (
            grad_input,
            grad_ln_weight,
            grad_ln_bias,
            grad_i2h_weight,
            grad_i2h_bias,
            grad_h2o_weight,
            grad_h2o_bias,
            None,
            None,
            None,
            None,
        )


class RotaryEmbeddings(nn.Module):
    """Applies rotary position embeddings to a tensor, uses caching to improve performance"""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, x: torch.Tensor, offset: int = 0):
        """
        :param x: tensor of shape [batch_size, seq_len, nhead, hid_size]
        :param offset: add this value to all position indices
        """
        seq_len = x.shape[1]
        cos, sin = getattr(self, "cos", None), getattr(self, "sin", None)
        if cos is None or seq_len + offset >= cos.shape[0] or x.dtype != cos.dtype or x.device != cos.device:
            cos, sin = get_auxiliary_tensors(seq_len + offset, self.dim, x.dtype, x.device, self.base)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        return rotate(x, cos[None, offset: seq_len + offset, None, :], sin[None, offset: seq_len + offset, None, :])


@torch.no_grad()
@torch.jit.script
def get_auxiliary_tensors(seq_len: int, dim: int, dtype: torch.dtype, device: torch.device, base: int):
    """
    Compute auxiliary sine and cosine tensors for rotary position embedding
    :returns: a tuple of (cos, sin) tensors of shape [seq_len, hid_size]
    """
    _buf = torch.linspace(0, -1 + 2 / dim, dim // 2, dtype=torch.float32, device=device)
    inv_freq = torch.pow(base, _buf, out=_buf).repeat(2)
    time_ix = torch.arange(seq_len, dtype=inv_freq.dtype, device=device)

    freqs = time_ix[:, None] * inv_freq[None, :]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs, out=freqs)
    return cos.to(dtype), sin.to(dtype)


@torch.jit.script
def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """rotate pairwise coordinate using precomputed cos & sin tensors"""
    dim = x.shape[-1]
    x_left, x_right = x.split(split_size=dim // 2, dim=x.ndim - 1)
    x_rotated = torch.cat([x_right.neg(), x_left], dim=x.ndim - 1)
    return x * cos + x_rotated * sin


class LeanSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_positions: int,
        attention_core: Optional[nn.Module] = None,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        """Attention layer that does not hog GPU memory"""
        super().__init__()
        if attention_core is None:
            attention_core = SimpleAttentionCore(hidden_size, num_attention_heads, max_positions, **kwargs)
        else:
            assert len(kwargs) == 0, f"Unexpected parameters: {kwargs}"

        self.hidden_size = hidden_size
        self.attention_core = attention_core
        self.dense_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        hidden_states_ln = self.layer_norm(hidden_states)
        qkv_output = self.dense_qkv(hidden_states_ln)
        query, key, value = qkv_output.split(self.hidden_size, dim=qkv_output.ndim - 1)
        attention_output, attention_probs = checkpoint(self.attention_core, query, key, value, attention_mask)
        projected_context_layer = self.dense_out(attention_output)
        layernormed_context_layer = projected_context_layer + hidden_states
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class SimpleAttentionCore(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, max_positions):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size, self.num_attention_heads = hidden_size, num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask):
        """
        :param query: [batch_size, query_seq_len, hidden_size]
        :param key: [batch_size, kv_seq_len, hidden_size]
        :param value: [batch_size, kv_seq_len, hidden_size]
        :param attention_mask: [batch, query_seq_len, hidden_size]
        :return: (outputs, probs)
          - outputs shape: [batch_size, query_seq_len, hidden_size]
          - probs shape: [batch_size, num_heads, query_seq_len, kv_seq_len]
        """
        query, key, value = map(self.transpose_for_scores, (query, key, value))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attention_scores = torch.where(causal_mask, attention_scores, self.masked_bias.to(attention_scores.dtype))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(2, 1).flatten(2)
        return attention_output, attention_probs


class RotaryAttentionCore(SimpleAttentionCore):
    """Attention core that applies rotary embeddings to queries and keys before computing dot products"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_positions: int,
        rotary_emb: Optional[RotaryEmbeddings] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, num_attention_heads, max_positions, **kwargs)
        if rotary_emb is None:
            rotary_emb = RotaryEmbeddings(self.attention_head_size)
        self.rotary_emb = rotary_emb

    def rotate(self, tensor: torch.Tensor):
        """:param tensor: query or key, shape: [batch_size, query_seq_len, hidden_size]"""
        tensor_split_heads = tensor.view(*(tensor.shape[:-1] + (self.num_attention_heads, self.attention_head_size)))
        return self.rotary_emb(tensor_split_heads).view(*tensor.shape)

    def forward(self, query, key, value, attention_mask):
        return super().forward(self.rotate(query), self.rotate(key), value, attention_mask)


def get_input_embedding(config: LeanAlbertConfig):
    if config.position_embedding_type == "absolute":
        return nn.Embedding(config.max_position_embeddings, config.embedding_size)
    elif config.position_embedding_type == "rotary":
        return None
    else:
        raise NotImplementedError(f"Unsupported embedding type: {config.position_embedding}")


@lru_cache()
def get_attention_core(config: LeanAlbertConfig):
    if config.position_embedding_type == "absolute":
        return None
    elif config.position_embedding_type == "rotary":
        rotary_emb = RotaryEmbeddings(config.hidden_size // config.num_attention_heads, config.rotary_embedding_base)
        return RotaryAttentionCore(
            config.hidden_size, config.num_attention_heads, config.max_position_embeddings, rotary_emb
        )
    else:
        raise NotImplementedError(f"Unsupported embedding type: {config.position_embedding_type}")


class LeanAlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.position_embeddings = get_input_embedding(config)

        self.layernorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

        if self.position_embeddings is not None:
            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.layernorm(embeddings)
        return embeddings


class LeanAlbertLayer(nn.Module):
    def __init__(self, config: LeanAlbertConfig):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.attention = LeanSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.max_position_embeddings,
            attention_core=get_attention_core(config),
            layer_norm_eps=config.layer_norm_eps,
        )

        self.ffn = LeanFFN(
            config.hidden_size,
            config.intermediate_size,
            activation=ACT2FN[config.hidden_act],
            gated=config.hidden_act_gated,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attention_output, *extras = self.attention(hidden_states, attention_mask, output_attentions)
        ffn_output = self.ffn(attention_output)
        return (ffn_output, attention_output, *extras)


class LeanAlbertLayerGroup(AlbertLayerGroup):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.albert_layers = nn.ModuleList([LeanAlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        if head_mask is not None and any(head_mask):
            raise NotImplementedError(f"head mask was provided, but it is not supported")

        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class LeanAlbertTransformer(AlbertTransformer):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList(
            [LeanAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # TODO this should entire be replaced with inheritance and post_layer_norm
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                None,
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.post_layer_norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    "The bare LeanALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class LeanAlbertModel(AlbertModel):
    config_class = LeanAlbertConfig

    def __init__(self, config: AlbertConfig, add_pooling_layer=True):
        PreTrainedModel.__init__(self, config)

        self.config = config
        self.embeddings = LeanAlbertEmbeddings(config)
        self.encoder = LeanAlbertTransformer(config)

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.init_weights()


class LeanAlbertForPreTraining(AlbertForPreTraining, PreTrainedModel):
    config_class = LeanAlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig):
        PreTrainedModel.__init__(self, config)

        self.albert = LeanAlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.init_weights()


from hivemind.moe.server.layers.custom_experts import register_expert_class

SEQUENCE_LENGTH = 2048

head_sample_input = lambda batch_size, hid_dim: (
    torch.randint(low=0, high=1000, size=(batch_size, SEQUENCE_LENGTH), dtype=torch.long),
)


@register_expert_class("lm_head", head_sample_input)
class HeadExpert(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        config = LeanAlbertConfig.from_pretrained("albert-xxlarge-v2")
        config.hidden_size = hid_dim
        config.intermediate_size = 4 * config.hidden_size
        config.num_hidden_layers = 12
        config.vocab_size = 50304
        config.max_position_embeddings = SEQUENCE_LENGTH

        self.encoder = LeanAlbertTransformer(config)
        self.embeddings = LeanAlbertEmbeddings(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        (encoder_outputs,) = self.encoder(embedding_output, return_dict=False)

        return encoder_outputs


body_sample_input = lambda batch_size, hid_dim: (torch.empty((batch_size, SEQUENCE_LENGTH, hid_dim)),)


@register_expert_class("lm_body", body_sample_input)
class BodyExpert(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        config = LeanAlbertConfig.from_pretrained("albert-xxlarge-v2")
        config.hidden_size = hid_dim
        config.intermediate_size = 4 * config.hidden_size
        config.num_hidden_layers = 12
        config.vocab_size = 50304
        config.max_position_embeddings = SEQUENCE_LENGTH

        self.config = config
        self.albert_layer_groups = nn.ModuleList(
            [LeanAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states):
        for i in range(self.config.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                None,
                None,
                False,
                False,
            )
            hidden_states = layer_group_output[0]

        hidden_states = self.post_layer_norm(hidden_states)

        return hidden_states


tail_sample_input = lambda batch_size, hid_dim: (
    torch.empty((batch_size, SEQUENCE_LENGTH, hid_dim)),
    torch.randint(0, 1000, (batch_size, SEQUENCE_LENGTH), dtype=torch.long),
)


@register_expert_class("lm_tail", tail_sample_input)
class TailExpert(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        config = LeanAlbertConfig.from_pretrained("albert-xxlarge-v2")
        config.hidden_size = hid_dim
        config.intermediate_size = 4 * config.hidden_size
        config.num_hidden_layers = 12
        config.vocab_size = 50304
        config.max_position_embeddings = SEQUENCE_LENGTH

        self.config = config
        self.albert_layer_groups = nn.ModuleList(
            [LeanAlbertLayerGroup(config) for _ in range(config.num_hidden_groups)]
        )
        self.post_layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_eps)

        self.lm_head = AlbertMLMHead(config)

    def forward(self, hidden_states, labels):
        for i in range(self.config.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                None,
                None,
                False,
                False,
            )
            hidden_states = layer_group_output[0]

        hidden_states = self.post_layer_norm(hidden_states)

        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none")
        return loss
