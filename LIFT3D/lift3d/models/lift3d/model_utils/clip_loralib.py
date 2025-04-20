#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_param(curr_mod, name, param=None, mode="update"):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if "." in name:
        n = name.split(".")
        module_name = n[0]
        rest = ".".join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == "update":
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == "get":
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        fan_in_fan_out: bool = False,
        dropout_rate: float = 0,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:

            self.scaling = self.lora_alpha / math.sqrt(self.r)  
        self.merged = False
        self.fan_in_fan_out = fan_in_fan_out
        self.params_with_lora = {}

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f"self.{param_name}").size()) == 2
            self.register_parameter(
                f"{lora_name}_lora_A",
                nn.Parameter(
                    eval(f"self.{param_name}").new_zeros(
                        (self.r, eval(f"self.{param_name}").size()[1])
                    )
                ),
            )
            self.register_parameter(
                f"{lora_name}_lora_B",
                nn.Parameter(
                    eval(f"self.{param_name}").new_zeros(
                        (eval(f"self.{param_name}").size()[0], self.r)
                    )
                ),
            )

            eval(f"self.{param_name}").requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f"{lora_name}_lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(
                    eval(f"self.{lora_name}_lora_A"), a=math.sqrt(5)
                )
                nn.init.zeros_(eval(f"self.{lora_name}_lora_B"))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose(
            (eval(f"self.{lora_name}_lora_B") @ eval(f"self.{lora_name}_lora_A")).view(
                eval(f"self.{param_name}").shape
            )
        )

    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode="get")
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode="update")

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f"self.{param_name}").data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f"self.{param_name}").data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


class Embedding(nn.Embedding, LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.register_lora_param()
        nn.Embedding.reset_parameters(self)
        self.init_lora_param()

    def init_lora_param(self):
        if hasattr(self, "w_lora_A"):
            nn.init.zeros_(self.w_lora_A)
            nn.init.normal_(self.w_lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Embedding.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Embedding.forward(self, x, **kwargs)


class LinearLoRA(nn.Linear, LoRALayer):
    def __init__(
        self,
        existing_linear: nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        fan_in_fan_out: bool = False,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features,
        )

        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(
            self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out
        )

        # Actual trainable parameters
        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.dropout is None:  
            if self.r > 0 and not self.merged:
                self.merge_lora_param()
                result = nn.Linear.forward(self, x, **kwargs)
                self.sub_lora_data()
                return result
            else:
                return nn.Linear.forward(self, x, **kwargs)

        original_output = nn.Linear.forward(self, x)

        if self.training and self.dropout.p > 0:
            x = self.dropout(x)

        if self.r > 0 and not self.merged:
            lora_adjustment = (
                torch.matmul(x, self.merge_BA("weight").transpose(0, 1)) * self.scaling
            )
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result


class Conv1d(nn.Conv1d, LoRALayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_channels // self.groups * kernel_size, r * kernel_size)
                )
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv1d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv1d.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv1d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv1d.forward(self, x, **kwargs)


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_channels // self.groups * kernel_size, r * kernel_size)
                )
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)


class Conv3d(nn.Conv3d, LoRALayer):
    # LoRA implemented in a Conv3d layer
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs,
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {"weight": "w"}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros(
                    (out_channels // self.groups * kernel_size, r * kernel_size)
                )
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv3d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv3d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv3d.forward(self, x, **kwargs)


class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(
        self,
        existing_mha: nn.MultiheadAttention,
        enable_lora: list = ["q", "k", "v", "o"],
        r: int = 0,
        lora_alpha: int = 1,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.dropout = 0  # this module is not used to retrain the main block
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        # self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=existing_mha.in_proj_bias is not None)
        self.q_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None
        )
        self.proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None
        )

        # Initialize parameters
        with torch.no_grad():

            # Extract the existing weights and biases
            existing_weight = existing_mha.in_proj_weight.data
            existing_bias = (
                existing_mha.in_proj_bias.data
                if existing_mha.in_proj_bias is not None
                else None
            )

            # Initialize q_proj
            self.q_proj.weight.data.copy_(existing_weight[: self.embed_dim, :])
            if existing_bias is not None:
                self.q_proj.bias.data.copy_(existing_bias[: self.embed_dim])

            # Initialize k_proj
            self.k_proj.weight.data.copy_(
                existing_weight[self.embed_dim : 2 * self.embed_dim, :]
            )
            if existing_bias is not None:
                self.k_proj.bias.data.copy_(
                    existing_bias[self.embed_dim : 2 * self.embed_dim]
                )

            # Initialize v_proj
            self.v_proj.weight.data.copy_(existing_weight[2 * self.embed_dim :, :])
            if existing_bias is not None:
                self.v_proj.bias.data.copy_(existing_bias[2 * self.embed_dim :])

            # Initialize proj
            self.proj.weight.data.copy_(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data.copy_(existing_mha.out_proj.bias.data)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        # Init qkv as a new lora linear layer
        for item in enable_lora:
            if item == "q":
                self.q_proj = LinearLoRA(
                    self.q_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )
            elif item == "k":
                self.k_proj = LinearLoRA(
                    self.k_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )
            elif item == "v":
                self.v_proj = LinearLoRA(
                    self.v_proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )
            elif item == "o":
                self.proj = LinearLoRA(
                    self.proj,
                    r=r,
                    lora_alpha=lora_alpha,
                    fan_in_fan_out=False,
                    dropout_rate=dropout_rate,
                )

    def forward_module(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        """
        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        """

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.0

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def train(self, mode: bool = True):
        super().train(mode)
        # self.lora_train(mode)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs
    ):

        return self.forward_module(query, key, value, **kwargs)


INDEX_POSITIONS_VISION = {
    "ViT-B/32": {
        "bottom": [0, 1, 2, 3],
        "mid": [4, 5, 6, 7],
        "up": [8, 9, 10, 11],
        "half-up": [6, 7, 8, 9, 10, 11],
        "half-bottom": [0, 1, 2, 3, 4, 5],
        "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
}


def merge_lora_to_mha(lora_layer, original_mha):
    """
    Merges LoRA-modified weights from lora_layer into the original nn.MultiheadAttention layer.

    Args:
        lora_layer (LoRALayer): The LoRA-modified MultiheadAttention layer.
        original_mha (nn.MultiheadAttention): The original MultiheadAttention layer to merge weights into.

    """
    with torch.no_grad():
        # Merge query, key, and value projections
        original_mha.in_proj_weight.data[: lora_layer.embed_dim, :] = (
            lora_layer.q_proj.weight.data
        )
        if original_mha.in_proj_bias is not None:
            original_mha.in_proj_bias.data[: lora_layer.embed_dim] = (
                lora_layer.q_proj.bias.data
            )

        original_mha.in_proj_weight.data[
            lora_layer.embed_dim : 2 * lora_layer.embed_dim, :
        ] = lora_layer.k_proj.weight.data
        if original_mha.in_proj_bias is not None:
            original_mha.in_proj_bias.data[
                lora_layer.embed_dim : 2 * lora_layer.embed_dim
            ] = lora_layer.k_proj.bias.data

        original_mha.in_proj_weight.data[2 * lora_layer.embed_dim :, :] = (
            lora_layer.v_proj.weight.data
        )
        if original_mha.in_proj_bias is not None:
            original_mha.in_proj_bias.data[2 * lora_layer.embed_dim :] = (
                lora_layer.v_proj.bias.data
            )

        # Merge output projection
        original_mha.out_proj.weight.data = lora_layer.proj.weight.data
        if original_mha.out_proj.bias is not None:
            original_mha.out_proj.bias.data = lora_layer.proj.bias.data


def apply_lora(model):
    """
    Applies LoRA to the MultiheadAttention layers of the model.

    Args:
        model (nn.Module): The model to which LoRA modules will be applied.

    Returns:
        list: A list of LoRA layers that have been added to the model.
    """
    list_lora_layers = []
    backbone = "ViT-B/32"
    position = "all"
    params = ["q", "k", "v"]

    indices = INDEX_POSITIONS_VISION[backbone][position]
    vision_encoder = model
    for i, block in enumerate(vision_encoder.resblocks):
        if i in indices:
            for name, submodule in block.named_children():
                if isinstance(submodule, nn.MultiheadAttention):
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule,
                        enable_lora=params,
                        r=2,
                        lora_alpha=1,
                        dropout_rate=0.25,
                    )
                    setattr(block, name, new_multi_head_lora)
                    list_lora_layers.append(new_multi_head_lora)
    return list_lora_layers


def replace_module(model, name, new_module):
    components = name.split(".")
    submodule = model
    for comp in components[:-1]:
        submodule = getattr(submodule, comp)
    setattr(submodule, components[-1], new_module)


def merge_lora(model):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, PlainMultiheadAttentionLoRA):
            modules_to_replace.append((name, module))
            for name, submodule in module.named_modules():
                if isinstance(submodule, LinearLoRA):
                    submodule.train(mode=False)
    for name, module in modules_to_replace:
        original_mha = nn.MultiheadAttention(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            kdim=module.kdim,
            vdim=module.vdim,
            batch_first=module.batch_first,
        )
        merge_lora_to_mha(module, original_mha)
        replace_module(model, name, original_mha)
