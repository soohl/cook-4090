"""W8A8 transformer linears using Comfy Kitchen's ConvRot kernels."""

import torch
from torch import nn


class ConvRotLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        from comfy_kitchen.tensor import TensorWiseINT8Layout

        with torch.no_grad():
            weight, params = TensorWiseINT8Layout.quantize(
                linear.weight.detach().to("cuda"), per_channel=True, convrot=True)
        self.register_buffer("weight", weight.cpu())
        self.register_buffer("weight_scale", params.scale.cpu())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().cpu())
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, inputs):
        return torch.ops.comfy_kitchen.int8_linear(
            inputs.contiguous(), self.weight, self.weight_scale, self.bias, 2, True, 256)


def quantize_transformer(transformer):
    # Keep input/output projections, conditioning, norms, VAE and text encoder
    # in BF16. Quantize the large attention and MLP linears in repeated blocks.
    count = elements = 0
    for block in transformer.transformer_blocks:
        for name, module in list(block.named_modules()):
            if isinstance(module, nn.Linear) and module.in_features % 256 == 0:
                parent, _, attribute = name.rpartition(".")
                block.get_submodule(parent).set_submodule(attribute, ConvRotLinear(module))
                count += 1
                elements += module.weight.numel()
    torch.cuda.empty_cache()
    if not count:
        raise ValueError("No compatible transformer linears found for INT8 ConvRot")
    return {"quantization": "int8-convrot", "quantized_linears": count,
            "quantized_weight_elements": elements, "activation_quantization": "dynamic_per_row_int8",
            "rotation_group_size": 256}
