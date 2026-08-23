import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def xavier_init(
        self, fan_in: int, fan_out: int
    ) -> List[List[float]]:
        torch.manual_seed(0)

        weights = torch.empty(fan_out, fan_in)
        nn.init.xavier_normal_(weights)

        return [
            [round(value, 4) for value in row]
            for row in weights.tolist()
        ]

    def kaiming_init(
        self, fan_in: int, fan_out: int
    ) -> List[List[float]]:
        torch.manual_seed(0)

        weights = torch.empty(fan_out, fan_in)
        nn.init.kaiming_normal_(
            weights,
            mode="fan_in",
            nonlinearity="relu",
        )

        return [
            [round(value, 4) for value in row]
            for row in weights.tolist()
        ]

    def check_activations(
    self,
    num_layers: int,
    input_dim: int,
    hidden_dim: int,
    init_type: str,
) -> List[float]:
        torch.manual_seed(0)

        weights_list = []

        # 先初始化所有层的权重
        for layer_index in range(num_layers):
            current_input_dim = (
                input_dim if layer_index == 0 else hidden_dim
            )

            weights = torch.empty(hidden_dim, current_input_dim)

            if init_type == "random":
                nn.init.normal_(weights, mean=0.0, std=1.0)
            elif init_type.lower() == "xavier":
                nn.init.xavier_normal_(weights)
            elif init_type.lower() in {"kaiming", "he"}:
                nn.init.kaiming_normal_(
                    weights,
                    mode="fan_in",
                    nonlinearity="relu",
                )
            else:
                raise ValueError("Unsupported initialization type")

            weights_list.append(weights)

        # 所有权重初始化完成后，再生成输入
        activations = torch.randn(input_dim)
        activation_stds = []

        for weights in weights_list:
            activations = torch.relu(weights @ activations)
            activation_stds.append(
                round(activations.std().item(), 2)
            )

        return activation_stds