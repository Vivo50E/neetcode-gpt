import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        result = torch.reshape(to_reshape, (-1, 2))
        return torch.round(result, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean
        result = torch.mean(to_avg, dim=0)
        return torch.round(result, decimals=4)

    def concatenate(
        self,
        cat_one: TensorType[float],
        cat_two: TensorType[float],
    ) -> TensorType[float]:
        # Join two tensors side-by-side
        result = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(result, decimals=4)

    def get_loss(
        self,
        prediction: TensorType[float],
        target: TensorType[float],
    ) -> TensorType[float]:
        # Compute mean squared error
        result = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(result, decimals=4)