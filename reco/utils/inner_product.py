import torch
import torch.nn as nn


class InnerProduct(nn.Module):
    def __init__(self, featruemap):
        super(InnerProduct, self).__init__()
        feature_dict = featruemap["features"]
        self._num_fields = len(feature_dict)
        self._num_interactions = int(self._num_fields * (self._num_fields - 1) / 2)

    def forward(self, X_emb):
        sum_of_square = X_emb.sum(dim=1) ** 2
        square_of_sum = torch.sum(X_emb ** 2, dim=1)
        output = (sum_of_square - square_of_sum) / 2
        output = output.sum(dim=1, keepdim=True)
        return output


if __name__ == "__main__":
    num_samples = 1000
    num_features = 10
    X_emb = torch.randn(num_samples, num_features).unsqueeze(0)

    featruemap = {
        "features": {f"feature_{i}": i for i in range(num_features)}
    }

    inner_product = InnerProduct(featruemap)
    output = inner_product(X_emb)

    print("Input Shape:", X_emb.shape)
    print("Output Shape:", output.shape)
    print("Output Values:", output)
