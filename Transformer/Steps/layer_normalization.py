import torch
from torch import nn

# an example of normalization
inputs = torch.Tensor([[[0.2, 0.1, 0.3]], [[0.5, 0.1, 0.1]]])
parameter_shape = inputs.size()[-2:]
gamma = nn.Parameter(torch.ones(parameter_shape))  # learning rate
beta = nn.Parameter(torch.zeros(parameter_shape))  # bias
dims = [-(i + 1) for i in range(len(parameter_shape))]
mean = inputs.mean(dim=dims, keepdim=True)
var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
epsilon = 1e-5
std = (var + epsilon).sqrt()  # standard deviation
y = (inputs - mean) / std
out = gamma * y + beta
print(out)


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


batch_size = 3
sentence_length = 5
embedding_dim = 8
inputs_1 = torch.randn((sentence_length, batch_size, embedding_dim))
inputs_2 = inputs_1
inputs = inputs_1 + inputs_2
layer_norm = LayerNormalization(inputs.size()[-2:])
out_1 = layer_norm(inputs_1)
out_2 = layer_norm(inputs)
print(out_2, "\n", inputs_1)
