import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Any, TypeVar
from abc import abstractmethod

Tensor = TypeVar('torch.Tensor')


# implement the base VAE structure of abstract base class
class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class BetaVAE(BaseVAE):

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 embedding_dim: int,
                 context_length: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 bidirectional: bool = True,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.bidirectional = bidirectional
        self.num_direction_heads = bidirectional + 1

        if hidden_dims is None:
            hidden_dims = [1792, 896, 448, 224, 112, 56]
        self.hidden_dims = hidden_dims

        # Build Encoder
        self.encoder_lstm_layers = nn.ModuleList()
        self.encoder_bn_layers = nn.ModuleList()
        self.encoder_relu_layers = nn.ModuleList()
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            lstm_hidden_size = h_dim // self.num_direction_heads
            self.encoder_lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=lstm_hidden_size,
                    num_layers=2,
                    batch_first=True,
                    bidirectional=self.bidirectional
                )
            )
            self.encoder_bn_layers.append(
                nn.BatchNorm1d(h_dim)
            )
            self.encoder_relu_layers.append(
                nn.LeakyReLU()
            )
            in_dim = h_dim

        self.fc_mu = nn.Linear(
            hidden_dims[-1] * context_length, latent_dim)
        self.fc_var = nn.Linear(
            hidden_dims[-1] * context_length, latent_dim)

        # Build Decoder
        in_dim = hidden_dims[-1]
        self.decoder_input = nn.Linear(
            latent_dim, hidden_dims[-1] * context_length)
        hidden_dims.reverse()

        self.decoder_lstm_layers = nn.ModuleList()
        self.decoder_bn_layers = nn.ModuleList()
        self.decoder_relu_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i + 1 < len(hidden_dims):
                lstm_hidden_size = hidden_dims[i +
                                               1] // self.num_direction_heads
            else:
                lstm_hidden_size = self.embedding_dim // self.num_direction_heads
            self.decoder_lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=lstm_hidden_size,
                    batch_first=True,
                    bidirectional=self.bidirectional
                ))
            self.decoder_bn_layers.append(
                nn.BatchNorm1d(lstm_hidden_size * self.num_direction_heads)
            )
            self.decoder_relu_layers.append(
                nn.LeakyReLU()
            )
            in_dim = lstm_hidden_size * self.num_direction_heads

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x L x D]
        :return: (Tensor) List of latent codes
        """
        for lstm, bn, relu in zip(self.encoder_lstm_layers, self.encoder_bn_layers, self.encoder_relu_layers):
            input, _ = lstm(input)
            input = input.permute(0, 2, 1)
            input = bn(input)
            input = relu(input)
            input = input.permute(0, 2, 1)

        result = torch.flatten(input, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(
            z.shape[0], self.context_length, self.hidden_dims[0])
        for lstm, bn, relu in zip(self.decoder_lstm_layers, self.decoder_bn_layers, self.decoder_relu_layers):
            result, _ = lstm(result)
            result = result.permute(0, 2, 1)
            result = bn(result)
            result = relu(result)
            result = result.permute(0, 2, 1)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                                               log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter *
                            self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input tensor, returns the reconstructed tensor
        :param x: (Tensor) [B x L x D]
        :return: (Tensor) [B x L x D]
        """

        return self.forward(x)[0]


model = BetaVAE(embedding_dim=3584, context_length=128, latent_dim=10)
x = torch.randn(16, 128, 3584)
out = model(x)
print(out[1].shape)
# loss_dict = model.loss_function(*out, M_N=16 / 3200)
# print(loss_dict["loss"])
# print(loss_dict["Reconstruction_Loss"])
# print(loss_dict["KLD"])
