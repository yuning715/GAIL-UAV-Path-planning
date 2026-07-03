import torch

from src.models.gan_lstm import TrajectoryGenerator
from src.models.trajectory_discriminator import TrajectoryDiscriminator


def test_gan_shapes():
    batch, steps, features = 4, 16, 3
    generator = TrajectoryGenerator(features, hidden_size=16, num_layers=2, dropout=0.0)
    discriminator = TrajectoryDiscriminator(features, cnn_filters=[8, 12, 16], lstm_hidden_size=10, dropout=0.0)
    incomplete = torch.randn(batch, steps, features)
    mask = torch.ones(batch, steps, features)
    repaired = generator(incomplete, mask)
    prob = discriminator(repaired)
    assert repaired.shape == (batch, steps, features)
    assert prob.shape == (batch,)
    assert torch.all((prob >= 0) & (prob <= 1))

