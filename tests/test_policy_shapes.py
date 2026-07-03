import torch

from src.models.gail_discriminator import GAILDiscriminator
from src.models.policy import PolicyNetwork
from src.models.value import ValueNetwork


def test_policy_value_gail_shapes():
    state_dim, action_dim = 6, 2
    policy = PolicyNetwork(state_dim, action_dim, hidden_sizes=[16, 12], lstm_hidden_size=10)
    value = ValueNetwork(state_dim, [16, 12, 8])
    disc = GAILDiscriminator(state_dim, action_dim, embedding_dim=16, lstm_hidden_size=12)
    states = torch.randn(5, state_dim)
    actions = torch.randn(5, action_dim)
    mean, std = policy(states)
    values = value(states)
    probs = disc(states, actions)
    assert mean.shape == (5, action_dim)
    assert std.shape == (5, action_dim)
    assert values.shape == (5,)
    assert probs.shape == (5,)
    assert torch.all(std > 0)

