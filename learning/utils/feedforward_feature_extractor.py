import torch
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeedforwardFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, extra_state_features=0, dims=3, complex=False):
        self.n_state_features = 10 + extra_state_features
        self.n_ff_features = observation_space.shape[0] - self.n_state_features
        self.dims = dims
        self.complex = complex

        assert self.n_ff_features % self.dims == 0

        ff_feature_dim = (self.n_ff_features // self.dims - 4) * 8
        if complex:
            ff_feature_dim = (self.n_ff_features // self.dims - 6) * 8

        features_dim = self.n_state_features + ff_feature_dim

        super().__init__(observation_space, features_dim)

        if self.complex:
            self.layer1 = torch.nn.Conv1d(in_channels=self.dims, out_channels=32, kernel_size=3, stride=1)
            self.act1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
            self.layer3 = torch.nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3)
        else:
            self.layer1 = torch.nn.Conv1d(in_channels=self.dims, out_channels=8, kernel_size=3, stride=1)
            self.act1 = torch.nn.ReLU()
            self.layer2 = torch.nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3)
        
    def forward(self, full_obs):
        batch_dim = full_obs.shape[0]
        obs_dim = full_obs.shape[1]

        ff_features = full_obs[:, self.n_state_features:].reshape(batch_dim, (obs_dim - self.n_state_features) // self.dims, self.dims)
        # channel first
        ff_features = torch.permute(ff_features, (0, 2, 1))

        obs_features = full_obs[:, :self.n_state_features]

        if self.complex:
            x = self.act1(self.layer1(ff_features))
            x = self.act1(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.act1(self.layer1(ff_features))
            x = self.layer2(x)
        
        x = torch.flatten(x, start_dim=1)

        output = torch.cat([obs_features, x], axis=1)

        return output
