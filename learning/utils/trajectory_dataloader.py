import torch

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, data, device, e_dim=3, time_horizon=100):
        # Data should be shape (# episodes, length per episode, obs dim + u dim + e dim)

        self.X = torch.from_numpy(data[:, :, :-e_dim]).type(torch.FloatTensor).to(device)
        self.y = torch.from_numpy(data[:, :, -e_dim:]).type(torch.FloatTensor).to(device)
        self.X = torch.permute(self.X, (0, 2, 1))
        self.T = time_horizon
        self.num_episodes = data.shape[0]
        self.ep_len = data.shape[1]

    def __len__(self):
        return self.num_episodes * (self.ep_len - self.T + 1)

    def __getitem__(self, index):
        i0 = index // (self.ep_len - self.T + 1)
        i1 = index % (self.ep_len - self.T + 1)
        return self.X[i0, :, i1:i1+self.T], self.y[i0, 0, :]