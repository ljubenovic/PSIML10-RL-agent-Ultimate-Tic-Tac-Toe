import torch
import os

class DQNetwork(torch.nn.Module):

    def __init__(self, state_len, n_actions, learning_rate, output_range = (0,1)):

        super(DQNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print(self.device)

        self.learning_rate = learning_rate
        self.n_actions = n_actions

        output_layer = torch.nn.Tanh() if output_range == (-1,1) else torch.nn.Sigmoid()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_len, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 81),
            torch.nn.ReLU(),
            torch.nn.Linear(81, n_actions),
            output_layer
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        self.loss = torch.nn.MSELoss(reduction='mean')

        self.transition = list()

        self.to(self.device)


    def forward(self, state):
        return self.network(state)


    def save(self, name, results_folder):
        output_dir = os.path.join(results_folder, "models")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self, os.path.join(output_dir, "DQNetwork" + name + ".pt"))


    def load(self, name, results_folder):
        self = torch.load(os.path.join(results_folder, "models", "DQNetwork" + name + ".pt"), weights_only=False)
        self.eval()