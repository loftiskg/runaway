import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, state_size, num_actions):
    super(Model, self).__init__()
    self.state_size = state_size
    self.num_actions = num_actions
    self.layer1 = nn.Linear(state_size, 64)
    self.layer2 = nn.Linear(64,64)
    # self.layer3 = nn.Linear(64,64)
    # self.layer4 = nn.Linear(64,64)
    self.output = nn.Linear(64, self.num_actions)

  def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = torch.relu(self.layer2(x))
    # x = torch.relu(self.layer3(x))
    # x = torch.relu(self.layer4(x))
    return self.output(x)    