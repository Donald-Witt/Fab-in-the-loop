#Copyright ©Donald Witt 2023. All rights reserved.
#Code used to define the spectral predictor
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

class Network(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,fc3_dims,fc4_dims,fc5_dims,fc6_dims, n_actions, name,
                 chkpt_dir='./ddpg',usemodel=True):
        self.usemodel=usemodel
        super(Network, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name)
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        T.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        T.nn.init.uniform_(self.fc3.bias.data, -f3, f3)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        f4 = 1./np.sqrt(self.fc4.weight.data.size()[0])
        T.nn.init.uniform_(self.fc4.weight.data, -f4, f4)
        T.nn.init.uniform_(self.fc4.bias.data, -f4, f4)
        self.bn4 = nn.LayerNorm(self.fc4_dims)
        
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        f5 = 1./np.sqrt(self.fc5.weight.data.size()[0])
        T.nn.init.uniform_(self.fc5.weight.data, -f5, f5)
        T.nn.init.uniform_(self.fc5.bias.data, -f5, f5)
        self.bn5 = nn.LayerNorm(self.fc5_dims)
        
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        f6 = 1./np.sqrt(self.fc6.weight.data.size()[0])
        T.nn.init.uniform_(self.fc6.weight.data, -f6, f6)
        T.nn.init.uniform_(self.fc6.bias.data, -f6, f6)
        self.bn6 = nn.LayerNorm(self.fc6_dims)

        f7 = 0.003
        self.mu = nn.Linear(self.fc6_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f7, f7)
        T.nn.init.uniform_(self.mu.bias.data, -f7, f7)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')

        self.to(self.device)

    def reinitialize_optimizer(self,alpha):
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        
        x = F.relu(x)
        x = self.fc5(x)
        x = self.bn5(x)
        
        x = F.relu(x)
        x = self.fc6(x)
        x = self.bn6(x)
        
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
        self.eval()
