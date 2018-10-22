import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable


class PlasticNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(PlasticNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Parameter((.01 * torch.randn(hidden_dim, hidden_dim)), requires_grad=True)
        self.alpha = nn.Parameter((.01 * torch.randn(hidden_dim, hidden_dim)), requires_grad=True)
        self.eta = nn.Parameter((.01 * torch.ones(1)), requires_grad=True)

        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(-1)
        self.activ = nn.Tanh()

    def forward(self, x, hidden, hebb):
        x = Variable(torch.from_numpy(x))
        hout = self.activ(self.i2h(x) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
        hebb = (1 - self.eta) * hebb + self.eta * hidden.view(-1, self.batch_size).mm(hout)

        output = self.softmax(self.h2o(hout))
        return output, hidden, hebb

    def initial_zero_hebb(self):
        return Variable(torch.zeros(self.hidden_dim, self.hidden_dim), requires_grad=False)

    def initial_zero_state(self):
        return Variable(torch.zeros(self.batch_size, self.hidden_dim), requires_grad=False)