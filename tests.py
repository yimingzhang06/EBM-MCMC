import torch
from models import FCNet
from langevin import sample_langevin
from optimizers import *

def test_langevin():
    X = torch.tensor([[1,2],[4,5],[7,8],[11,12],[14,15]], dtype = torch.float)
    model = FCNet(2, 1, l_hidden=(50,))
    sample = sample_langevin(X, model, 0.1, 1)
