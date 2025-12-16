import torch 
import torch.nn as nn 

class FFN(nn.Module): 
    def __init__(self, d_model, d_hidden):

        super().__init__() 

        self.expand = nn.Linear(d_model, d_hidden) 
        self.shrink = nn.Linear(d_hidden, d_model) 

    def forward(self, x): 
        x = self.expand(x) 
        x = torch.nn.functional.silu(x)
        x = self.shrink(x) 
        return x