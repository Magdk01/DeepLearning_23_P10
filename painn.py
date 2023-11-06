import torch
from torch import nn
import numpy as np

def hadamard(m1, m2):
    assert m1.shape == m2.shape
    return m1 * m2

class painn(nn.Module):
    def __init__(self, atomic_numbers, positional_encodings) -> None:
        
        self.s = atomic_numbers
        self.r = positional_encodings
        self.v = torch.zeros_like(self.z)
        
        self.ø = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 384),
        )
        
        
class message(nn.Module):
    def __init__(self, atomic_numbers, positional_encodings, feature_vector) -> None:
        
        self.s = atomic_numbers
        self.r = positional_encodings
        self.v = feature_vector
    
    def forward(self):
        
        # s-block
        ø_out = self.ø(self.s)
        
        assert len(ø_out) == 384
        
        # left r-block
        r = self.__rbf(self.r)
        r = nn.Linear(20, 384) (r)
        w = self.__fcut(r)
        
        assert len(w) == 384
        
        split = hadamard(w, ø_out)
        
        out_s = torch.sum(split[128: 128 * 2], axis= 0)
        
        # right r-block
        r = r / torch.norm(self.r)
        r = hadamard(split[128 * 2:], r)
        
        # v-block
        v = hadamard(split[:128], self.v)
        v = torch.add(r, v)
        
        out_v = torch.sum(v, axis= 0)
        
        return out_v, out_s
        
        
    
    def __rbf(self):
        raise NotImplementedError()
    
    def __fcut(self):
        raise NotImplementedError()
    
    
class update(nn.Module):
    def __init__(self, atomic_numbers, positional_encodings, feature_vector) -> None:
        
        self.s = atomic_numbers
        self.r = positional_encodings
        self.v = feature_vector
        
        self.silu = nn.SiLU()
        
    def foward(self):
        
        # left v-block
        v = self.v #TODO
        
        # s-block
        torch.stack((torch.norm(self.v), self.s))