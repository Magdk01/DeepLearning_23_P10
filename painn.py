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
        
        self.a = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128,384)
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
        
        # top v-block
        v = self.v #TODO: Need to understand the supossed linear combination
        u = self.v #TODO: Need to understand the supossed linear combination
        
        # s-block
        s_stack = torch.stack((torch.norm(self.v), self.s))
        split = self.a(s_stack)
        
        # left v-block continues
        v = hadamard(u, split[:128])
        
        # right v-block continues
        s = v #TODO: What does the scalar product need to be between?
        s = hadamard(s, split[128:128 * 2])
        s = torch.add(s, split[:128 * 2])
        
        return v, s  