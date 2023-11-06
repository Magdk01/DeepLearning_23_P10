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
        
        self.message_model = message()
        self.update_model = update()
        
        self.output_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.SiLU(),
            nn.Linear(128,128)
        )
        
        def forward(self):
            for _ in range(3):
                v, s = self.v.copy(), self.s.copy()
                
                self.v, self.s = self.message_model()
                
                self.v = torch.add(self.v, v)
                self.s = torch.add(self.s, s)
                
                v, s = self.v.copy(), self.s.copy()
                
                self.v, self.s = self.update_model()
                
                self.v = torch.add(self.v, v)
                self.s = torch.add(self.s, s)
                
            out = self.output_layers(self.s)
            out = torch.sum(out)
            
            return out
                
            
class message(nn.Module):
    def __init__(self) -> None:
        
        self.ø = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 384),
        )
    
    def forward(self):
        
        s = self.s.copy()
        r = self.r.copy()
        v = self.v.copy()
        
        # s-block
        ø_out = self.ø(s)
        
        assert len(ø_out) == 384
        
        # left r-block
        r = self.__rbf(r)
        r = nn.Linear(20, 384) (r)
        w = self.__fcut(r)
        
        assert len(w) == 384
        
        split = hadamard(w, ø_out)
        
        out_s = torch.sum(split[128: 128 * 2], axis= 0)
        
        # right r-block
        r = r / torch.norm(r)
        r = hadamard(split[128 * 2:], r)
        
        # v-block
        v = hadamard(split[:128], v)
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
        
        self.a = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128,384)
        )
        
    def foward(self):
        
        s = self.s.copy()
        r = self.r.copy()
        v = self.v.copy()
        
        # top v-block
        v = v #TODO: Need to understand the supossed linear combination
        u = v #TODO: Need to understand the supossed linear combination
        
        # s-block
        s_stack = torch.stack((torch.norm(v), s))
        split = self.a(s_stack)
        
        # left v-block continues
        out_v = hadamard(u, split[:128])
        
        # right v-block continues
        s = v #TODO: What does the scalar product need to be between?
        s = hadamard(s, split[128:128 * 2])
        out_s = torch.add(s, split[:128 * 2])
        
        return out_v, out_s  