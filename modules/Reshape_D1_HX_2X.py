import torch
import torch.nn as nn

class Reshape_D1_HX_2X(nn.Module):     # Half, 2X
   
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res % 2 == 0, f"Input {in_res} must have even resolution."
        out_res = in_res // 2
        assert out_res % nWin == 0, f"Output {out_res} must have exactly {nWin} windows."
        assert out_res % pWin == 0, f"Output {out_res} must be divisible with {pWin}."
        out_dim = in_dim * 2    # lossless
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, ss=None):
        B, L, C = x.shape
        # assert C == self.dim
        
        x = x.view(B, L//2, 2, C)
        x = x.view(B, L//2, 2 * C)
        
        return x

class Reshape_D1_2X_HX(nn.Module):

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, ss=None):
        B, L, C = x.shape       
        # assert self.dim == C
        
        x = x.view(B, L, 2, C//2)
        x = x.view(B, L * 2, C//2)
        
        return x
