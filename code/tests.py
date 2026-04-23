from os import name

import torch
import attention
import math

def test_attention_scores():
    a = torch.tensor([  
        [[1.0, 0.0],
         [0.0, 1.0]],

        [[1.0, 1.0],
         [2.0, 0.0]]
    ]) #dim 2X2X2

    b = torch.tensor([
        [[1.0, 0.0],
         [0.0, 1.0]],
        
        [[1.0, 2.0],
         [3.0, 4.0]]
    ])#dim 2X2X2

    sqrt_d = math.sqrt(2)

    expected_output = torch.tensor([    
        [[1.0/sqrt_d, 0.0],
         [0.0, 1.0/sqrt_d]],

        [[3.0/sqrt_d, 2.0/sqrt_d],
         [7.0/sqrt_d, 6.0/sqrt_d]]
    ])

    A = attention.attention_scores(a, b)

    assert torch.allclose(A, expected_output)

if __name__ == "__main__":
    test_attention_scores()