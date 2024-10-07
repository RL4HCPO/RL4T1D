import torch.nn as nn


def NormedLinear(*args, scale=1.0):
    out = nn.Linear(*args)
    # print("normed linear")
    # print(out.weight.norm(dim=1, p=2, keepdim=True))
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    return out
