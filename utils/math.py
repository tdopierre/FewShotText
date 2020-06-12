import torch


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_similarity(x, y):
    x = (x / x.norm(dim=1).view(-1, 1))
    y = (y / y.norm(dim=1).view(-1, 1))

    return x @ y.T
