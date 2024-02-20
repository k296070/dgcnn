import torch
import torch.cuda.nvtx as nvtx

def print(*args):
    return

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    print("ip in", points.shape, idx.shape)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    print("ip out", new_points.shape)
    return new_points

def knn(x, k=20):
    nvtx.range_push("knn")
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    nvtx.range_pop()
    return idx


def get_graph_feature(x, k=20, idx=None):
    nvtx.range_push("ggf")
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    #idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((x,feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
    nvtx.range_pop()
  
    return feature

def get_graph_feature_DA(x, k=20, idx=None):

    nvtx.range_push("ggf")    
    batch_size = x.size(0)
    num_points = x.size(2)
    print("input",x.shape)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    #idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)

    num_dims = x.size(1)
    print("after view",x.shape)
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    
    print("feature",feature.shape)
    feature = feature.view(batch_size, num_points, k, num_dims) 
    feature = feature.permute(0, 3, 1, 2)
    feature = feature.max(dim= -1, keepdim=True)[0]
    feature = feature.permute(0, 2, 3, 1)
    print("view before",x.shape)
    x = x.view(batch_size, num_points, 1, num_dims)
    print("after view",x.shape)
    feature = torch.cat((x,feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
    nvtx.range_pop()

    return feature