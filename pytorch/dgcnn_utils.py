import torch

#def print(*args):
#    return

def knn(x, k=20):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    #nvtx.range_push("ggf")
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    #idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    num_dims = x.size(1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((x,feature-x), dim=3).permute(0, 3, 1, 2).contiguous()
    #nvtx.range_pop()
  
    return feature

def get_graph_feature_DA(x, k=20, idx=None):

#    nvtx.range_push("ggf")    
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    #idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    point_cloud_shape = x.size()
    batch_size = point_cloud_shape[0]
    num_points = point_cloud_shape[2]
    num_dims = point_cloud_shape[1]

    idx_ = torch.arange(batch_size) * num_points

    idx_ = idx_.view(batch_size, 1, 1)

    point_cloud_flat = x.reshape(-1, num_dims)

    point_cloud_neighbors = point_cloud_flat[idx + idx_]
    
    point_cloud_neighbors = torch.max(point_cloud_neighbors, dim=-2, keepdim=True)[0]
    
    x = x.view(batch_size, num_points, 1, num_dims)
    
    feature = torch.cat((x,point_cloud_neighbors-x), dim=3).permute(0, 3, 1, 2).contiguous()
#    nvtx.range_pop()
    return feature