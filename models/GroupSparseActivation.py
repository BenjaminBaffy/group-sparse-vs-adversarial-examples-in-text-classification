import torch


# Taken from Zoltán Milacski: https://github.com/srph25/group-ksparse-temporal-cnns/blob/aeee40a8c3d650afd4fc6d704f1a1f1d1d27c7d6/utils/ops.py#L85

class GroupSparseActivation(torch.nn.Module):
    def __init__(self, k, groups, axis_group, axis_sparse, norm=2, epsilon=None):
        super(GroupSparseActivation, self).__init__()
        self.k = k
        self.groups = groups

        if isinstance(axis_group, int):
            self.axis_group = (axis_group,)
        elif isinstance(axis_group, list):
            self.axis_group = tuple(axis_group)

        if isinstance(axis_sparse, int):
            self.axis_sparse = (axis_sparse,)
        elif isinstance(axis_sparse, list):
            self.axis_sparse = tuple(axis_sparse)

        assert(1 - bool(set(self.axis_group) & set(self.axis_sparse)))

        self.epsilon = epsilon
        if self.epsilon is None:
            self.epsilon = 10e-10

        self.norm=norm


    def forward(self, x:torch.Tensor):

        axis_complement = tuple(set(range(x.dim())) - set(self.axis_group) - set(self.axis_sparse))
        shape_reduce_group = int(torch.prod([x.size(dim=j) for j in self.axis_group]))
        shape_reduce_sparse = int(torch.prod([x.size(dim=j) for j in self.axis_sparse]))
        
        _k = min(self.k, shape_reduce_sparse)
        inputs_permute_dimensions = x.permute(axis_complement + self.axis_sparse + self.axis_group)
        inputs_permute_dimensions_reshape = inputs_permute_dimensions.reshape(-1, shape_reduce_sparse, shape_reduce_group)
        norm_group_permute_dimensions_reshape = self.group_norms(inputs=inputs_permute_dimensions_reshape, groups=self.groups, dim=-1, norm=norm, epsilon=epsilon)
        norm_group_permute_dimensions_reshape = norm_group_permute_dimensions_reshape.permute(0, 2, 1)
        norm_group_permute_dimensions_reshape = norm_group_permute_dimensions_reshape.reshape(-1, shape_reduce_sparse)
        _, indices = torch.topk(norm_group_permute_dimensions_reshape, k=_k)
        scatter_indices = torch.cat([(torch.arange(norm_group_permute_dimensions_reshape.size(0))[:, None] * torch.ones(1, _k, dtype=torch.int32))[:, :, None], indices[:, :, None]])
        scatter_updates = torch.ones((int(norm_group_permute_dimensions_reshape.size(0)), _k))
        mask_group_permute_dimensions_reshape = norm_group_permute_dimensions_reshape.index_add(0, scatter_indices, scatter_updates).to(torch.float)
        mask_group_permute_dimensions_reshape = mask_group_permute_dimensions_reshape.reshape(-1, self.groups, shape_reduce_sparse)
        mask_group_permute_dimensions_reshape = mask_group_permute_dimensions_reshape.permute(0, 2, 1)
        mask_permute_dimensions_reshape = (mask_group_permute_dimensions_reshape[:, :, :, None] * torch.ones((1, 1, 1, floor_div(shape_reduce_group, groups))))
        mask_permute_dimensions = mask_permute_dimensions_reshape.reshape(inputs_permute_dimensions.size())
        mask = mask_permute_dimensions.permute(tuple(torch.argsort(axis_complement + self.axis_sparse + self.axis_group)))
        return  mask * x


    def group_norms(self, inputs:torch.Tensor, dim):
        if dim == -1:
            dim = inputs.dim() - 1
        inputs_group = torch.split(inputs, split_size_or_sections=self.groups, dim=dim)
        inputs_group = torch.cat([torch.unsqueeze(t, dim=dim) for t in inputs_group], dim=dim)
        return n_p(inputs_group, p=self.norm, dim=(dim + 1), epsilon=self.epsilon)
    

def n_p(Z, p, dim, epsilon=None):
    return torch.pow(torch.sum(torch.pow(torch.abs(Z), p), dim=dim) + epsilon, 1. / p)


def floor_div(x, y):
    return torch.ones((x,))[::y].size(0)