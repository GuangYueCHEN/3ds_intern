import torch
from models.layers.mesh import Mesh

class SamplePoints(object):
    r"""Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.
    Args:
        num (int): The number of points to sample.
    """
    def __init__(self, num, device):
        self.num = num
        self.device = device

    def __call__(self, mesh, pos = None):
        if pos == None:
            pos = torch.from_numpy(mesh.vs).float().to(self.device)
        else:
            pos = pos[0:len(mesh.v_mask)]
        face = torch.from_numpy(mesh.faces).long().to(self.device)

        pos_max = pos.max()
        pos = pos / pos_max
        area = (pos[face[:,1]] - pos[face[:,0]]).cross(pos[face[:,2]] - pos[face[:,0]])
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num - len(pos), replacement=True)
        face = face[sample,:]

        frac = torch.rand(self.num - len(pos), 2, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[:,1]] - pos[face[:,0]]
        vec2 = pos[face[:,2]] - pos[face[:,0]]

        pos_sampled = pos[face[:,0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        pos_sampled = torch.cat((pos, pos_sampled),dim = 0)
        return pos_sampled

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)