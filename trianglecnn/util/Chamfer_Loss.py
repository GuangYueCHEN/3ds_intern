import torch
from torch.nn.modules.loss import _Loss
from util.sampling import SamplePoints
from torch import Tensor
from pytorch3d.loss import chamfer_distance

"""
Chamfer loss for generative network
Additions:
    loss area: the average of output meshes surface area, aim to penalize the auto-intersection.
    loss_volumes: the difference of volume between input and output
    loss_areas_ratio : the ratio between the biggest face and the smallest face of output mesh.
"""

class ChamferLoss(_Loss):
    def __init__(self, device, num = 3072, size_average=None,
                 reduce=None, reduction: str = 'mean') -> None:
        super(ChamferLoss, self).__init__(size_average, reduce, reduction)
        self.num = num
        self.device = device
        self.sample = SamplePoints(num, device)

    def forward(self, target, vs, input) -> Tensor:
        pointsa = torch.stack([self.sample(i) for i in input])
        pointsb = torch.stack([self.sample(mesh, vs[i]) for i,mesh in enumerate(target)])
        loss, _ = chamfer_distance(pointsb, pointsa, point_reduction='sum')
        loss += 1 *  self.loss_volumes(input, target, vs) + 0.001 * self.loss_areas_ratio(target,vs)
        return loss

    def loss_areas(self, meshes, vs):
        sum_area = torch.tensor(0.).to(self.device)
        for i, mesh in enumerate(meshes):
            face = torch.from_numpy(mesh.faces).long().to(self.device)
            pos = vs[i]
            pos_max = pos.max()
            pos = pos / pos_max
            area = (pos[face[:, 1]] - pos[face[:, 0]]).cross(pos[face[:, 2]] - pos[face[:, 0]])
            area = area.norm(p=2, dim=1).abs() / 2
            sum_area += torch.sum(area)
        return sum_area/len(meshes)


    def loss_areas_ratio(self, meshes, vs):
        sum_area_ratio = torch.tensor(0.).to(self.device)
        for i, mesh in enumerate(meshes):
            face = torch.from_numpy(mesh.faces).long().to(self.device)
            pos = vs[i]
            pos_max = pos.max()
            pos = pos / pos_max
            area = (pos[face[:, 1]] - pos[face[:, 0]]).cross(pos[face[:, 2]] - pos[face[:, 0]])
            area = area.norm(p=2, dim=1).abs() / 2
            max_area = torch.max(area)
            min_area = torch.min(area)
            sum_area_ratio += max_area / min_area
        return sum_area_ratio/len(meshes)

    def loss_volumes(self, inputs, targets, vs):
        sum_distance = torch.tensor(0.).to(self.device)
        for i, mesh in enumerate(targets):
            face_target = torch.from_numpy(mesh.faces).long().to(self.device)
            pos_target = vs[i]
            face_input = torch.from_numpy(inputs[i].faces).long().to(self.device)
            pos_input = torch.from_numpy(inputs[i].vs).float().to(self.device)
            pos_target = pos_target / pos_target.max()
            pos_input = pos_input / pos_input.max()
            volume_target = self.compute_volume(pos_target, face_target)
            volume_input = self.compute_volume(pos_input, face_input)
            sum_distance += (volume_target - volume_input).abs()
        return sum_distance/len(inputs)

    @staticmethod
    def compute_volume(pos, faces):
        volume = (pos[faces[:, 0]].view(-1)).dot( pos[faces[:, 1]].cross(pos[faces[:, 2]]).view(-1)) / 6.
        return volume.abs()

