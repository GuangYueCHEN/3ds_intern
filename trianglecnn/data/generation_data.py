import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
import numpy as np
from models.layers.mesh import Mesh

class GenerationData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = self.make_dataset(self.dir)
        self.size = len(self.paths)
        self.get_mean_std()
        # # modify for network later.
        opt.nclasses = 0
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(file=path, opt=self.opt, hold_history=True, export_folder=self.opt.export_folder)
        meta = {}
        meta['mesh'] = mesh
        # get edge features
        face_features = mesh.extract_features()
        meta['num_triangle'] = mesh.faces_count
        face_features = pad(face_features, self.opt.ninput_faces)
        meta['face_features'] = (face_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    @staticmethod
    def make_dataset(path):
        meshes = []
        assert os.path.isdir(path), '%s is not a valid directory' % path

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    path = os.path.join(root, fname)
                    meshes.append(path)

        return meshes