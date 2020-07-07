from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh


class Mesh:

    def __init__(self, file=None, opt=None, hold_history=False, export_folder=''):
        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.faces = self.gemm_faces =None
        self.pool_count = 0
        fill_mesh(self, file, opt)
        self.export_folder = export_folder
        self.history_data = None
        if hold_history:
            self.init_history()
        self.export()

    def extract_features(self):
        return self.features

    def merge_vertices(self, face_id):
        #done
        self.remove_face(face_id)
        face = self.faces[face_id]
        v_a = self.vs[face[0]]
        v_b = self.vs[face[1]]
        v_c = self.vs[face[2]]
        # update pA
        v_a.__iadd__(v_b).__iadd__(v_c)
        v_a.__itruediv__(3)
        self.v_mask[face[1]] = False
        self.v_mask[face[2]] = False
        mask = self.faces == face[1]
        mask2 = self.faces == face[2]

        self.vf[face[0]].extend(self.vf[face[1]])
        self.vf[face[0]].extend(self.vf[face[2]])
        self.faces[mask] = face[0]
        self.faces[mask2] = face[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_face(self, face_id):
        vs = self.faces[face_id]
        for v in vs:
            if face_id not in self.vf[v]:
                print("error remove face")
            self.vf[v].remove(face_id)

    def clean(self, faces_mask, groups):
        # done
        faces_mask = faces_mask.astype(bool)
        torch_mask = torch.from_numpy(faces_mask.copy())
        self.gemm_faces = self.gemm_faces[faces_mask]
        self.faces = self.faces[faces_mask]
        new_vf = []
        faces_mask = np.concatenate([faces_mask, [False]])
        new_indices = np.zeros(faces_mask.shape[0], dtype=np.int32)
        new_indices[-1] = -1
        new_indices[faces_mask] = np.arange(0, np.ma.where(faces_mask)[0].shape[0])
        self.gemm_faces[:, :] = new_indices[self.gemm_faces[:, :]]
        for v_index, vf in enumerate(self.vf):
            update_vf = []
            # if self.v_mask[v_index]:
            for f in vf:
                update_vf.append(new_indices[f])
            new_vf.append(update_vf)
        self.vf = new_vf
        self.__clean_history(groups, torch_mask)
        self.pool_count += 1
        self.export()


    def export(self, file=None, vcolor=None):
        if file is None:
            if self.export_folder:
                filename, file_extension = os.path.splitext(self.filename)
                file = '%s/%s_%d%s' % (self.export_folder, filename, self.pool_count, file_extension)
            else:
                return
        faces = []
        vs = self.vs[self.v_mask]

        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(self.faces) - 1):
                f.write("f %d %d %d\n" % (self.faces[face_id][0] + 1, self.faces[face_id][1] + 1, self.faces[face_id][2] + 1))
            f.write("f %d %d %d" % (self.faces[-1][0] + 1, self.faces[-1][1] + 1, self.faces[-1][2] + 1))


    def export_segments(self, segments):
        if not self.export_folder:
            return
        cur_segments = segments
        for i in range(self.pool_count + 1):
            filename, file_extension = os.path.splitext(self.filename)
            file = '%s/%s_%d%s' % (self.export_folder, filename, i, file_extension)
            fh, abs_path = mkstemp()
            face_key = 0
            with os.fdopen(fh, 'w') as new_file:
                with open(file) as old_file:
                    for line in old_file:
                        if line[0] == 'f':
                            new_file.write('%s %d' % (line.strip(), cur_segments[face_key]))
                            if face_key < len(cur_segments):
                                face_key += 1
                                new_file.write('\n')
                        else:
                            new_file.write(line)
            os.remove(file)
            move(abs_path, file)
            if i < len(self.history_data['faces_mask']):
                cur_segments = segments[:len(self.history_data['faces_mask'][i])]
                cur_segments = cur_segments[self.history_data['faces_mask'][i]]


    def init_history(self):
        self.history_data = {
                               'groups': [],
                               'gemm_faces': [self.gemm_faces.copy()],
                               'occurrences': [],
                               'old2current': np.arange(self.faces_count, dtype=np.int32),
                               'current2old': np.arange(self.faces_count, dtype=np.int32),
                               'faces_mask': [torch.ones(self.faces_count,dtype=torch.bool)],
                               'faces_count': [self.faces_count],
                              }
        if self.export_folder:
            self.history_data['collapses'] = MeshUnion(self.faces_count)

    def union_groups(self, source, target):
        if self.export_folder and self.history_data:
            self.history_data['collapses'].union(self.history_data['current2old'][source], self.history_data['current2old'][target])
        return

    def remove_group(self, index):
        if self.history_data is not None:
            self.history_data['faces_mask'][-1][self.history_data['current2old'][index]] = 0
            self.history_data['old2current'][self.history_data['current2old'][index]] = -1
            if self.export_folder:
                self.history_data['collapses'].remove_group(self.history_data['current2old'][index])

    def get_groups(self):
        return self.history_data['groups'].pop()

    def get_occurrences(self):
        return self.history_data['occurrences'].pop()
    
    def __clean_history(self, groups, pool_mask):
        if self.history_data is not None:
            mask = self.history_data['old2current'] != -1
            self.history_data['old2current'][mask] = np.arange(self.faces_count, dtype=np.int32)
            self.history_data['current2old'][0: self.faces_count] = np.ma.where(mask)[0]
            if self.export_folder != '':
                self.history_data['faces_mask'].append(self.history_data['faces_mask'][-1].clone())
            self.history_data['occurrences'].append(groups.get_occurrences())
            self.history_data['groups'].append(groups.get_groups(pool_mask))
            self.history_data['gemm_faces'].append(self.gemm_faces.copy())
            self.history_data['faces_count'].append(self.faces_count)
    
    def unroll_gemm(self):
        self.history_data['gemm_faces'].pop()
        self.gemm_faces = self.history_data['gemm_faces'][-1]
        self.history_data['faces_count'].pop()
        self.faces_count = self.history_data['faces_count'][-1]

    def get_edge_areas(self):
        return self.edge_areas
