import torch
import torch.nn as nn
from threading import Thread
import numpy as np
from heapq import heappop, heapify
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_conv import MeshConv
from torch.nn import ConstantPad2d

class MeshUnpoolOptimisor(nn.Module):
    """
    Vertex position optimizer
    to move the vertices according to the face features
    """
    def __init__(self , name=''):
        super(MeshUnpoolOptimisor, self).__init__()
        self.rate = None
        self.name = name

    def __call__(self, features, vs, meshes):
        return self.forward(features, vs, meshes)

    def forward(self, features, vs,  meshes):
        """
        There are two ways to move the vertex
        the sum of the 3-dimentional vector of adjacent faces
        the average of the 3-dimentional vector of adjacent faces
        Need to change the move rate for different options.
        """
        x = torch.transpose(features, 1, 2)
        for i, mesh in enumerate(meshes):
            self.rate = mesh.min_edge_length
            #cpt = torch.ones((len(vs[i]), 1), device=features.device)
            _sum = torch.zeros(vs[i].shape, device=features.device)
            for j, face in enumerate(mesh.faces):
                for k in range(3):
                    _sum[face[k]] += x[i][j]
                    #cpt[face[k]] += 1
            #_sum /= cpt
            mesh.vs += self.rate * _sum[0:len(mesh.vs)].detach().cpu().numpy()
            vs[i] += self.rate * _sum
            mesh.export(name='(output_of_MeshUnpoolOptimisor)'+self.name)
        return vs, meshes


class MeshUnpoolValenceOptimisor(nn.Module):
    """
    MeshUnpoolValenceOptimisor flips the edges around the high valence vertex
    when flip edge it change also the features
    Perhaps it's a better way to transfer features by matrix operation like pooling an unpooling, quicker than my codes.
    """
    def __init__(self, max_iter=200, conv = False, channel = 0):
        super(MeshUnpoolValenceOptimisor, self).__init__()
        self.max_iter = max_iter
        self.conv = None
        if conv:
            self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)
            self.conv2 = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=2)
            self.channel = channel

    def __call__(self, meshes, fe):
        return self.forward(meshes, fe)

    def forward(self, meshes, fe):
        for i, mesh in enumerate(meshes):
            v_score = np.zeros(len(mesh.vs), dtype=int)
            for j in range(len(mesh.vs)):
                v_score[j] = np.sum(mesh.faces == j) - 8
            face_scores = self.build_score(mesh,v_score)
            score_sum = np.sum(np.abs(face_scores), axis=1)
            _index = np.arange(len(mesh.faces))
            _sum = np.array((_index, score_sum)).T
            _sum = _sum[(-_sum[:, 1]).argsort()]
            count = 0
            j = 0
            while count < self.max_iter and j < len(mesh.faces):
                index = _sum[j,0]
                a= mesh.faces[index]
                c= mesh.gemm_faces[index]
                v_min = np.argmin(face_scores[index])
                index_neigh = mesh.gemm_faces[index, (v_min + 1) % 3]
                v_nei = np.setdiff1d(mesh.faces[index_neigh], mesh.faces[index])
                v_nei_min = np.argmax(mesh.faces[index_neigh] == v_nei)
                score_pre = np.abs(v_score[mesh.faces[index, v_min]]) + \
                            np.abs(v_score[mesh.faces[index_neigh, v_nei_min]]) + \
                            np.abs(v_score[mesh.faces[index, (v_min+1) % 3]]) + \
                            np.abs(v_score[mesh.faces[index, (v_min + 2) % 3]])
                score_after = np.abs(v_score[mesh.faces[index, v_min]]+1) + \
                            np.abs(v_score[mesh.faces[index_neigh, v_nei_min]]+1) + \
                            np.abs(v_score[mesh.faces[index, (v_min+1) % 3]]-1) + \
                            np.abs(v_score[mesh.faces[index, (v_min + 2) % 3]]-1)
                if score_after < score_pre and \
                        self.not_contain_edge(mesh, mesh.faces[index_neigh, v_nei_min], mesh.faces[index, v_min]):
                    b = mesh.faces[index_neigh]
                    d = mesh.gemm_faces[index_neigh]
                    v_nei = np.setdiff1d(mesh.faces[index_neigh], mesh.faces[index])
                    assert len(v_nei) == 1
                    """if mesh.faces[index_neigh, v_nei_min] != v_nei[0]:
                        j += 1
                        continue"""
                    face = np.array([mesh.faces[index, v_min], mesh.faces[index, (v_min+1)%3], mesh.faces[
                            index_neigh, v_nei_min]]).reshape(1, 3)
                    face_neigh = np.array([mesh.faces[index_neigh,v_nei_min] , mesh.faces[index, (v_min+2) % 3],
                            mesh.faces[index, v_min]]).reshape(1, 3)
                    gemm_face = np.array([mesh.gemm_faces[index, v_min], mesh.gemm_faces[index_neigh, (v_nei_min + 2) % 3], index_neigh]).reshape(1,3)
                    gemm_face_neigh = np.array([mesh.gemm_faces[index_neigh, v_nei_min], mesh.gemm_faces[index,
                            (v_min + 2) % 3], index]).reshape(1,3)
                    v_score[mesh.faces[index, v_min]] += 1
                    v_score[mesh.faces[index_neigh, v_nei_min]] += 1
                    v_score[mesh.faces[index, (v_min + 1) % 3]] -= 1
                    v_score[mesh.faces[index, (v_min + 2) % 3]] -= 1

                    face_scores = self.build_score(mesh,v_score)
                    score_sum = np.sum(np.abs(face_scores), axis=1)
                    mesh.faces[index] = face
                    mesh.gemm_faces[index] = gemm_face

                    gemm_neigh_a = mesh.gemm_faces[gemm_face[0, 1]]
                    gemm_neigh_a[gemm_neigh_a == index_neigh] = index
                    mesh.gemm_faces[gemm_face[0, 1]] = gemm_neigh_a

                    mesh.faces[index_neigh] = face_neigh
                    mesh.gemm_faces[index_neigh] = gemm_face_neigh

                    gemm_a = mesh.gemm_faces[gemm_face_neigh[0, 1]]
                    gemm_a[gemm_a == index] = index_neigh
                    mesh.gemm_faces[gemm_face_neigh[0, 1]] = gemm_a

                    _sum = np.array((_index, score_sum)).T
                    _sum = _sum[(-_sum[:, 1]).argsort()]
                    if self.conv:
                        old_fe = fe[i, :, [index, index_neigh]].unsqueeze(dim = 0)
                        old_fe2 = fe[i, :, [ index_neigh, index]].unsqueeze(dim = 0)
                        fe[i, :, index] = self.conv(old_fe)[0, :, 0]
                        fe[i, :, index_neigh] = self.conv2(old_fe2)[0, :, 0]
                    count += 1
                    #mesh.export(name="flip_edge"+str(count))
                    j = -1
                j += 1
            MeshUnpoolLearned.check_gemm(mesh)
            #print(count)
            mesh.export(name = "flip_edge" + str(self.channel))
        return fe, meshes

    @staticmethod
    def build_score(mesh, v_score):
        face_scores = np.zeros(mesh.faces.shape, dtype=int)
        for j, scores in enumerate(face_scores):
            for k in range(3):
                face_scores[j, k] = v_score[mesh.faces[j, k]]
        return face_scores

    @staticmethod
    def not_contain_edge(mesh, v1, v2):
        for face in mesh.faces:
            if v1 in face and v2 in face:
                return False
        return True

    @staticmethod
    def volume_valid(mesh, f1, f2, seuil):
        v_ids = np.union1d(mesh.faces[f1], mesh.faces[f2])
        max_edge = np.max([ mesh.vs[v_ids[0]]-mesh.vs[v_ids[3]] ,  mesh.vs[v_ids[1]]-mesh.vs[v_ids[3]], mesh.vs[v_ids[2]]-mesh.vs[v_ids[3]] ] )
        assert len(v_ids) == 4
        volume = np.abs(np.dot( (mesh.vs[v_ids[0]]-mesh.vs[v_ids[3]]) / max_edge,
                                np.cross((mesh.vs[v_ids[1]]-mesh.vs[v_ids[3]]) / max_edge,
                                         (mesh.vs[v_ids[2]]-mesh.vs[v_ids[3]]) / max_edge)))/6.
        return volume < seuil

class MeshUnpoolLearned(nn.Module):
    def __init__(self, unroll_target, vs_target, multi_thread=False):
        super(MeshUnpoolLearned, self).__init__()
        self.unroll_target = unroll_target
        self.vs_target = vs_target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__vs = None
        self.__updated_vs = None
        self.__meshes = None
        self.v_count = None

    def __call__(self, features, vs, meshes):
        return self.forward(features, vs, meshes)

    def forward(self, fe, vs, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        self.__updated_vs = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__vs = vs
        self.__meshes = meshes
        self.v_count = [len(mesh.vs) for mesh in meshes]
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index, self.unroll_target)))
                pool_threads[-1].start()
            else:
                self.__unpool_main(mesh_index, self.unroll_target)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_vs = torch.cat(self.__updated_vs).view(len(meshes), self.vs_target, -1)
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.unroll_target)
        return out_features, out_vs, meshes

    def __unpool_main(self, mesh_index, unroll_target):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.faces_count], mesh.faces_count,mesh)
        face_groups = MeshUnion(mesh.faces_count, self.__fe.device)
        vs_groups = MeshUnion(self.v_count[mesh_index], self.__fe.device)
        not_split = np.ones(mesh.faces_count, dtype=np.bool)
        while mesh.faces_count + 6 < unroll_target:
            # print("face count " + str(mesh.faces_count))
            value, face_id = heappop(queue)
            face_id = int(face_id)
            if self.check_valid(mesh, face_id) and not_split[face_id]:
                not_split = self.__unpool_face(mesh, mesh_index, face_id, face_groups, vs_groups, not_split)
        mesh.pool_count -= 1
        mask = np.ones(mesh.faces_count, dtype=np.bool)
        # mesh.export(name='unpool')
        fe = face_groups.rebuild_features(self.__fe[mesh_index], mask, self.unroll_target)
        padding_b = self.vs_target - vs_groups.groups.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b), 0)
            vs_groups.groups = padding_b(vs_groups.groups)
        vs = vs_groups.rebuild_vs_average(self.__vs[mesh_index], self.vs_target)
        self.__updated_fe[mesh_index] = fe
        self.__updated_vs[mesh_index] = vs

    def __unpool_face(self, mesh, mesh_index, face_id, face_groups, vs_groups, mask_split):
        v_ids = mesh.faces[face_id, :]
        new_v_ids = np.zeros(3, dtype=int)
        mask_split[mesh.gemm_faces[face_id]] = False
        np.append(mask_split,False)
        for i in range(3):
            new_v_ids[i] = self.v_count[mesh_index]
            un = (mesh.vs[v_ids[i], :] + mesh.vs[v_ids[(i+1) % 3], :]) / 2.
            mesh.vs = np.concatenate((mesh.vs, un.reshape(1, 3)), axis=0)
            vs_groups.copy(v_ids[i])
            vs_groups.union(v_ids[(i+1) % 3], new_v_ids[i])
            self.v_count[mesh_index] += 1
        mesh.v_mask = np.append(mesh.v_mask, [True, True, True])
        new_face_ids = np.zeros(3, dtype=int)
        for i in range(3):
            mesh.faces = np.concatenate((mesh.faces, np.array([v_ids[i], new_v_ids[i],
                                                               new_v_ids[(i+2) % 3]]).reshape(1, 3)), axis=0)
            mesh.gemm_faces = np.concatenate((mesh.gemm_faces, np.array([mesh.gemm_faces[face_id,i], face_id,
                                                               mesh.gemm_faces[face_id,(i+2) % 3]]).reshape(1, 3)), axis=0)
            face_groups.copy(face_id)
            new_face_ids[i] = mesh.faces_count
            mesh.faces_count += 1
            mask_split =  np.append(mask_split, False)
        mesh.faces[face_id] = new_v_ids.reshape(1, 3)
        for i in range(3):
            face = mesh.gemm_faces[face_id, i]
            if face == -1:
                mesh.gemm_faces[new_face_ids[i], i] = -1
                mesh.gemm_faces[new_face_ids[(i+1) % 3], i] = -1
            else:
                for j in range(3):
                    if mesh.faces[face, j] not in v_ids:
                        mesh.faces = np.concatenate(
                            (mesh.faces, mesh.faces[face].reshape(1, 3)), axis=0)
                        mesh.gemm_faces = np.concatenate((mesh.gemm_faces, mesh.gemm_faces[face].reshape(1, 3)), axis=0)
                        new_face = mesh.faces_count
                        face_groups.copy(face)
                        mesh.faces_count += 1
                        mask_split = np.append(mask_split, False)
                        mesh.faces[face, (j+1) % 3] = new_v_ids[i]
                        mesh.faces[new_face, (j+2) % 3] = new_v_ids[i]
                        other_side_id = mesh.gemm_faces[face, j]
                        for k in range(3):
                            if mesh.gemm_faces[other_side_id, k] == face:
                                mesh.gemm_faces[other_side_id, k] = new_face
                        mesh.gemm_faces[face, j] = new_face
                        mesh.gemm_faces[face, (j+1) % 3] = new_face_ids[i]
                        mesh.gemm_faces[new_face, (j + 2) % 3] = face
                        mesh.gemm_faces[new_face, (j + 1) % 3] = new_face_ids[(i+1) % 3]
                        mesh.gemm_faces[new_face_ids[(i+1) % 3], 2] = new_face
                        break
        for i in range(3):
            mesh.gemm_faces[face_id, i] = new_face_ids[(i+1) % 3]
        if not self.check_gemm(mesh):
            print(face_id)
            print(new_face_ids)
            print("ztf")
        return mask_split


    @staticmethod
    def check_valid(mesh, face_id):
        ids = mesh.gemm_faces[face_id]
        _id = np.intersect1d(np.intersect1d(mesh.gemm_faces[ids[0]], mesh.gemm_faces[ids[1]]), mesh.gemm_faces[ids[2]])
        return len(_id) != 0 and _id[0] == face_id

    @staticmethod
    def is_boundaries(mesh, face_id):
        for face in mesh.gemm_faces[face_id]:
            if face == -1 or -1 in mesh.gemm_faces[face]:
                return True
        return False


    @staticmethod
    def __build_queue(features, faces_count):
        squared_magnitude = torch.sum(features * features, 0)

        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        face_ids = torch.arange(faces_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, face_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    """
    @staticmethod
    def __build_queue(features, faces_count,mesh):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        v_score = np.zeros(len(mesh.vs), dtype=int)
        for j in range(len(mesh.vs)):
            v_score[j] = np.sum(mesh.faces == j) - 8
        face_scores = MeshUnpoolValenceOptimisor.build_score(mesh, v_score)
        face_scores = np.concatenate((face_scores, np.zeros((faces_count,1))), axis = 1)
        score_max = torch.from_numpy(np.max(face_scores, axis=1).reshape(faces_count,1)).to(features.device)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        face_ids = torch.arange(faces_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude - score_max, face_ids), dim=-1).tolist()
        heapify(heap)
        return heap
    """

    @staticmethod
    def check_gemm(mesh):
        count = 0
        for i, face in enumerate(mesh.faces):
            for j in range(3):
                index_neigh = mesh.gemm_faces[i,j]
                if len(np.intersect1d(face, mesh.faces[index_neigh])) != 2:
                    count += 1
        if count == 0:
            return True
        else:
            print("some gemm incorrect")
            print(count)
            print(mesh.filename)
            return False

    @staticmethod
    def __redirect_faces(mesh, face_a_id, face_id):
        # done
        for index, face in enumerate(mesh.gemm_faces[face_a_id]):
            if face == face_id:
                mesh.gemm_faces[face_a_id, index] = face_id
