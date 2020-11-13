import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify


class MeshPool(nn.Module):

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.faces_count], mesh.faces_count)
        mask = np.ones(mesh.faces_count, dtype=np.bool)
        face_groups = MeshUnion(mesh.faces_count, self.__fe.device)
        while mesh.faces_count > self.__out_target:
            # print("face count " + str(mesh.faces_count))
            value, face_id = heappop(queue)
            face_id = int(face_id)
            if mask[face_id]:
                self.__pool_face(mesh, face_id, mask, face_groups)
        mesh.clean(mask, face_groups)
        """ mesh.export(name = 'pool'+str(self.__out_target))"""
        fe = face_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_face(self, mesh, face_id, mask, face_groups):
        if self.is_boundaries(mesh, face_id):
            print("pool face is boundary")
            return False
        elif self.__clean_side(mesh, face_id, mask, face_groups) and self.__is_one_ring_valid(mesh, face_id, mask) \
                and mask[face_id]:
            """elif self.__clean_side(mesh, face_id, mask, face_groups) and self.__is_link_condition_valid(mesh, face_id, mask) \
                and mask[face_id]:"""
            for i in range(3):
                self.__pool_side(mesh, face_id, mask, face_groups, i)

            MeshPool.__remove_group(mesh, face_groups, face_id)
            mesh.merge_vertices(face_id)
            mask[face_id] = False
            MeshPool.__remove_group(mesh, face_groups, face_id)
            mesh.faces_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, face_id, mask, face_groups):
        for face in mesh.gemm_faces[face_id]:
            if mask[face]:
                flag = self.__clean_self(mesh, face, mask, face_groups)
                if not flag:
                    return False
        return True

    def __clean_self(self, mesh, face_id, mask, face_groups):
        # done
        if mesh.faces_count <= self.__out_target:
            return False
        invalid_faces = MeshPool.__get_invalids(mesh, face_id, face_groups)
        while len(invalid_faces) != 0 and mesh.faces_count > self.__out_target:
            if len(invalid_faces) > 3:
                # has cone
                return False
            self.__remove_invalid_faces(mesh, mask, face_groups, invalid_faces)
            if mesh.faces_count <= self.__out_target:
                return False
            if self.is_boundaries(mesh, face_id):
                print("clean side is boundary")
                return False
            invalid_faces = self.__get_invalids(mesh, face_id, face_groups)
        return True

    def __pool_side(self, mesh, face_id, mask, face_groups, index):
        # done
        face_a_index = mesh.gemm_faces[face_id, index]
        face_ids_a = np.setdiff1d(mesh.gemm_faces[face_a_index], [face_id])
        assert(len(face_ids_a) == 2)
        self.__redirect_side_faces(mesh, face_a_index, face_ids_a)
        MeshPool.__union_groups(mesh, face_groups, face_a_index, face_ids_a[0])
        MeshPool.__union_groups(mesh, face_groups, face_id, face_ids_a[0])
        MeshPool.__union_groups(mesh, face_groups, face_a_index, face_ids_a[1])
        MeshPool.__union_groups(mesh, face_groups, face_id, face_ids_a[1])
        mask[face_a_index] = False
        MeshPool.__remove_group(mesh, face_groups, face_a_index)
        mesh.remove_face(face_a_index)
        mesh.faces_count -= 1

    def __build_queue(self, features, faces_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        face_ids = torch.arange(faces_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, face_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    def __is_one_ring_valid(self, mesh, face_id, mask):
        f_a = mesh.vf[mesh.faces[face_id, 0]]
        f_b = mesh.vf[mesh.faces[face_id, 1]]
        f_c = mesh.vf[mesh.faces[face_id, 2]]
        for f in f_a:
            if not mask[f]:
                f_a.remove(f)
        for f in f_b:
            if not mask[f]:
                f_b.remove(f)
        for f in f_c:
            if not mask[f]:
                f_c.remove(f)
        return len(np.intersect1d(f_a, f_b)) == len(np.intersect1d(f_c, f_b)) == len(np.intersect1d(f_a, f_c)) == 2


    def __is_link_condition_valid(self, mesh, face_id, mask):
        """
        For the generation network, it's necessary that the output of pooling is manifold,
        so we check the link condition.
        however, this function will make the program to be too slow.
        Thus, for the segmentation task, we can use the function 'is one ring valid' which is initial code of MeshCNN
        """
        face = mesh.faces[face_id]
        f_a = mesh.vf[mesh.faces[face_id, 0]]
        f_b = mesh.vf[mesh.faces[face_id, 1]]
        f_c = mesh.vf[mesh.faces[face_id, 2]]
        for f in f_a:
            if not mask[f]:
                f_a.remove(f)
        for f in f_b:
            if not mask[f]:
                f_b.remove(f)
        for f in f_c:
            if not mask[f]:
                f_c.remove(f)
        v_a = set(mesh.faces[f_a].flatten())
        v_b = set(mesh.faces[f_b].flatten())
        v_c = set(mesh.faces[f_c].flatten())
        e_ab = set(mesh.faces[face_id, 0:2])
        e_bc = set(mesh.faces[face_id, 1:3])
        e_ca = set(mesh.faces[face_id, [0, 2]])
        if not len(v_a.intersection(v_b).difference(e_ab)) == len(v_b.intersection(v_c).difference(e_bc)) == \
                len(v_c.intersection(v_a).difference(e_ca)) == 2:
            return False
        else:
            return self.check_link(mesh, v_a, v_b, e_ab) and self.check_link(mesh, v_b, v_c, e_bc) and self.check_link(mesh, v_c, v_a, e_ca)

    @staticmethod
    def is_boundaries(mesh, face_id):
        for face in mesh.gemm_faces[face_id]:
            if face == -1 or -1 in mesh.gemm_faces[face]:
                return True
        return False

    @staticmethod
    def check_link(mesh, v_a, v_b, e_ab):
        two_points = np.array(list(v_a.intersection(v_b).difference(e_ab)))
        for face in mesh.faces:
            if len(np.intersect1d(face, two_points)) == 2:
                return False
        return True

    @staticmethod
    def __redirect_side_faces(mesh, face_a_id, face_ids_a):
        for _id, face in enumerate(mesh.gemm_faces[face_ids_a[0]]):
            if face == face_a_id:
                mesh.gemm_faces[face_ids_a[0], _id] = face_ids_a[1]
        for _id, face in enumerate(mesh.gemm_faces[face_ids_a[1]]):
            if face == face_a_id:
                mesh.gemm_faces[face_ids_a[1], _id] = face_ids_a[0]

    @staticmethod
    def __get_invalids(mesh, face_id, face_groups):
        # done
        neighbors = mesh.gemm_faces[face_id]
        other_faces = []
        for face in neighbors:
            other_faces.append(mesh.gemm_faces[face])
        other_faces = np.array(other_faces)
        other_faces2 = np.unique(other_faces.reshape(other_faces.size, 1))
        shared_items = np.intersect1d(neighbors, other_faces2)
        if len(shared_items) == 0:
            return []
        elif len(shared_items) > 2:
            return [-1, -1, -1, -1]
        else:
            for i in shared_items:
                MeshPool.__redirect_faces(mesh, i, face_id, shared_items)
                MeshPool.__union_groups(mesh, face_groups, i, face_id)
            return [shared_items[0], shared_items[1], face_id]

    @staticmethod
    def __redirect_faces(mesh, face_a_id, face_id, shared_items):
        # done
        for index, face in enumerate(mesh.gemm_faces[face_id]):
            if face == face_a_id:
                other_side_a_ids = mesh.gemm_faces[face]
                other_side_a_ids = np.setdiff1d(other_side_a_ids, shared_items)
                other_side_a_id = other_side_a_ids[0] if other_side_a_ids[1] == face_id else other_side_a_ids[1]
                for index2, face2 in enumerate(mesh.gemm_faces[other_side_a_id]):
                    if face2 ==face_a_id:
                        mesh.gemm_faces[other_side_a_id, index2] = face_id
                mesh.gemm_faces[face_id, index] = other_side_a_id

    @staticmethod
    def __remove_invalid_faces(mesh, mask, face_groups, invalid_faces):
        # done
        vertex = set(mesh.faces[invalid_faces[0]])
        for face_key in invalid_faces[0:2]:
            vertex &= set(mesh.faces[face_key])
            mask[face_key] = False
            MeshPool.__remove_group(mesh, face_groups, face_key)
            mesh.remove_face(face_key)
        mesh.faces_count -= 2
        vertex1 = vertex & set(mesh.faces[invalid_faces[2]])
        vertex2 = list(vertex - vertex1)
        vertex1 = list(vertex1)
        assert (len(vertex1) == 1)
        assert (len(vertex2) == 1)
        mesh.faces[invalid_faces[2]][mesh.faces[invalid_faces[2]] == vertex1[0]] = vertex2[0]
        mesh.vf[vertex2[0]].append(invalid_faces[2])
        mesh.remove_vertex(vertex1[0])

    @staticmethod
    def __union_groups(mesh, face_groups, source, target):
        face_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, face_groups, index):
        face_groups.remove_group(index)
        mesh.remove_group(index)

