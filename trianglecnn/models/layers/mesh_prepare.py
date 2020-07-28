import numpy as np
import os
import ntpath
import heapq
TOLERENCE = 1.e-6

def fill_mesh(mesh2fill, file: str, opt):
    load_path = get_mesh_path(file, opt.num_aug)
    if os.path.exists(load_path):
        mesh_data = np.load(load_path, encoding='latin1', allow_pickle=True)
    else:
        mesh_data = from_scratch(file, opt)
        ''' gemm_edges=mesh_data.gemm_edges,'''
        np.savez_compressed(load_path, gemm_faces =mesh_data.gemm_faces, vs=mesh_data.vs, edges=mesh_data.edges,faces = mesh_data.faces,
                            edges_count=mesh_data.edges_count, faces_count=mesh_data.faces_count, ve=mesh_data.ve, vf=mesh_data.vf, v_mask=mesh_data.v_mask, faces_edges =mesh_data.faces_edges, edge_faces= mesh_data.edge_faces,
                            filename=mesh_data.filename,
                            edge_lengths=mesh_data.edge_lengths, areas=mesh_data.areas,
                            features=mesh_data.features)
    mesh2fill.vs = mesh_data['vs']
    mesh2fill.edges = mesh_data['edges']
    mesh2fill.faces = mesh_data['faces']
    '''mesh2fill.gemm_edges = mesh_data['gemm_edges']'''
    mesh2fill.gemm_faces = mesh_data['gemm_faces']

    mesh2fill.edges_count = int(mesh_data['edges_count'])
    mesh2fill.faces_count = int(mesh_data['faces_count'])
    mesh2fill.ve = mesh_data['ve']
    mesh2fill.vf = mesh_data['vf']
    mesh2fill.v_mask = mesh_data['v_mask']
    mesh2fill.filename = str(mesh_data['filename'])
    mesh2fill.edge_lengths = mesh_data['edge_lengths']
    mesh2fill.faces_edges = mesh_data['faces_edges']
    mesh2fill.edge_faces = mesh_data['edge_faces']
    mesh2fill.areas = mesh_data['areas']
    mesh2fill.features = mesh_data['features']

def get_mesh_path(file: str, num_aug: int):
    filename, _ = os.path.splitext(file)
    dir_name = os.path.dirname(filename)
    prefix = os.path.basename(filename)
    load_dir = os.path.join(dir_name, 'cache')
    load_file = os.path.join(load_dir, '%s_%03d.npz' % (prefix, np.random.randint(0, num_aug)))
    if not os.path.isdir(load_dir):
        os.makedirs(load_dir, exist_ok=True)
    return load_file

def from_scratch(file, opt):

    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = mesh_data.faces = None
    mesh_data.gemm_faces = None
    ''' mesh_data.gemm_edges = None'''
    mesh_data.edges_count = None
    mesh_data.faces_count = None
    mesh_data.ve = None
    mesh_data.vf = None
    mesh_data.v_mask = None
    mesh_data.faces = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.areas = None
    mesh_data.edge_faces = None
    mesh_data.faces_edges = None

    mesh_data.vs, faces = fill_from_file(mesh_data, file)

    faces, areas = remove_non_manifolds(mesh_data, faces)
    if opt.num_aug > 1:
        faces, areas = augmentation(mesh_data, opt, faces, areas)
    mesh_data.v_mask = np.ones(len(mesh_data.vs), dtype=bool)
    _, edge_faces, edges_dict = get_edge_faces(faces)
    build_gemm(mesh_data, faces, areas, edge_faces)
    if opt.num_aug > 1:
        post_augmentation(mesh_data, opt)
    set_edge_lengths(mesh_data)
    mesh_data.features = extract_features(mesh_data)
    return mesh_data


def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def remove_non_manifolds(mesh, faces):
    edges_set = set()
    mask = np.ones(len(faces), dtype=bool)
    _, face_areas = compute_face_normals_and_areas(mesh, faces)
    for face_id, face in enumerate(faces):
        if face_areas[face_id] == 0:
            mask[face_id] = False
            continue
        face_edges = []
        is_manifold = False
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            if cur_edge in edges_set:
                is_manifold = True
                break
            else:
                face_edges.append(cur_edge)
        if is_manifold:
            mask[face_id] = False
        else:
            for idx, edge in enumerate(face_edges):
                edges_set.add(edge)
    return faces[:], face_areas[:]


def build_gemm(mesh, faces, face_areas, edge_faces):
    """
    gemm_faces: array (#F x 3) of the 3 one-ring neighbors for each face
    """
    mesh.ve = [[] for _ in mesh.vs]
    mesh.vf = [[] for _ in mesh.vs]
    face_nb = []
    '''edge_nb = []'''
    edge2key = dict()
    edges = []
    edges_count = 0
    faces_count = 0
    nb_count = []
    faces_edges = []
    for face_id, face in enumerate(faces):
        faces_count += 1
        face_edges = []
        face_nb.append([-1, -1, -1])
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            face_edges.append(cur_edge)
            mesh.vf[face[i]].append(face_id)
        for idx, edge in enumerate(face_edges):
            edge = tuple(sorted(list(edge)))
            face_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                '''edge_nb.append([-1, -1, -1, -1])'''
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                nb_count.append(0)
                edges_count += 1
        face_eid = []
        for idx, edge in enumerate(face_edges):
            edge_key = edge2key[edge]
            face_eid.append(edge_key)
            face_nb[face_id][idx] = edge_faces[edge_key][3 if edge_faces[edge_key][2] == face_id else 2]
        faces_edges.append(face_eid)
        '''for idx, edge in enumerate(face_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]] = edge2key[face_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[face_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2'''

    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_faces = np.array(face_nb, dtype=np.int64)
    '''mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)'''
    mesh.edges_count = edges_count
    mesh.faces_count = faces_count
    mesh.faces = np.array(faces, dtype=np.int32)
    mesh.edge_faces = np.array(edge_faces, dtype=np.int32)
    mesh.faces_edges = np.array(faces_edges, dtype=np.int32)
    mesh.areas = np.array(face_areas, dtype=np.float32) / np.sum(face_areas)
    '''export_obj(mesh, file="./datasets/test_curvature/%s" % (mesh.filename))'''



def compute_face_normals_and_areas(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_normals /= face_areas[:, np.newaxis]
    assert (not np.any(face_areas[:, np.newaxis] == 0)), 'has zero area face: %s' % mesh.filename
    face_areas *= 0.5
    return face_normals, face_areas


# Data augmentation methods
def augmentation(mesh, opt, faces=None, areas = None):
    if hasattr(opt, 'scale_verts') and opt.scale_verts:
        scale_verts(mesh)
    if hasattr(opt, 'flip_edges') and opt.flip_edges:
        faces, areas = flip_edges(mesh, opt.flip_edges, faces, areas, opt.dataset_mode, opt.dataroot, opt.aug_triangulation)
    if hasattr(opt, 'aug_triangulation') and opt.aug_triangulation:
        faces, areas = aug_triangulation(mesh, opt.aug_triangulation, faces, areas, opt.dataroot)

    return faces, areas


def post_augmentation(mesh, opt):
    if hasattr(opt, 'slide_verts') and opt.slide_verts:
        slide_verts(mesh, opt.slide_verts)

def aug_triangulation(mesh, num, faces, areas, dataroot):
    count = 0
    seg_file = os.path.join(dataroot, 'seg/' + os.path.splitext(mesh.filename)[0] + '.eseg')
    sseg_file = os.path.join(dataroot, 'sseg/' + os.path.splitext(mesh.filename)[0] + '.seseg')
    assert (os.path.isfile(seg_file))
    seg_labels = read_seg(seg_file)
    sseg_labels = read_sseg(sseg_file)
    while count < num:

        tt = np.concatenate((areas.reshape(len(areas), 1),
                             np.arange(len(areas)).reshape(len(areas), 1)), axis=1).tolist()
        heapq._heapify_max(tt)
        while len(tt) > 0:
            if count >= num:
                break
            area, id = heapq._heappop_max(tt)
            id = int(id)
            new_point = np.mean(mesh.vs[faces[id]], axis=0).reshape(1, 3)
            mesh.vs = np.concatenate((mesh.vs, new_point), axis=0)
            v_id = len(mesh.vs) - 1
            v_2 = faces[id, 2]
            faces[id] = [faces[id, 0], faces[id, 1], v_id]
            new_faces = []
            new_faces.append([faces[id, 1], v_2, faces[id, 2]])
            new_faces.append([faces[id, 2], v_2, faces[id, 0]])
            faces =np.concatenate((faces, np.array(new_faces, dtype= int)), axis =0)
            areas[id] = area / 3
            new_areas = [area, area]
            areas = np.concatenate((areas, np.array(new_areas, dtype=int)), axis=0)
            seg_labels = np.append(seg_labels, seg_labels[id])
            seg_labels = np.append(seg_labels, seg_labels[id])
            sseg_labels = np.append(sseg_labels, sseg_labels[id].reshape(1, sseg_labels.shape[1]), axis=0)
            sseg_labels = np.append(sseg_labels, sseg_labels[id].reshape(1, sseg_labels.shape[1]), axis=0)
            count += 2
    write_seg(seg_labels, seg_file);
    write_sseg(sseg_labels, sseg_file);
    return faces, areas


def slide_verts(mesh, prct):
    set_edge_lengths(mesh)
    curvatures, min_curvature_vectors = curvature_of_vs(mesh)
    curvatures_min = curvatures[:, 0]
    curvatures_max = curvatures[:, 1]
    vids = np.random.permutation(len(mesh.ve))
    target = int(prct * len(vids))
    shifted = 0
    for vi in vids:
        if shifted < target:
            main1 = curvatures_min[vi]
            main2 = curvatures_max[vi]
            edges = mesh.ve[vi]
            if main1 < 0.01 and main2 < 0.01:
                edge = mesh.edges[np.random.choice(edges)]
                vi_t = edge[1] if vi == edge[0] else edge[0]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5) * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
            elif main1 < 0.01:
                vi_ts = []
                norms = []
                for edge in mesh.edges[edges]:
                    vi_t = edge[1] if vi == edge[0] else edge[0]
                    vector = min_curvature_vectors[vi, :]
                    norms.append(np.linalg.norm(np.cross(mesh.vs[vi_t] - mesh.vs[vi], vector)))
                    vi_ts.append(vi_t)
                vi_t = vi_ts[np.argmin(norms)]
                nv = mesh.vs[vi] + np.random.uniform(0.2, 0.5)  * (mesh.vs[vi_t] - mesh.vs[vi])
                mesh.vs[vi] = nv
                shifted += 1
        else:
            break
    mesh.shifted = shifted / len(mesh.ve)
'''    print(mesh.filename)
    print(shifted)
    export_obj(mesh, file="./datasets/test_curvature/%s.obj" % (mesh.filename))'''

def export_obj(mesh, file=None, vcolor=None):

        vs = mesh.vs[mesh.v_mask]

        with open(file, 'w+') as f:
            for vi, v in enumerate(vs):
                vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
                f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
            for face_id in range(len(mesh.faces) - 1):
                v1 = np.size(mesh.v_mask[0:mesh.faces[face_id][0]]) - np.sum(mesh.v_mask[0:mesh.faces[face_id][0]])
                v2 = np.size(mesh.v_mask[0:mesh.faces[face_id][1]]) - np.sum(mesh.v_mask[0:mesh.faces[face_id][1]])
                v3 = np.size(mesh.v_mask[0:mesh.faces[face_id][2]]) - np.sum(mesh.v_mask[0:mesh.faces[face_id][2]])
                f.write("f %d %d %d\n" % (
                    mesh.faces[face_id][0] - v1 + 1, mesh.faces[face_id][1] - v2 + 1, mesh.faces[face_id][2] - v3 + 1))
            v1 = np.size(mesh.v_mask[0:mesh.faces[-1][0]]) - np.sum(mesh.v_mask[0:mesh.faces[-1][0]])
            v2 = np.size(mesh.v_mask[0:mesh.faces[-1][1]]) - np.sum(mesh.v_mask[0:mesh.faces[-1][1]])
            v3 = np.size(mesh.v_mask[0:mesh.faces[-1][2]]) - np.sum(mesh.v_mask[0:mesh.faces[-1][2]])
            f.write("f %d %d %d" % (mesh.faces[-1][0] - v1 + 1, mesh.faces[-1][1] - v2 + 1, mesh.faces[-1][2] - v3 + 1))


def scale_verts(mesh, mean=1, var=0.1):
    for i in range(mesh.vs.shape[1]):
        mesh.vs[:, i] = mesh.vs[:, i] * np.random.normal(mean, var)


def angles_from_faces(mesh, edge_faces, faces):
    normals = [None, None]
    for i in range(2):
        edge_a = mesh.vs[faces[edge_faces[:, i], 2]] - mesh.vs[faces[edge_faces[:, i], 1]]
        edge_b = mesh.vs[faces[edge_faces[:, i], 1]] - mesh.vs[faces[edge_faces[:, i], 0]]
        normals[i] = np.cross(edge_a, edge_b)
        div = fixed_division(np.linalg.norm(normals[i], ord=2, axis=1), epsilon=0)
        normals[i] /= div[:, np.newaxis]
    dot = np.sum(normals[0] * normals[1], axis=1).clip(-1, 1)
    angles = np.pi - np.arccos(dot)
    return angles

def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype='float64')
    return sseg_labels


def write_seg(labels, seg):
    dir_name = os.path.join(os.path.dirname(seg), 'cache')
    prefix = os.path.basename(seg)
    write_file = os.path.join(dir_name, prefix)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if os.path.isfile(write_file):
        return
    np.savetxt(write_file, labels, delimiter='\n', fmt='%.1e')



def write_sseg(labels, sseg_file):
    sum = np.sum(labels, axis=1).reshape(labels.shape[0], 1).repeat(labels.shape[1], axis= 1)*2
    dir_name = os.path.join(os.path.dirname(sseg_file), 'cache')
    prefix = os.path.basename(sseg_file)
    write_file = os.path.join(dir_name, prefix)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    if os.path.isfile(write_file):
        return
    np.savetxt(write_file, labels/sum, delimiter=' ', newline='\n', fmt='%.2e')


def flip_edges(mesh, prct, faces, areas, mode, dataroot, aug = None):
    edge_count, edge_faces, edges_dict = get_edge_faces(faces)
    dihedral = angles_from_faces(mesh, edge_faces[:, 2:], faces)
    edges2flip = np.random.permutation(edge_count)
    # print(dihedral.min())
    # print(dihedral.max())
    target = int(prct * edge_count)
    flipped = 0
    if mode == 'segmentation':
        if aug:
            seg_file = os.path.join(dataroot, 'seg/cache/' + os.path.splitext(mesh.filename)[0] + '.eseg')
        else:
            seg_file = os.path.join(dataroot, 'seg/' + os.path.splitext(mesh.filename)[0] + '.eseg')
        assert (os.path.isfile(seg_file))
        seg_labels = read_seg(seg_file)
    for edge_key in edges2flip:
        if flipped == target:
            break
        if dihedral[edge_key] > 2.8:
            edge_info = edge_faces[edge_key]
            if edge_info[3] == -1:
                continue
            if mode == 'segmentation':
                two_faces = edge_faces[edge_key, 2:4]
                if seg_labels[two_faces[0]] != seg_labels[two_faces[1]]:
                    continue
            new_edge = tuple(sorted(list(set(faces[edge_info[2]]) ^ set(faces[edge_info[3]]))))
            if new_edge in edges_dict:
                continue
            new_faces = np.array(
                [[edge_info[1], new_edge[0], new_edge[1]], [edge_info[0], new_edge[0], new_edge[1]]])
            if check_area(mesh, new_faces):
                del edges_dict[(edge_info[0], edge_info[1])]
                edge_info[:2] = [new_edge[0], new_edge[1]]
                edges_dict[new_edge] = edge_key
                rebuild_face(faces[edge_info[2]], new_faces[0])
                rebuild_face(faces[edge_info[3]], new_faces[1])
                area = areas[edge_info[2]] + areas[edge_info[3]] / 2.
                areas[edge_info[2]] = areas[edge_info[3]] = area
                for i, face_id in enumerate([edge_info[2], edge_info[3]]):
                    cur_face = faces[face_id]
                    for j in range(3):
                        cur_edge = tuple(sorted((cur_face[j], cur_face[(j + 1) % 3])))
                        if cur_edge != new_edge:
                            cur_edge_key = edges_dict[cur_edge]
                            for idx, face_nb in enumerate(
                                    [edge_faces[cur_edge_key, 2], edge_faces[cur_edge_key, 3]]):
                                if face_nb == edge_info[2 + (i + 1) % 2]:
                                    edge_faces[cur_edge_key, 2 + idx] = face_id
                flipped += 1
    '''print(flipped)'''
    return faces, areas


def rebuild_face(face, new_face):
    new_point = list(set(new_face) - set(face))[0]
    for i in range(3):
        if face[i] not in new_face:
            face[i] = new_point
            break
    return face

def check_area(mesh, faces):
    face_normals = np.cross(mesh.vs[faces[:, 1]] - mesh.vs[faces[:, 0]],
                            mesh.vs[faces[:, 2]] - mesh.vs[faces[:, 1]])
    face_areas = np.sqrt((face_normals ** 2).sum(axis=1))
    face_areas *= 0.5
    return face_areas[0] > 0 and face_areas[1] > 0


def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys


def set_edge_lengths(mesh):
    edge_lengths = np.linalg.norm(mesh.vs[mesh.edges[:, 0]] - mesh.vs[mesh.edges[:, 1]], ord=2, axis=1)
    mesh.edge_lengths = edge_lengths

def get_face_elengths(mesh):
    edges_lengths = []
    for face_id, face in mesh.faces:
        edge_lengths = []
        edge_ids = mesh.faces_edges[face_id]
        for edge_id in edge_ids:
            edge_lengths.append(mesh.edge_lengths[edge_id])
        edges_lengths.append(edge_lengths)
    return edges_lengths

def extract_features(mesh):
    features = []
    # done
    with np.errstate(divide='raise'):
        try:
            for extractor in [symmetric_ratios, area_ratios, symmetric_opposite_angles, normal, dihedral_angle]:
                feature = extractor(mesh)
                features.append(feature)
            features_curvature = curvature(mesh)
            for feature in features_curvature:
                features.append(feature)
            return np.concatenate(features, axis=0)
        except Exception as e:
            print(e)
            raise ValueError(mesh.filename, 'bad features')


def dihedral_angle(mesh):
    angles = []
    for i in range(3):
        normals_a, normals_b = get_normals(mesh, i)
        dot = np.sum(normals_a * normals_b, axis=1)
        angles_i = np.pi - np.arccos(dot)
        angles.append(angles_i)
    angles = np.array(angles)
    return np.sort(angles, axis=0)


def normal(mesh):
    normals_a, _ = get_normals(mesh, 0)
    return normals_a.T

def symmetric_opposite_angles(mesh):
    """ computes two angles: one for each face shared between the edge
        the angle is in each face opposite the edge
        sort handles order ambiguity
    """
    angles = []
    for i in range(3):
        angles_i = get_opposite_angles(mesh, i)
        angles.append(angles_i)
    angles = np.array(angles)
    return np.sort(angles, axis=0)


def symmetric_ratios(mesh):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios = []
    for i in range(3):
        ratios_i = get_ratios(mesh, i)
        ratios.append(ratios_i)
    ratios = np.array(ratios)
    return np.sort(ratios, axis=0)

def area_ratios(mesh):
    """ computes two ratios: one for each face shared between the edge
        the ratio is between the height / base (edge) of each triangle
        sort handles order ambiguity
    """
    ratios = []
    for i in range(3):
        ratios_i = get_area_ratios(mesh, i)
        ratios.append(ratios_i)
    ratios = np.array(ratios)
    return np.sort(ratios, axis=0)
'''
def get_edge_points(mesh):
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int32)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
        # edge_points[edge_id, 3:] = mesh.get_side_points(edge_id, 2)
    return edge_points


def get_side_points(mesh, edge_id):
    # if mesh.gemm_edges[edge_id, side] == -1:
    #     return mesh.get_side_points(edge_id, ((side + 2) % 4))
    # else:
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]
'''

def get_normals(mesh,  index):
    face_neighbor_id = mesh.gemm_faces[:, index]
    v_neighbor_ids = []
    for face_id, face in enumerate(mesh.faces):
        v_neighbor_id = np.setdiff1d(mesh.faces[face_neighbor_id[face_id]], mesh.faces[face_id])[0]
        v_neighbor_ids.append(v_neighbor_id)
    edge_a = mesh.vs[mesh.faces[:, (index + 2) % 3]] - mesh.vs[mesh.faces[:, (index + 1) % 3]]
    edge_b = mesh.vs[mesh.faces[:, index]] - mesh.vs[mesh.faces[:, (index + 1) % 3]]
    edge_b_inverse = mesh.vs[mesh.faces[:, (index + 1) % 3]] - mesh.vs[mesh.faces[:, index]]
    edge_c = mesh.vs[v_neighbor_ids] - mesh.vs[mesh.faces[:, index]]
    normals = np.cross(edge_a, edge_b)
    normals_neighbor = np.cross(edge_c, edge_b_inverse)
    div = fixed_division(np.linalg.norm(normals, ord=2, axis=1), epsilon=1.e-6)
    div_ = fixed_division(np.linalg.norm(normals_neighbor, ord=2, axis=1), epsilon=1.e-6)
    normals /= div[:, np.newaxis]
    normals_neighbor /= div_[:, np.newaxis]
    return normals, normals_neighbor

def get_opposite_angles(mesh, index):
    edges_a = mesh.vs[mesh.faces[:, index]] - mesh.vs[mesh.faces[:, (index+2) % 3]]
    edges_b = mesh.vs[mesh.faces[:, (index+1) % 3]] - mesh.vs[mesh.faces[:, (index+2) % 3]]

    edges_a /= fixed_division(np.linalg.norm(edges_a, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    edges_b /= fixed_division(np.linalg.norm(edges_b, ord=2, axis=1), epsilon=0.1)[:, np.newaxis]
    dot = np.sum(edges_a * edges_b, axis=1).clip(-1, 1)
    return np.arccos(dot)


def get_ratios(mesh, index):
    edges_id = mesh.faces_edges[:, index]
    edges_lengths = mesh.edge_lengths[edges_id]
    point_o = mesh.vs[mesh.faces[:, (index+2) % 3]]
    point_a = mesh.vs[mesh.faces[:, index]]
    point_b = mesh.vs[mesh.faces[:, (index+1) % 3]]
    line_ab = point_b - point_a
    projection_length = np.sum(line_ab * (point_o - point_a), axis=1) / fixed_division(
        np.linalg.norm(line_ab, ord=2, axis=1), epsilon=0.1)
    closest_point = point_a + (projection_length / edges_lengths)[:, np.newaxis] * line_ab
    d = np.linalg.norm(point_o - closest_point, ord=2, axis=1)
    return d / edges_lengths

def get_area_ratios(mesh, index):
    face_neighbor_id = mesh.gemm_faces[:, index]
    return mesh.areas[face_neighbor_id] / mesh.areas

def curvature(mesh):
    """
    """
    curvatures_main1 = []
    curvatures_main2 = []
    curvatures_gauss = []
    curvatures_mean = []

    curvatures, _ = curvature_of_vs(mesh)
    for i in range(3):
        curvature_i = get_curvature(mesh, i, curvatures)
        curvatures_main1.append(curvature_i[:, 0])
        curvatures_main2.append(curvature_i[:, 1])
        curvatures_gauss.append(curvature_i[:, 2])
        curvatures_mean.append(curvature_i[:, 3])
    curvatures_main1 = np.array(curvatures_main1)
    curvatures_main2 = np.array(curvatures_main2)
    curvatures_gauss = np.array(curvatures_gauss)
    curvatures_mean = np.array(curvatures_mean)
    return np.sort(curvatures_main1, axis=0), np.sort(curvatures_main2, axis=0), np.sort(curvatures_gauss, axis=0),\
        np.sort(curvatures_mean, axis=0)

def get_curvature(mesh, index, curvatures):
    vertices_ids = mesh.faces[:, index]
    curvatures_i = curvatures[vertices_ids, :]
    return np.array(curvatures_i)


'''
def curvature_of_vs(mesh):
    curvatures = []
    for vertex_id, edge_one_ring in enumerate(mesh.ve):
        normals_a, normals_b = get_normals_by_edge(mesh, edge_one_ring)
        normals_edge = (normals_a + normals_b) / np.linalg.norm(normals_a + normals_b, ord=2, axis=1).reshape(
            len(normals_a), 1)
        normals_vertex = np.sum(normals_edge/mesh.edge_lengths[edge_one_ring].reshape(
            len(normals_edge), 1), axis=0)
        normals_vertex /=  np.linalg.norm(normals_vertex, ord=2)
        dot = np.sum(normals_a/ np.linalg.norm(normals_a, ord=2, axis=1).reshape(
            len(normals_a), 1) * normals_b/ np.linalg.norm(normals_b, ord=2, axis=1).reshape(
            len(normals_b), 1), axis=1)
        angles_i = np.pi - np.arccos(dot)
        shape_operators = np.zeros((3, 3), dtype=float)
        for edge_index, edge in enumerate(edge_one_ring):
            edge_vector = mesh.vs[mesh.edges[edge][0]] - mesh.vs[vertex_id] if mesh.edges[edge][1] == vertex_id else \
            mesh.vs[mesh.edges[edge][1]] - mesh.vs[vertex_id]
            cross = np.cross(edge_vector / np.linalg.norm(edge_vector), normals_edge[edge_index]).reshape((3,1))
            he = 2 / np.linalg.norm(edge_vector) * np.cos(angles_i[edge_index] / 2)
            shape_operator_edge = he * cross.T * cross
            shape_operators += np.dot(normals_edge[edge_index], normals_vertex) * shape_operator_edge
        eigenvalues, _ = np.linalg.eig(shape_operators/2)
        eigenvalues = sorted(eigenvalues, key=abs)
        curvatures.append([eigenvalues[-1], eigenvalues[-2], eigenvalues[-1] * eigenvalues[-2],
                             (eigenvalues[-1] + eigenvalues[-2]) / 2.])
        print(curvatures[-1])
    return np.array(curvatures)

def get_normals_by_edge(mesh, edge_ids):
    face_ids = mesh.edge_faces[edge_ids]
    normals, _ = get_normals(mesh, 0)
    return normals[face_ids[:, 2]], normals[face_ids[:, 3]]
'''
def curvature_of_vs(mesh):
    normals_triangles, _ = get_normals(mesh, 0)
    v1s, v2s = complete_orthogonal_basis(normals_triangles)
    normals_vetices = get_normals_vertices(mesh, normals_triangles)
    vertices_v1s, vertices_v2s = complete_orthogonal_basis(normals_vetices)
    second_ff_vertices = np.zeros((len(mesh.vs), 2, 2), dtype=float)
    total_weight = np.zeros((len(mesh.vs), 1), dtype=float)
    weights = get_weight_by_triangle(mesh)
    for face_id, face in enumerate(mesh.faces):
        v1 = v1s[face_id]
        v2 = v2s[face_id]
        n_t = normals_triangles[face_id]
        mat = np.zeros((3, 3), dtype=float)
        b = np.zeros((3, 1), dtype=float)
        for i in range(3):
            edge_vector = mesh.vs[face[(i+1) % 3]] - mesh.vs[face[i]]
            nc_input_x = np.dot(edge_vector, v1)
            nc_input_y = np.dot(edge_vector, v2)
            tmp = normals_vetices[face[(i+1) % 3]] - normals_vetices[face[i]]
            nc_output_x = np.dot(tmp, v1)
            nc_output_y = np.dot(tmp, v2)
            mat[0, 0] += nc_input_x * nc_input_x
            mat[1, 0] += nc_input_x * nc_input_y
            b[0] += nc_input_x * nc_output_x
            mat[1, 1] += nc_input_x * nc_input_x + nc_input_y * nc_input_y
            b[1] += (nc_input_x * nc_output_y + nc_output_x * nc_input_y)
            mat[2, 2] += nc_input_y * nc_input_y
            b[2] += nc_input_y * nc_output_y
        mat[2, 1] = mat[0, 1] = mat[1, 2] = mat[1, 0]
        mat = np.linalg.inv(mat)
        tmp = np.dot(mat, b)
        second_ff = np.array([[tmp[0], tmp[1]], [tmp[1], tmp[2]]]).reshape(2, 2)
        for i in range(3):
            n_v = normals_vetices[face[i]]
            v_v1 = vertices_v1s[face[i]]
            v_v2 = vertices_v2s[face[i]]
            cross = np.cross(n_t, n_v)
            if np.linalg.norm(cross) < TOLERENCE:
                local_transfo = change_second_fundamental_from_frame(second_ff, v1, v2, v_v1, v_v2)
                second_ff_vertices[face[i]] += weights[face_id, i] * local_transfo
            else:
                mat_rotation = rotate_frame(n_t, n_v)
                new_v1 = np.dot(mat_rotation, v1)
                new_v2 = np.dot(mat_rotation, v2)
                local_transfo = change_second_fundamental_from_frame(second_ff, new_v1, new_v2, v_v1, v_v2)
                second_ff_vertices[face[i]] += weights[face_id, i] * local_transfo
            total_weight[face[i]] += weights[face_id, i]
    curvatures = []
    min_curvature_vectors = []
    for v_id, vertex in enumerate(mesh.vs):
        total_weight_v = total_weight[v_id]
        if total_weight_v == 0.:
            total_weight_v = 1.
        second_ff_vertices[v_id] = 1./total_weight_v * second_ff_vertices[v_id]
        eigenvalues, eigenvectors = np.linalg.eig(second_ff_vertices[v_id])
        if np.min(eigenvalues) == eigenvalues[0]:
            min_curvature_vector = np.array([eigenvectors[0, 0], eigenvectors[0, 1], 0.], dtype=float)
        else:
            min_curvature_vector = np.array([eigenvectors[1, 0], eigenvectors[1, 1], 0.], dtype=float)
        n_e = np.array([0., 0., 1.])
        mat_rotation = rotate_frame(n_e, n_v)
        min_curvature_vectors.append(np.dot(mat_rotation, min_curvature_vector))
        eigenvalues = sorted(eigenvalues, key=abs)
        curvatures_i = [eigenvalues[0], eigenvalues[1], eigenvalues[0] * eigenvalues[1],
                        (eigenvalues[0] + eigenvalues[1]) / 2.]
        curvatures.append(curvatures_i)
    curvatures = np.array(curvatures, dtype=float)
    return np.array(curvatures, dtype=float), np.array(min_curvature_vectors, dtype=float)

def get_weight_by_triangle(mesh):
    weights = []
    for face_id, face in enumerate(mesh.faces):
        res = [0., 0., 0.]
        if mesh.areas[face_id] < TOLERENCE:
            weights.append(res)
            continue
        obtuse_angle = [False, False, False]
        for i in range(3):
            e1 = mesh.vs[face[(i + 2) % 3]] - mesh.vs[face[i]]
            e2 = mesh.vs[face[(i + 1) % 3]] - mesh.vs[face[i]]
            if np.dot(e1, e2) < 0.:
                obtuse_angle[i] = True
        is_triangle_obtuse = obtuse_angle[0] | obtuse_angle[1] | obtuse_angle[2]
        for i in range(3):
            if not is_triangle_obtuse:
                a1_prev = mesh.vs[face[i]] - mesh.vs[face[(i + 1) % 3]]
                a1_next = mesh.vs[face[(i + 2) % 3]] - mesh.vs[face[(i + 1) % 3]]
                a2_prev = mesh.vs[face[(i + 1) % 3]] - mesh.vs[face[(i + 2) % 3]]
                a2_next = mesh.vs[face[i]] - mesh.vs[face[(i + 2) % 3]]
                e1 = mesh.vs[face[(i + 2) % 3]] - mesh.vs[face[i]]
                e2 = mesh.vs[face[(i + 1) % 3]] - mesh.vs[face[i]]
                test_cotan1 = abs(np.dot(a1_prev, a1_next) / np.linalg.norm(np.cross(a1_prev, a1_next)))
                test_cotan2 = abs(np.dot(a2_prev, a2_next) / np.linalg.norm(np.cross(a2_prev, a2_next)))
                res[i] = 1./8. * (np.linalg.norm(e2)**2 * test_cotan2 + np.linalg.norm(e1)**2 * test_cotan1)
            else:
                if obtuse_angle[i]:
                    res[i] = 0.5 * mesh.areas[face_id]
                else:
                    res[i] = 0.25 * mesh.areas[face_id]
        weights.append(res)
    return np.array(weights, dtype=float)

def change_second_fundamental_from_frame(second_ff, v1, v2, v_v1, v_v2):
    up = np.zeros((2, 1), dtype=float)
    vp = np.zeros((2, 1), dtype=float)
    res = np.zeros((2, 2), dtype=float)
    up[0] = np.dot(v1, v_v1)
    up[1] = np.dot(v2, v_v1)
    vp[0] = np.dot(v1, v_v2)
    vp[1] = np.dot(v2, v_v2)
    res[0, 0] = np.dot(np.dot(second_ff, up).T, up)
    res[0, 1] = res[1, 0] = np.dot(np.dot(second_ff, vp).T, up)
    res[1, 1] = np.dot(np.dot(second_ff, vp).T, vp)
    return res

def rotate_frame(normal_triangle, normal_vertex):
    axis = np.cross(normal_triangle, normal_vertex)
    cos = np.dot(normal_triangle, normal_vertex)
    sin = np.linalg.norm(axis)
    div = fixed_division(np.linalg.norm(axis, ord=2), epsilon=1.e-6)
    axis /= div
    res = np.zeros((3, 3), dtype=float)
    res[0, 0] = cos + axis[0] * axis[0] * (1. - cos)
    res[0, 1] = axis[0] * axis[1] * (1. - cos) + axis[2] * sin
    res[0, 2] = axis[2] * axis[1] * (1. - cos) - axis[1] * sin
    res[1, 0] = axis[0] * axis[1] * (1. - cos) - axis[2] * sin
    res[1, 1] = cos + axis[1] * axis[1] * (1. - cos)
    res[1, 2] = axis[2] * axis[1] * (1. - cos) + axis[0] * sin
    res[2, 0] = axis[0] * axis[2] * (1. - cos) + axis[1] * sin
    res[2, 1] = axis[2] * axis[1] * (1. - cos) - axis[0] * sin
    res[2, 2] = cos + axis[2] * axis[2] * (1. - cos)
    return res.T

def get_normals_vertices(mesh, normals_triangles):
    normals_vetices = []
    for vertex_id, edge_one_ring in enumerate(mesh.ve):
        face_ids = mesh.edge_faces[edge_one_ring]
        normals_a = normals_triangles[face_ids[:, 2]]
        normals_b = normals_triangles[face_ids[:, 3]]
        div = fixed_division(np.linalg.norm(normals_a + normals_b, ord=2, axis=1), epsilon=1.e-6)
        normals_edge = (normals_a + normals_b) / div[:, np.newaxis]
        normal_vertex = np.sum(normals_edge / mesh.edge_lengths[edge_one_ring].reshape(
            len(normals_edge), 1), axis=0)
        div = fixed_division(np.linalg.norm(normal_vertex, ord=2), epsilon=1.e-6)
        normal_vertex /= div
        normals_vetices.append(normal_vertex)
    return np.array(normals_vetices, dtype=np.float32)

def complete_orthogonal_basis(normals):
    basis_x = []
    basis_y = []
    for new_z in normals:
        new_x = np.array([1., 0., 0.])
        new_y = np.cross(new_z, new_x)
        if np.linalg.norm(new_y)**2 < TOLERENCE:
            new_x = np.array([0., 1., 0.])
            new_y = np.cross(new_z, new_x)
            if np.linalg.norm(new_y)**2 < TOLERENCE:
                new_x = np.array([0., 0., 1.])
                new_y = np.cross(new_z, new_x)
        div = fixed_division(np.linalg.norm(new_y, ord=2), epsilon=1.e-6)
        new_y /= div
        new_x = np.cross(new_y, new_z)
        div = fixed_division(np.linalg.norm(new_x, ord=2), epsilon=1.e-6)
        new_x /= div
        basis_x.append(new_x)
        basis_y.append(new_y)
    return np.array(basis_x, dtype=np.float32), np.array(basis_y, dtype=np.float32)

def fixed_division(to_div, epsilon):
    if epsilon == 0:
        to_div[to_div == 0] = 0.1
    else:
        to_div += epsilon
    return to_div
