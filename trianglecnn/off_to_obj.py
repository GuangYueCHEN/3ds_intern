import os
from models.layers.mesh import Mesh
from models.layers.mesh_prepare import fill_mesh
from data.segmentation_data import read_seg
from options.test_options import TestOptions
import numpy as np
def color_obj():
    """coloring mesh with the given seg file"""
    path = "E:/3ds_intern/meshcnn/datasets/coseg_aliens/labels_aliens/5.obj"
    path2 = "E:/3ds_intern/meshcnn/datasets/coseg_aliens/labels_aliens/5.eseg"
    opt = TestOptions().parse()
    mesh = Mesh(file=path, opt=opt, hold_history=False, export_folder="E:/3ds_intern/meshcnn/datasets/coseg_aliens/labels_aliens")
    seg = read_seg(path2).astype(int)
    mesh.export_segments(seg)



def off_to_obj():
    """transform the mesh files from off to obj"""
    path = "E:/3ds_intern/trianglecnn/datasets/ModelNet40"
    files= os.listdir(path)
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            filename, file_extension = os.path.splitext(name)
            if file_extension == '.off':
                write_file = os.path.join(root, filename + ".obj")
                load_file = os.path.join(root, name)
                first = True
                with open(write_file, 'w+') as new_file:
                    with open(load_file) as old_file:
                        for line in old_file:
                            line = line.strip()
                            splitted_line = line.split()
                            if not splitted_line:
                                continue
                            if len(splitted_line) == 3 and not first:
                                new_file.write("v %s\n" % line)
                            elif len(splitted_line) == 4:
                                new_file.write("f %d %d %d\n" % (int(splitted_line[1])+1, int(splitted_line[2])+1, int(splitted_line[3])+1))
                            elif len(splitted_line) == 3 and first:
                                first = False
                #os.remove(file)


if __name__ == '__main__':
    color_obj()