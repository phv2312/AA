import numpy as np
import time
from lib.md import drawMesh
from lib.md import cls
from utils import image_utils as im_utils
from lib.interfaces import Mesh


def find_neighbor(edge, faces):
    # find the four neighbouring vertices of the edge
    neighbours = [np.nan, np.nan]
    count = 0
    for i, face in enumerate(faces):
        if np.any(face == edge[0]) and np.any(face == edge[1]):
            neighbor_idx = np.where(face[np.where(face != edge[0])] != edge[1])
            neighbor_idx = neighbor_idx[0][0]
            n = face[np.where(face != edge[0])]
            neighbours[count] = int(n[neighbor_idx])
            count += 1

    l, r = neighbours
    return l, r


def main():
    obj_path = "./sample_data/man.obj"
    selected_path = "./sample_data/selected.npy"
    locations_path = "./sample_data/locations.npy"

    #
    constraint_v_ids = np.load(selected_path)
    constraint_v_coords = np.load(locations_path)
    no_vertices, no_faces, vertices, faces = drawMesh.read_file(open(obj_path, 'r'))
    edges = drawMesh.get_edges(no_faces, faces)

    # get neighbor for each edge
    # conduct edge list
    edge_list = []
    for k, edge in enumerate(edges):
        #
        i, j = int(edge[0]), int(edge[1])
        l, r = find_neighbor(edge, faces)

        edge = cls.Edge(i - 1, j - 1, l - 1, (r - 1) if not np.isnan(r) else None)
        edge.calculate_g(vertices)
        edge.calculate_h(vertices)

        edge_list += [edge]

    #
    solver = cls.Solver(edge_list, constraint_v_ids, constraint_v_coords, vertices, w=100.)
    vertices_step1 = solver.step_one()
    vertices_step2 = solver.step_two(vertices_step1)
    # print('new_v:', vertices_step2)

    # visualize the result
    print('-- after')
    img = np.zeros((800, 1280, 3), np.uint8)
    drawMesh.draw_mesh(vertices_step2, edges, img)
    im_utils.imshow(img)


class ARAPDeformation:
    def load_from_vertices_faces(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def load_from_obj_path(self, obj_path):
        _, no_faces, self.vertices, self.faces = drawMesh.read_file(open(obj_path, 'r'))

    def load_from_mesh(self, mesh):
        self.vertices = mesh.vertices
        self.faces = mesh.faces

    def setup(self):
        assert self.vertices is not None, 'vertices have not been setup correctly.'
        assert self.faces is not None, 'faces have not been setup correctly.'
        no_faces = len(self.faces)
        edges = drawMesh.get_edges(no_faces, self.faces)

        # get neighbor for each edge
        # conduct edge list
        edge_list = []

        for k, edge in enumerate(edges):
            #
            i, j = int(edge[0]), int(edge[1])
            l, r = find_neighbor(edge, self.faces)

            edge = cls.Edge(i - 1, j - 1, l - 1, (r - 1) if not np.isnan(r) else None)
            edge.calculate_g(self.vertices)
            edge.calculate_h(self.vertices)

            edge_list += [edge]
        self.edge_list = edge_list

    def __init__(self):
        self.vertices = None
        self.faces = None
        self.edge_list = None

    def deform(self, constraint_v_ids, constraint_v_coords, w=1000.):
        """

        :param constraint_v_ids:
        :param constraint_v_coords:
        :param w:
        :return:
        """
        solver = cls.Solver(self.edge_list, constraint_v_ids, constraint_v_coords, self.vertices, w)
        vertices_step1 = solver.step_one()
        vertices_step2 = solver.step_two(vertices_step1)

        m = Mesh(vertices_step2, self.faces, texture_image=None)
        return m


if __name__ == '__main__':
    print('start deforming ...')
    main()


