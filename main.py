import cv2
import time
import numpy as np
from scipy.spatial.distance import cdist

from lib.interfaces import Mesh
from lib.mc.mc import TriangleMeshCreator
from lib.md.deform import ARAPDeformation
from utils import image_utils as im_utils


VISUALIZE = True


def augment_handle_points(poses2d, size):
    target_poses2d = poses2d.copy()
    target_poses2d[5] = [100, 150]

    return target_poses2d


def save_obj_format(file_path, vertices, faces):
    """
    Save obj wavefront to file
    :param file_path:
    :param vertices:
    :param faces:
    :return:
    """

    f = open(file_path, 'w')

    # number of vertices
    no_v = len(vertices)
    no_f = len(faces)

    f.write('#vertices: %d\n' % no_v)
    f.write('#faces: %d\n' % no_f)

    # vertices
    for i in range(no_v):
        v = vertices[i]
        f.write("v %.4f %.4f %d\n" % (v[0], v[1], 0))

    # triangle faces
    for t in faces:
        f.write("f")
        for i in t:
            f.write(" %d" % (i))
        f.write("\n")

    f.close()


def main():
    image_path = "./data/image_alok.png"
    mask_path = "./data/mask_alok.png"
    poses2d_path = "./data/poses2d.npy"

    #
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    mask = cv2.imread(mask_path, 0)
    poses2d = np.load(poses2d_path)

    #
    tri_mc = TriangleMeshCreator(interval=20, angle_constraint=20, area_constraint=200, dilated_pixel=5)
    mesh = tri_mc.create(image, mask)

    #
    vertices = 0.5 * (mesh.vertices + 1) * np.array([w, h]).reshape((1, 2)).astype(np.float32)
    distance = cdist(poses2d, vertices)
    constraint_v_ids = np.argmin(distance, axis=1)
    poses2d = vertices[constraint_v_ids]
    constraint_v_coords = augment_handle_points(poses2d, size=(w, h))

    constraint_v_ids = np.array([e for i, e in enumerate(constraint_v_ids) if i != 3])
    constraint_v_coords = np.array([e for i, e in enumerate(constraint_v_coords) if i != 3])

    if VISUALIZE:
        vis_image = mesh.get_image()
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        for x, y in poses2d.astype(np.int):
            cv2.circle(vis_image, (x, y), radius=3, color=(255, 0, 0), thickness=2)

        for x, y in constraint_v_coords.astype(np.int):
            cv2.circle(vis_image, (x, y), radius=3, color=(0, 255, 0), thickness=2)

        im_utils.imshow(vis_image)

    #
    constraint_v_coords = Mesh.normalize_vertices(constraint_v_coords, size=(w, h))

    #
    save_obj_format(file_path='./source_mesh.obj', vertices=mesh.vertices, faces=mesh.faces)
    np.save('selected.npy', constraint_v_ids)
    np.save('locations.npy', constraint_v_coords)

    #
    arap_deform = ARAPDeformation()
    arap_deform.load_from_mesh(mesh)
    arap_deform.setup()

    deformed_mesh = arap_deform.deform(constraint_v_ids, constraint_v_coords, w=1000.)

    vis_image = deformed_mesh.get_image(size=(w, h))
    im_utils.imshow(vis_image)


if __name__ == '__main__':
    main()
