#
# MC: Mesh Creating
#

import cv2
import numpy as np
import triangle as tr

from lib.mc import mc_utils
from lib.interfaces import Mesh
from utils import image_utils as im_utils


class TriangleMeshCreator:
    def __init__(self, interval=20, angle_constraint=20, area_constraint=200, dilated_pixel=5):
        self.interval = interval
        self.angle_constraint = angle_constraint
        self.area_constraint = area_constraint
        self.dilated_pixel = dilated_pixel

    def create(self, image, mask):
        """
        Step1: sampling
        Step2: triangulate

        :param image:
        :param mask:
        :return:
        """
        mask = cv2.dilate(mask,
                          kernel=np.ones((self.dilated_pixel, self.dilated_pixel), dtype=np.uint8),
                          iterations=1)

        sampling_points = mc_utils.sampling_points(image, mask, interval=self.interval)
        vertices, faces = mc_utils.do_triangulate(sampling_points, mask, self.angle_constraint, self.area_constraint)

        #
        size = (image.shape[1], image.shape[0])
        normalized_vertices = Mesh.normalize_vertices(vertices, size)
        mesh = Mesh(normalized_vertices, faces + 1, texture_image=image)

        return mesh


def main_mc():
    image = cv2.imread('./../../data/image_alok.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread('./../../data/mask_alok.png', 0)
    mask = mask.astype(np.uint8)

    # im_utils.imshow(image)
    # im_utils.imshow(mask)

    mesh_mc_algo = TriangleMeshCreator()
    m = mesh_mc_algo.create(image, mask)
    triangle_mesh_image = m.get_image()
    im_utils.imshow(triangle_mesh_image)

    #
    #
    # vertices = vertices.astype(np.int)
    # faces = faces.astype(np.int)
    #
    # # visualize vertices & faces
    # vis_image = image.copy()
    # for x, y in vertices:
    #     cv2.circle(vis_image, (x, y), radius=3, color=(255,0,0), thickness=1)
    #
    # im_utils.imshow(vis_image)


if __name__ == '__main__':
    main_mc()


