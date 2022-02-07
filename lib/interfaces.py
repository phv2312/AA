import cv2
import numpy as np


class Mesh:
    def __init__(self, vertices, faces, texture_image=None):
        self.vertices = vertices
        self.faces = faces
        self.texture_image = texture_image

    @staticmethod
    def normalize_vertices(vertices, size):
        """
        Normalize vertices to range [-1., 1.]
        :param vertices: shape (N, 2)
        :param size: (w, h)
        :return:
        """
        w, h = size
        vertices_ = vertices.copy()
        vertices_ = vertices_ / np.array([w, h]).reshape(1, 2)
        vertices_ = 2 * (vertices_ - 0.5)

        return vertices_

    def initialize_texture_image(self, texture_image):
        self.texture_image = texture_image

    def get_image(self, size=None):
        """

        :param size:
        :return:
        """
        if size is None:
            assert self.texture_image is not None, 'Both size & texture_image are None...'
            size = (self.texture_image.shape[1], self.texture_image.shape[0])

        w, h = size
        visualize_image = np.zeros(shape=(h, w), dtype=np.uint8)

        # drawing mesh with triangle only
        for face in self.faces:
            i, j, k = face

            contours = np.array([
                self.vertices[i - 1],
                self.vertices[j - 1],
                self.vertices[k - 1]
            ]).reshape((-1, 1, 2))
            contours = 0.5 * (contours + 1) * np.array([w, h]).reshape((1, 1,2 ))
            contours = contours.astype(np.int)

            cv2.drawContours(visualize_image, [contours], -1, color=(255, 0, 0), thickness=1)

        # return
        return visualize_image


if __name__ == '__main__':
    pass