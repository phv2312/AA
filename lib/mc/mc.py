import triangle as tr

from lib.mc import mc_utils
from utils import image_utils as im_utils


class TriangleMeshCreator:
    def __init__(self):
        self.interval = 20
        self.angle_constraint = 20
        self.area_constraint  = 120

    def create(self, image, mask):
        """
        Step1: sampling
        Step2: triangulate

        :param image:
        :param mask:
        :return:
        """

        sampling_points = mc_utils.sampling_points(image, mask, interval=self.interval)
        vertices, faces = mc_utils.do_triangulate(sampling_points, mask, self.angle_constraint, self.area_constraint)


