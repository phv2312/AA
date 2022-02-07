import numpy as np
import triangle as tr
from utils import image_utils as im_utils


def sampling_points(image, image_mask, interval=20):
    """
    Step1: get boundary contour
    Step2: sampling by interval

    :param image:
    :param image_mask:
    :param interval
    :return:
    """

    boundary_contour, _ = im_utils.get_contour(image_mask)
    boundary_points = boundary_contour[::int(interval), 0, :].astype(np.int)
    return boundary_points


def do_triangulate(vertices, image_mask, angle_constraint=20, area_constraint=100):
    """

    :param vertices: vertices
    :param image_mask:
    :param angle_constraint:
    :param area_constraint:
    :return:
    """
    segments = []
    n_points = len(vertices)
    for start_i, end_i in zip(
            range(0, n_points - 1), range(1, n_points)
    ):
        segments += [(start_i, end_i)]
    segments += [(n_points - 1, 0)]

    # definition is here:
    # https://rufat.be/triangle/API.html
    output_dict = tr.triangulate(tri={
        'vertices': vertices,
        'segments': segments
    }, opts='-p -q%s -D -Y -a%s' % (str(angle_constraint), str(area_constraint)))

    #
    vertices = output_dict['vertices']
    faces = output_dict['triangles']

    return vertices, faces
