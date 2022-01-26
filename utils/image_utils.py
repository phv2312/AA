import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure

class ConnectedComponentRegion:
    def __init__(self, region):
        self.coords = region['coords'] # list of row, col
        self.area = region['area'] # number of pixels in the component
        self.rgb_color  = None

    def update_rgb_color(self, rgb_image):
        sampling_coord = self.coords[0]
        y, x = sampling_coord
        self.rgb_color = rgb_image[y, x, :]


def get_component(np_image, background_value=0):
    """
    Run measure.label for the given image
    :param np_image:
    :return:
    """

    labels = measure.label(np_image, neighbors=4, background=background_value)
    regions = measure.regionprops(labels)

    cc_list = []
    for region in regions:
        cc_list += [ConnectedComponentRegion(region)]

    return cc_list


def heuristic_segment_character(np_image, background_image=(255,255,255)):
    """
    Use heuristic rules to segment the anime character out of image
    One character per image
    :param np_image:
    :return:
    """

    ys, xs = np.where(
        (np_image[:,:,0] == background_image[0]) &
        (np_image[:,:,1] == background_image[1]) &
        (np_image[:,:,2] == background_image[2])
    )

    mask = np.zeros(shape=(np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
    mask[ys, xs] = 1

    cc_list = get_component(mask, background_value=0)
    cc = sorted(cc_list, key=lambda elem: elem.area)[-1]

    new_mask = np.zeros(shape=(np_image.shape[0], np_image.shape[1]), dtype=np.uint8)
    new_mask[cc.coords[:,0], cc.coords[:,1]] = 1
    new_mask = 1 - new_mask

    return new_mask


def imshow(im):
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
    image_path = r"./../data/tobi.jpg"
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    print ('raw image')
    imshow(im)

    mask = heuristic_segment_character(im)
    imshow(mask)
