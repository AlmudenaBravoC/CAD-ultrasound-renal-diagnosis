import numpy as np


def mAd(im1, im2):
    av = np.abs(im1 - im2)
    return np.mean(av)


def IoU(im1, im2):
    """
    :param im1: ground truth as np array
    :param im2: output mask as np array
    :return: IoU
    """
    set1 = set([(row, col) for row in range(im1.shape[0]) for col in
                range(im1.shape[0]) if im1[row, col] > 0])
    set2 = set([(row, col) for row in range(im2.shape[0]) for col in
                range(im2.shape[0]) if im2[row, col] > 0])
    inter = np.intersect1d(set1, set2)
    union = np.union1d(set1, set2)
    return len(inter) / len(union)


def mean_iou(images, masks):
    """
    :param images: image set
    :param masks: ground truth set
    :return:
    """
    t = zip(*[images, masks])
    # the same as above but for a set of images
    return np.mean([IoU(im, mask) for im, mask in t])

def wpintersect(im1, im2):
    """

    :param im1: model output
    :param im2: ground truth
    :return: percentage of points in the intersection with the
    ground truth that belong to the output mask
    """
    # white pixels of the first image
    wp1 = np.argwhere(im1 == 255)
    wp2 = np.argwhere(im2 == 255)
    inter = np.intersect1d(wp1, wp2, return_indices = False)
    return len(inter) / len(wp1)