import cv2
from IPython.display import display, Image
import matplotlib.pyplot as plt
import numpy as np

def lrSchedule(base_lr, iter, iters, epoch=0, step = (30, 60, 90), target_lr=0.0, mode='cosine'):
    lr = target_lr if target_lr else base_lr
    iters = iters if iter < iters else iter
    # every iteration
    if mode is 'cosine':
        lr += (base_lr - target_lr) * (1 + np.cos(np.pi * iter / iters)) / 2.0
    # every epochs
    if mode is 'step':
        if epoch in step:
            pass
    return lr




components = [[i for i in range(33)], [76, 87, 86, 85, 84, 83, 82], [88, 95, 94, 93, 92], [88, 89, 90, 91, 92],
              [76, 77, 78, 79, 80, 81, 82], [55 + i for i in range(5)], [51 + i for i in range(4)],
              [60, 67, 66, 65, 64], [60 + i for i in range(5)], [33 + i for i in range(9)], [68, 75, 74, 73, 72],
              [68 + i for i in range(5)], [42 + i for i in range(9)]]


def show(img_path):
    display(Image(img_path))


def landmark(path, points, detection=False):
    img = cv2.imread('WFLW_crop/' + path)
    if detection:
        x0, y0, x1, y1 = points[98 * 2:98 * 2 + 4]
        # img = img[y0:y1, x0:x1]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    for i in range(0, 98 * 2, 2):
        cv2.circle(img, (points[i], points[i + 1]), 1, (23, 25, 0), 1)
    plt.imshow(img[:, :, ::-1])


def drawLine(img, points, color=(25, 100, 0), return_image=True, thickness=1):
    """

    :param thickness:
    :param return_image:
    :param img:
    :param points: index 9 and 12 must be close (196 landmarks and 4 detection boxes)
    :param color:
    :return:
    """

    if return_image:
        color = color if len(img.shape) else (255,)
        for com in range(len(components)):
            for i in range(len(components[com]) - 1):
                p1 = components[com][i]
                p2 = components[com][i + 1]
                cv2.line(img, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
            if com is 9 or com is 12:
                p1 = components[com][0]
                p2 = components[com][-1]
                cv2.line(img, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
    else:
        img = np.zeros((img.shape[0], img.shape[1], 13), dtype=np.uint8)
        color = (255,)
        for com in range(len(components)):
            image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for i in range(len(components[com]) - 1):
                p1 = components[com][i]
                p2 = components[com][i + 1]
                cv2.line(image, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]),
                         color,
                         2, )
            if com in [9, 12]:
                p1 = components[com][0]
                p2 = components[com][-1]
                cv2.line(image, (points[p1 * 2], points[p1 * 2 + 1]),
                         (points[p2 * 2], points[p2 * 2 + 1]), color, thickness, )
            img[:, :, com] = image
    return img


def drawDistanceImg(img, points, color=(25, 100, 0)):
    img = drawLine(img, points, color=color, return_image=False)
    assert img.shape[2] is 13, "This is not the 13 components heatmap!"
    for i in range(13):
        img[:, :, i] = cv2.distanceTransform(255 - img[:, :, i], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return img


def drawGaussianHeatmap(img, points, color=(25, 100, 0), sigma=4):
    dist_img = drawDistanceImg(img, points, color=color)
    # heatmap = (1.0 / np.sqrt(2 * np.pi * sigma)) * np.exp(-1.0 * dist_img ** 2 / (2.0 * sigma ** 2))
    heatmap = np.exp(-1.0 * dist_img ** 2 / (2.0 * sigma ** 2))
    heatmap = np.where(dist_img < (3.0 * sigma), heatmap, 0)
    for i in range(13):
        maxVal = heatmap[:, :, i].max()
        minVal = heatmap[:, :, i].min()
        if maxVal == minVal:
            heatmap[:, :, i] = 0
        else:
            heatmap[:, :, i] = (heatmap[:, :, i] - minVal) / (maxVal - minVal)
    return heatmap


def drawPoint(img, points, color=(25, 100, 0)):
    """

    :param color: 
    :param img: RGB Image
    :param points: list type
    :return:
    """
    for i in range(0, 98 * 2, 2):
        cv2.circle(img, (points[i], points[i + 1]), 2, color, 2)
    return img


if __name__ == "__main__":
    with open('WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt') as files:
        path = files.readline()
        while path:
            path = path[:-1]
            info = path.split(' ')
            landmark(info)
            path = files.readline()
