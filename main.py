import cv2
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np

import matplotlib as mpl
mpl.use('TkAgg')

def main():

    image = cv2.imread('data/grid.png', cv2.IMREAD_GRAYSCALE)
    image = image / 255
    image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)[1]
    # image = ~image

    # scale_percent = 60  # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    #
    # # resize image
    # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    final = np.ones(shape=image.shape)
    regions = measure.regionprops(measure.label(image))
    for region in regions:
        # x=1
        plt.imshow(image[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]])
        plt.show()
    #
    #     if region.area > 100000:#and not (0.98 < (region.bbox[2] / region.bbox[3]) < 1.02):
    # #         # final[region.coords] = 0
    #         continue
    # #
    #     final[region.coords]= 0
    #
    #
    # #
    # # label_map, num = measure.label(image, return_num=True, background=0)
    # # for label in range(num):
    # #     # if num <= 1:
    # #     #     continue
    # # #
    # #     pixel_ids = np.where(label_map == label)
    # #     # centroid = np.array([np.mean(dimension) for dimension in pixel_ids])
    # #     if len(pixel_ids) < 20000:
    # #         final[pixel_ids] = 0
    # #     x=1
    # # # label_map[label_map != 0] = 5

    # im = np.ones(image.shape)
    final[(0,0)] = 0
    plt.imshow(final, cmap='gray')
    plt.show()
    x=1


if __name__ == '__main__':
    main()
