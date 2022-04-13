import cv2
from skimage import measure
import numpy as np
import os
import shutil

READ_SCAN = False

ODIA_DIR = os.path.join(os.path.dirname(__file__), 'data/odia')
SCAN_DIR = os.path.join(ODIA_DIR, 'scan/cropped')
OUTPUT_DIR = os.path.join(ODIA_DIR, 'images')
OUTPUT_ALL_DIR = os.path.join(OUTPUT_DIR, 'all')
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
OUTPUT_TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
TEMPLATE_NAME = 'template.png'
CROP_PERCENTAGE = 0.10
EPSILON = 0.001


def make_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)
    os.makedirs(OUTPUT_ALL_DIR)
    os.makedirs(OUTPUT_TRAIN_DIR)
    os.makedirs(OUTPUT_TEST_DIR)


def read_numerals_and_save():
    make_dirs()

    # load a template grid where numerals are placed
    template = cv2.imread(os.path.join(SCAN_DIR, TEMPLATE_NAME), cv2.IMREAD_GRAYSCALE)
    template = cv2.threshold(template, 200, 255, cv2.THRESH_BINARY)[1]

    # get the region for each numeral
    regions = measure.regionprops(measure.label(template))
    regions = [region for region in regions if region.area > 5000]
    assert len(regions) == 130  # should be 10 * 13 = 130 grid squares

    numeral_image_num = 0
    # iterate over pages of handwritten digits
    for filename in os.listdir(SCAN_DIR):
        if 'template' in filename:
            continue

        # read in the scan
        scan = cv2.imread(os.path.join(SCAN_DIR, filename), cv2.IMREAD_GRAYSCALE)

        for region in regions:

            # filter out image that was messed up when writing
            if numeral_image_num == 422:
                numeral_image_num += 1
                continue

            # get the bounding box of where the numeral should be
            bbox = region.bbox
            bbox_height = bbox[2] - bbox[0] + 1
            bbox_width = bbox[3] - bbox[1] + 1

            # crop the box slightly to remove grid lines
            cropped_bounds = (int(bbox[0] + bbox_height * CROP_PERCENTAGE),
                              int(bbox[2] - bbox_height * CROP_PERCENTAGE),
                              int(bbox[1] + bbox_width * CROP_PERCENTAGE),
                              int(bbox[3] - bbox_width * CROP_PERCENTAGE))
            crop = scan[cropped_bounds[0]:cropped_bounds[1], cropped_bounds[2]:cropped_bounds[3]]

            # the background of the paper may not be perfectly white; set it to white
            crop[crop > 220] = 255

            # isolate the numeral in the image using extrema of pixels that are not white
            binarized = cv2.threshold(crop, 200, 255, cv2.THRESH_BINARY)[1]
            pixel_ids = np.where(binarized < 255)
            crop = crop[np.min(pixel_ids[0]):np.max(pixel_ids[0]) + 1, np.min(pixel_ids[1]):np.max(pixel_ids[1]) + 1]

            # pad the numeral with white pixels to make it square-ish
            if crop.shape[0] < crop.shape[1]:
                diff = crop.shape[1] - crop.shape[0]
                fill = np.full((diff // 2, crop.shape[1]), 255)
                crop = np.vstack([fill, crop, fill])
            else:
                diff = crop.shape[0] - crop.shape[1]
                fill = np.full((crop.shape[0], diff // 2), 255)
                crop = np.hstack([fill, crop, fill])

            # apply a Gaussian blur
            crop = cv2.GaussianBlur(crop.astype(np.uint8), (3, 3), cv2.BORDER_DEFAULT)

            # the goal is for the images to be 28 x 28

            # make the numeral smaller than the target
            crop = cv2.resize(crop.astype(np.uint8), (16, 16))
            crop = ~crop.astype(np.uint8)  # invert the numeral to make it white on black
            crop = np.pad(crop, (8, 8))  # pad the numeral to make it 28 x 28
            crop = np.clip(np.log(crop + EPSILON), 0, 255)  # scale the pixels to brighten the numeral
            crop = (255 * crop / crop.max()).astype(np.uint8)  # stretch the values back to 0-255

            # save the numeral image
            image_save_path = os.path.join(OUTPUT_ALL_DIR, f'{numeral_image_num}.png')
            cv2.imwrite(image_save_path, crop)
            numeral_image_num += 1


if __name__ == '__main__':
    if READ_SCAN:
        read_numerals_and_save()
    else:
        # data augment and save train/test split
        pass
