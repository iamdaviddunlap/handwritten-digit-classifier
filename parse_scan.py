import cv2
import random
from skimage import measure
import numpy as np
import os
from scipy import ndimage
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


ODIA_DIR = os.path.join(os.path.dirname(__file__), 'data/odia')
SCAN_DIR = os.path.join(ODIA_DIR, 'scan/cropped')
OUTPUT_DIR = os.path.join(ODIA_DIR, 'images')
OUTPUT_ALL_DIR = os.path.join(OUTPUT_DIR, 'all')
TEMPLATE_NAME = 'template.png'
EPSILON = 0.001
NUM_AUGS = 3


def make_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)
    os.makedirs(OUTPUT_ALL_DIR)


def read_numerals_and_save():
    make_dirs()
    crop_percentage = 0.10

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
            cropped_bounds = (int(bbox[0] + bbox_height * crop_percentage),
                              int(bbox[2] - bbox_height * crop_percentage),
                              int(bbox[1] + bbox_width * crop_percentage),
                              int(bbox[3] - bbox_width * crop_percentage))
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
            crop = np.pad(crop, (6, 6))  # pad the numeral to make it 28 x 28
            crop = np.clip(np.log(crop + EPSILON), 0, 255)  # scale the pixels to brighten the numeral
            crop = (255 * crop / crop.max()).astype(np.uint8)  # stretch the values back to 0-255

            # save the numeral image
            image_save_path = os.path.join(OUTPUT_ALL_DIR, f'{numeral_image_num}.png')
            cv2.imwrite(image_save_path, crop)
            numeral_image_num += 1


def augment(image):
    images = list()

    # pad the input image for scaling and rotation augmentations
    image_temp = np.pad(image, image.shape)
    # get the bounding box of where original image is
    bbox = (len(image), len(image), 2 * len(image), 2 * len(image))

    # scale
    for _ in range(NUM_AUGS):
        # random positive or negative crop; crop sign has inverse effect on numeral scaling
        crop_percentage = random.choice([random.uniform(-0.5, -0.2), random.uniform(0.05, 0.15)])

        # crop the box to effectively scale the numeral
        cropped_bounds = (int(bbox[0] + len(image) * crop_percentage),
                          int(bbox[2] - len(image) * crop_percentage),
                          int(bbox[1] + len(image) * crop_percentage),
                          int(bbox[3] - len(image) * crop_percentage))
        scaled = image_temp[cropped_bounds[0]:cropped_bounds[1], cropped_bounds[2]:cropped_bounds[3]]
        scaled = cv2.resize(scaled, image.shape)  # resizing back to the original size completes the scaling

        images.append(scaled)

        # plt.figure(dpi=100)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title('Original')
        # plt.subplot(1, 2, 2)
        # plt.imshow(scaled, cmap='gray')
        # plt.title('Scaled')
        # plt.suptitle(f'Scale Factor: {round(1 / (1 - crop_percentage), 2)}')
        # plt.show()
        # break

    # translate
    image_temp = np.pad(image, image.shape)
    for _ in range(NUM_AUGS):
        # random positive or negative translation in each direction as a percentage of the input image size
        translate_values = [random.uniform(0.05, 0.15) * random.choice([-1, 1]) for _ in range(2)]

        # crop the box to effectively translate the numeral
        cropped_bounds = (int(bbox[0] + len(image) * translate_values[0]),
                          int(bbox[2] + len(image) * translate_values[0]),
                          int(bbox[1] + len(image) * translate_values[1]),
                          int(bbox[3] + len(image) * translate_values[1]))
        translated = image_temp[cropped_bounds[0]:cropped_bounds[1], cropped_bounds[2]:cropped_bounds[3]]

        images.append(translated)

        # plt.figure(dpi=100)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title('Original')
        # plt.subplot(1, 2, 2)
        # plt.imshow(translated, cmap='gray')
        # plt.title('Translated')
        # plt.suptitle(f'Translation: ({round(-translate_values[0] * len(image), 2)}, '
        #              f'{round(-translate_values[1] * len(image), 2)}) pixels')
        # plt.show()
        # break

    # rotate
    for _ in range(NUM_AUGS):
        # random positive or negative angle to rotate the numeral by
        angle = random.uniform(15, 30)
        if random.random() < 0.5:
            angle *= -1

        # rotate the image
        rotated = ndimage.rotate(image, angle, reshape=False)

        images.append(rotated)

        # plt.figure(dpi=100)
        # plt.subplot(1, 2, 1)
        # plt.imshow(image, cmap='gray')
        # plt.title('Original')
        # plt.subplot(1, 2, 2)
        # plt.imshow(rotated, cmap='gray')
        # plt.title('Rotated')
        # plt.suptitle(f'Rotation angle: {round(angle, 2)} degrees')
        # plt.show()
        # break

    return images


def create_train_test_split():
    images = list()
    labels = list()

    # augment each image
    for filename in os.listdir(OUTPUT_ALL_DIR):
        numeral = int(filename.split('.')[0]) % 10  # get the true numeral

        # augment
        filepath = os.path.join(OUTPUT_ALL_DIR, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        augmented_images = augment(image)

        # add to all images and labels
        images.append(image)
        images.extend(augmented_images)
        labels.extend([numeral] * (len(augmented_images) + 1))

    # create train/test split
    X_train, X_test, y_train, y_test = \
        train_test_split(images, labels, test_size=(1/7), random_state=42)

    # save split
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train, allow_pickle=True)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test, allow_pickle=True)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train, allow_pickle=True)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test, allow_pickle=True)


if __name__ == '__main__':
    read_numerals_and_save()
    create_train_test_split()
