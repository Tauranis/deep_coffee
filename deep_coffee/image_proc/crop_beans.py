from deep_coffee.image_proc.opencv_stream import OpenCVStream
import argparse
import cv2
import numpy as np
import os
import uuid
import tqdm
import glob

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CropBeans_CV(object):

    def _get_bboxes(self, image, expand_borders_offset=0.005):

        def _expand_bbox(_bbox, _offset_x, _offset_y):
            return (_bbox[0]-_offset_x,
                    _bbox[1]-_offset_y,
                    _bbox[2]+2*_offset_x,
                    _bbox[3]+2*_offset_y)

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # denoise through median blur
        image_gray = cv2.medianBlur(image_gray, 7)

        # thresholding
        _, image_bin = cv2.threshold(
            image_gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # denoise through morph ops
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        image_bin = cv2.morphologyEx(
            image_bin, cv2.MORPH_CLOSE, morph_kernel, iterations=5)  # remove black dots
        image_bin = cv2.morphologyEx(
            image_bin, cv2.MORPH_OPEN, morph_kernel, iterations=5)  # fill inside beans

        # get bbox
        contours, _ = cv2.findContours(
            image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = [cv2.boundingRect(c) for c in contours[1:]]

        # expand borders
        offset_x = int(image_bin.shape[0]*expand_borders_offset)
        offset_y = int(image_bin.shape[1]*expand_borders_offset)

        bbox_list = [_expand_bbox(c, offset_x, offset_y) for c in bbox_list]

        return bbox_list

    def _crop_objects(self, image, bbox_list):
        image_c = np.copy(image)
        obj_list = []
        for bbox in bbox_list:
            bean_image = image_c[bbox[1]:(bbox[1] +
                                          bbox[3]), bbox[0]:(bbox[0]+bbox[2])]

            if bean_image.shape[0]*bean_image.shape[1] == 0:
                continue

            if (((bean_image.shape[0] > bean_image.shape[1]) and (bean_image.shape[0]/bean_image.shape[1] > 2))
                    or ((bean_image.shape[1] > bean_image.shape[0]) and (bean_image.shape[1]/bean_image.shape[0] > 2))):
                continue
            
            # pad image to be square
            pad_size = 0
            if bean_image.shape[0] < bean_image.shape[1]:
                pad_size = int((bean_image.shape[1] - bean_image.shape[0])/2)
                bean_image = cv2.copyMakeBorder(
                    bean_image, pad_size, pad_size, 0, 0, cv2.BORDER_REPLICATE)
            else:
                pad_size = int((bean_image.shape[0] - bean_image.shape[1])/2)
                bean_image = cv2.copyMakeBorder(
                    bean_image, 0, 0, pad_size, pad_size, cv2.BORDER_REPLICATE)

            obj_list.append(bean_image)

        return obj_list

    def crop(self, frame):
        sample_bbox_list = self._get_bboxes(frame)
        bean_list = self._crop_objects(frame, sample_bbox_list)

        return bean_list


def crop_beans(image_list, output_dir, ext, min_area):

    cropper = CropBeans_CV()
    stream = OpenCVStream(image_list)

    pbar = tqdm.tqdm(total=len(image_list))
    while True:
        frame = stream.next_frame()

        if frame is None:
            break

        bean_list = cropper.crop(frame)

        for bean_image in bean_list:

            if bean_image.shape[0]*bean_image.shape[1] >= min_area:
                bean_image_filename = os.path.join(
                    output_dir, str(uuid.uuid4())+"."+ext)

                bean_image = cv2.cvtColor(
                    bean_image, cv2.COLOR_RGB2BGR)  # convert to BGR
                cv2.imwrite(bean_image_filename, bean_image)
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Crop Beans")
    parser.add_argument('--raw_images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ext', type=str, default='jpg')
    parser.add_argument('--min_area', type=int, default=22500)  # 150x150
    args = parser.parse_args()

    raw_images_list = glob.glob(
        "{}/*{}".format(args.raw_images_dir, args.ext), recursive=True)
    crop_beans(raw_images_list, args.output_dir, args.ext, args.min_area)
