import cv2
import numpy as np
import os
import glob


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
        _, contours, _ = cv2.findContours(
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
            obj_list.append(
                image_c[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]])
        return obj_list


    def crop(self, frame):
        sample_bbox_list = self._get_bboxes(frame)
        bean_list = self._crop_objects(frame, sample_bbox_list)

        return bean_list
