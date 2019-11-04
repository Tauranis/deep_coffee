from deep_coffee.image_proc.opencv_stream import OpenCVStream
import argparse
import cv2
import numpy as np
import os
import tqdm
import glob
import pathlib

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataAugRotate(object):

    def rotate(self, frame, angle):
        frame_height, frame_width = frame.shape[0], frame.shape[1]
        center_x, center_y = frame_width // 2, frame_height // 2

        rot_M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)

        cos = np.abs(rot_M[0, 0])
        sin = np.abs(rot_M[0, 1])

        rot_frame_height = int((frame_height * cos) + (frame_width * sin))
        rot_frame_width = int((frame_height * cos) + (frame_width * sin))

        rot_M[0, 2] += (rot_frame_width / 2) - center_x
        rot_M[1, 2] += (rot_frame_height / 2) - center_y

        # return cv2.warpAffine(frame, rot_M, (rot_frame_height, rot_frame_width), borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(frame, rot_M, (rot_frame_height, rot_frame_width), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def rotate_objects(image_list, output_dir, angle_list, ext):

    rotator = DataAugRotate()

    for image_filename in tqdm.tqdm(image_list):

        stream = OpenCVStream(image_filename)
        frame = stream.next_frame()

        if frame is None:
            continue

        for angle in angle_list:
            aug_image_filename = os.path.join(output_dir, ((".".join(image_filename.split(
                '.')[:-1])+"_{}".format(angle)).split('/')[-1] + "."+ext))
            image_path = pathlib.Path(aug_image_filename)

            if not image_path.exists():
                frame_rotated = rotator.rotate(frame, angle)
                stream.save_frame(frame_rotated, aug_image_filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Rotate Beans")
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--angle_list', type=str,
                        default='45,90,135,180,225,270')
    parser.add_argument('--ext', type=str, default='jpg')
    args = parser.parse_args()

    angle_list = [int(a) for a in args.angle_list.split(',')]

    image_list = glob.glob(
        "{}/*{}".format(args.input_dir, args.ext), recursive=True)

    rotate_objects(image_list, args.output_dir, angle_list, args.ext)
