
import unittest
from deep_coffee.image_proc import OpenCVStream
import numpy as np


RED_IMAGE_PATH = "/app/test/image_proc/images/red.png"
GREEN_IMAGE_PATH = "/app/test/image_proc/images/green.png"
BLUE_IMAGE_PATH = "/app/test/image_proc/images/blue.png"


class TestOpenCVStream(unittest.TestCase):

    def test_isred(self):
        stream = OpenCVStream(RED_IMAGE_PATH)
        frame = stream.next_frame()

        frame_red_color = np.sum(frame[:, :, 0])
        frame_green_color = np.sum(frame[:, :, 1])
        frame_blue_color = np.sum(frame[:, :, 2])

        frame_is_red = (frame_red_color > 250) or (
            frame_green_color < 10) or (frame_blue_color < 10)

        self.assertTrue(frame_is_red, "image not in RGB format")

    def test_isgreen(self):
        stream = OpenCVStream(GREEN_IMAGE_PATH)
        frame = stream.next_frame()

        frame_red_color = np.sum(frame[:, :, 0])
        frame_green_color = np.sum(frame[:, :, 1])
        frame_blue_color = np.sum(frame[:, :, 2])

        frame_is_green = (frame_red_color < 10) or (
            frame_green_color > 250) or (frame_blue_color < 10)

        self.assertTrue(frame_is_green, "image not in RGB format")

    def test_isblue(self):
        stream = OpenCVStream(BLUE_IMAGE_PATH)
        frame = stream.next_frame()

        frame_red_color = np.sum(frame[:, :, 0])
        frame_green_color = np.sum(frame[:, :, 1])
        frame_blue_color = np.sum(frame[:, :, 2])

        frame_is_blue = (frame_red_color < 10) or (
            frame_green_color < 10) or (frame_blue_color > 250)

        self.assertTrue(frame_is_blue, "image not in RGB format")


if __name__ == '__main__':
    unittest.main()
