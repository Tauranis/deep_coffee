
import unittest
from deep_coffee.image_proc import OpenCVStream
from deep_coffee.image_proc import CropBeans_CV
import numpy as np


BEAN_IMAGE_PATH = "/app/test/image_proc/images/beans.jpg"


class TestCropBeans_CV(unittest.TestCase):

    def test_count_beans(self):
        stream = OpenCVStream(BEAN_IMAGE_PATH)
        frame = stream.next_frame()
        cropper = CropBeans_CV()

        beans_list = cropper.crop(frame)

        self.assertGreaterEqual(len(beans_list), 10, "There should be at least 10 beans on the image")


if __name__ == '__main__':
    unittest.main()
