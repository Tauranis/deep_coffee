from deep_coffee.image_proc.base_video_stream import BaseVideoStream
import cv2

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenCVStream(BaseVideoStream):

    def __init__(self, stream_path):
        self.stream_path = stream_path
        super().__init__(stream_path)

        assert (isinstance(stream_path, str) or isinstance(stream_path, int)
                or isinstance(stream_path, list)), "stream_path must be string or int or list of str"

        self._cap = None
        if not isinstance(stream_path, list):
            self._cap = cv2.VideoCapture(stream_path)

        self._i = 0

    def next_frame(self):

        if isinstance(self.stream_path, list):
            if self._i < len(self.stream_path):
                frame = cv2.imread(self.stream_path[self._i])
                self._i += 1
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    return None

            else:
                return None

            print(self._i)
        else:
            if self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    return None
            else:
                return None

    def save_frame(self, frame, filename):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_frame)
