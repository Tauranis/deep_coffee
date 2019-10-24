from deep_coffee.image_proc.base_video_streamer import BaseVideoStream
import cv2


class OpenCVStream(BaseVideoStream):

    def __init__(self, stream_path):
        super().__init__(stream_path)

        assert (isinstance(stream_path, str) or isinstance(stream_path,
                                                           int)), "stream_path must be string or int"

        self.cap = cv2.VideoCapture(stream_path)

    def next_frame(self):

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None
        else:
            return None
