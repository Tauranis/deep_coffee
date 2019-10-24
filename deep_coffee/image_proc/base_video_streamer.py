

from abc import ABC

class BaseVideoStream(ABC):

    def __init__(self, stream_path):
        self.steam_path = stream_path
    
    def next_frame(self):
        pass
