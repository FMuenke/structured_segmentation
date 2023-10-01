# pylint ignore

import os
import numpy as np
from structured_segmentation.data_structure.experimental.video import Video


class VideoSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path = path_to_data_set
        self.color_coding = color_coding
        self.video_path = os.path.join(self.path, "Videos")
        self.label_path = os.path.join(self.path, "Labels")
        self.roi_path = os.path.join(self.path, "roi")
        self.work_path = os.path.join(self.path, "_work")
        self.segmentation_path = os.path.join(self.path, "segmentation")

        self.videos = dict()

        self.working_file_formats = ['avi', 'mp4', 'mov']

    def is_complete(self):
        if not os.path.isdir(self.video_path):
            return False
        if not os.path.isdir(self.label_path):
            return False
        if not os.path.isdir(self.work_path):
            return False
        return True

    def _load(self, path, video_type, console_out=True):
        if os.path.isdir(path):
            for f in os.listdir(path):
                v = os.path.join(path, f)
                if os.path.isdir(v):
                    if video_type in v:
                        self._load(v, video_type, console_out)
                else:
                    if f.split('.')[-1] not in self.working_file_formats:
                        continue
                    video = Video(video_id=f[:-4],
                                  data_path=v,
                                  label_path=self.label_path,
                                  region_of_interest_path=self.roi_path,
                                  segmentation_path=self.segmentation_path)
                    if console_out:
                        print(video)
                    self.videos[len(self.videos)] = video

    def load(self, video_type="Compressed", console_out=True):
        self._load(self.video_path, video_type, console_out)
        return self.get_tags()

    def get_tags(self):
        tag_set = {}
        for _, v in self.videos.items():
            tag_list = v.get_tag_list(self.color_coding)
            for t in tag_list:
                tag_set[len(tag_set)] = t
        return tag_set

    def get_videos(self):
        return self.videos

    def split(self, tag_set, percentage=0.2, random=True):
        train_set = []
        validation_set = []
        if random:
            dist = np.random.permutation(len(tag_set))
        else:
            dist = range(len(tag_set))
        for d in dist:
            if len(validation_set) > percentage * len(tag_set):
                train_set.append(tag_set[d])
            else:
                validation_set.append(tag_set[d])
        print("Training Samples: {}".format(len(train_set)))
        print("Validation Samples: {}".format(len(validation_set)))
        print(" ")
        return np.array(train_set), np.array(validation_set)
