import os
import cv2
import numpy as np

from data_structure.image_handler import ImageHandler


class VideoFrameTag:
    def __init__(self, video, frame_no, color_coding):
        self.video = video
        self.frame_no = frame_no
        self.color_coding = color_coding

    def __str__(self):
        return "Tag: VideoID: {} - FrameNo.: {}".format(self.video.id, self.frame_no)

    def get_offset_frame(self, offset):
        return VideoFrameTag(video=self.video, frame_no=self.frame_no+offset, color_coding=self.color_coding)

    def load_x(self, neighbours=5, offset=0):
        frame_no = self.frame_no + offset
        if neighbours == 0:
            data = self.video.get_frame_of_index(frame_no)
            if data is None:
                print(frame_no)
            return data

        x = []
        xc = self.video.get_frame_of_index(frame_no)
        if len(xc.shape) < 3:
            height, width = xc.shape[:2]
            xc = np.reshape((height, width, 1))
        height, width, ch = xc.shape
        x.append(xc)
        for i in range(neighbours):
            xi = self.video.get_frame_of_index(frame_no - i)
            if xi is None:
                xi = np.zeros((height, width, ch))
            x.append(xi)

        for i in range(neighbours):
            xi = self.video.get_frame_of_index(frame_no + i)
            if xi is None:
                xi = np.zeros((height, width, ch))
            x.append(xi)

        return np.concatenate(x, axis=2)

    def load_y_as_color_map(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], 3))
        lbm = self.video.get_label_map_of_index(self.frame_no)

        if lbm is None:
            return y_img

        lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
        for idx, cls in enumerate(self.color_coding):
            for x in range(label_size[1]):
                for y in range(label_size[0]):
                    if lbm[y, x, 0] == self.color_coding[cls][0][2] \
                            and lbm[y, x, 1] == self.color_coding[cls][0][1] \
                            and lbm[y, x, 2] == self.color_coding[cls][0][0]:
                        y_img[y, x, :] = self.color_coding[cls][1]
        return y_img

    def load_y(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1]))
        lbm = self.video.get_label_map_of_index(self.frame_no)

        if lbm is None:
            return y_img

        lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
        for idx, cls in enumerate(self.color_coding):
            for x in range(label_size[1]):
                for y in range(label_size[0]):
                    if lbm[y, x, 0] == self.color_coding[cls][0][2] \
                            and lbm[y, x, 1] == self.color_coding[cls][0][1] \
                            and lbm[y, x, 2] == self.color_coding[cls][0][0]:
                        y_img[y, x] = idx + 1
        return y_img

    def write_result(self, res_path, color_map):
        im_id = "{}-{}.jpg".format(self.video.id, self.frame_no)
        h, w = color_map.shape[:2]
        label = self.load_y_as_color_map((h, w))
        border = 255 * np.ones((h, 10, 3))
        r = np.concatenate([label, border, color_map], axis=1)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, r)

    def eval(self, color_map, stats_handler):
        height, width = color_map.shape[:2]
        lbm = self.video.get_label_map_of_index(self.frame_no)
        color_map = np.array(color_map, dtype=np.int)
        if lbm is not None:
            lbm = cv2.resize(lbm, (width, height), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(width):
                    for y in range(height):
                        a = lbm[y, x, :] == self.color_coding[cls][0]
                        b = color_map[y, x, :] == self.color_coding[cls][1]
                        if a.all():
                            if b.all():
                                stats_handler.count(cls, "tp")
                            else:
                                stats_handler.count(cls, "fn")
                        else:
                            if b.all():
                                stats_handler.count(cls, "fp")

    def visualize_result(self, vis_path, color_map):
        im_id = "{}-{}.jpg".format(self.video.id, self.frame_no)
        vis_file = os.path.join(vis_path, im_id)
        img_h = ImageHandler(self.load_x(neighbours=0))
        cv2.imwrite(vis_file, img_h.overlay(color_map))
