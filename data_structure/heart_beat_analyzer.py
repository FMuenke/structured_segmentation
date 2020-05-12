import numpy as np

from machine_learning.time_series import TimeSeries


class HeartBeatAnalyzer:
    def __init__(self, video):
        self.video = video

    def run(self):
        t_g = []
        v_g = []
        t_p = []
        v_p = []
        for i in range(len(self.video)):
            gtr = self.video.get_label_map_of_index(i)
            if gtr is not None:
                gtr = gtr / 4
                t_g.append(i)
                v_g.append(np.sum(gtr))
            pre = self.video.get_segmentation_of_index(i)
            if pre is not None:
                pre = pre / 255
                t_p.append(i)
                v_p.append(np.sum(pre))

        if len(t_p) > 0:
            ts = TimeSeries(t_p, v_p)
            ts.plot()
