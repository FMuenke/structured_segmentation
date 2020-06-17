import cv2
import os
import csv
import numpy as np
from skimage.measure import regionprops, label as sklabel
import tifffile as tiff
from tqdm import tqdm

from data_structure.video_frame_tag import VideoFrameTag

from utils.utils import check_n_make_dir

# from warnings import warn as warn


class Video:
    def __init__(self, video_id, data_path, label_path, region_of_interest_path, segmentation_path=None):
        self.id = video_id.replace(' ', '_').replace('-', '_')
        self.data_path = data_path
        self.label_path = label_path
        self.region_of_interest_path = region_of_interest_path
        self.segmentation_path = segmentation_path

        self.num_of_frames = self._count_frames()
        self.label_maps = self._register_label_maps()
        self.regions_of_interest = self._register_regions_of_interest()
        self.segmentation = self._register_segmentation()

        self.frame_height, self.frame_width = 100, 100
        frame = self.get_frame_of_index(1)
        self.frame_height, self.frame_width = frame.shape[:2]

    def __len__(self):
        return self.num_of_frames

    def __str__(self):
        return "Video: {} with {} Frames, {} Labels, {} RoI, {} Segmentation".format(self.id,
                                                                                     self.num_of_frames,
                                                                                     len(self.label_maps),
                                                                                     len(self.regions_of_interest),
                                                                                     len(self.segmentation))

    def _count_frames(self):
        cap = cv2.VideoCapture(self.data_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def _get_frame_and_class_from_name(self, f):
        f = f.replace('{}_'.format(self.id), '', 1)
        f = f.replace('frame', '')
        f = f.replace('_label', '')
        f = f[:-4]  # removes the '.tif' at the end of the name. TODO: remove also longer file endings (e.g. .tiff)
        return f.split('_')

    def _register_label_maps(self):
        label_maps = dict()
        if os.path.isdir(self.label_path):
            for f in os.listdir(self.label_path):
                f_norm = f.replace('-', '_').replace(' ', '_')
                if f_norm.startswith(self.id + '_') and "label" in f:
                    label_f = os.path.join(self.label_path, f)
                    try:
                        frame, cls = self._get_frame_and_class_from_name(f_norm)
                    except ValueError:
                        # warn('Label \'' + f + '\' wrong formatted')
                        # If a video has the same name as another with '_<addition>' at the end, the shorter \
                        # video tries to load the labels of the other video but fails do to the name extension.
                        # TODO: Find a better solution to suppress this
                        continue
                    if frame.startswith('0'):
                        frame = str(int(frame))
                    if frame not in label_maps:
                        label_maps[frame] = {cls: label_f}
                    else:
                        label_maps[frame][cls] = label_f
        return label_maps

    def _register_segmentation(self):
        label_maps = dict()
        if self.segmentation_path is not None:
            if os.path.isdir(self.segmentation_path):
                for f in os.listdir(self.segmentation_path):
                    if f.startswith(self.id + '_'):
                        frame = f.replace(self.id + "_", "")
                        frame = frame.replace(".png", "")
                        label_maps[frame] = os.path.join(self.segmentation_path, f)

        return label_maps

    def _register_regions_of_interest(self):
        regions_of_interest = dict()
        csv_file = os.path.join(self.region_of_interest_path, "{}.csv".format(self.id))
        if os.path.isfile(csv_file):
            with open(csv_file) as csv_f:
                csv_reader = csv.reader(csv_f, delimiter=";")
                for row in csv_reader:
                    if row[0] != "frame_no":
                        regions_of_interest[row[0]] = [max(0, float(row[1])),
                                                       max(0, float(row[2])),
                                                       min(1, float(row[3])),
                                                       min(1, float(row[4]))]
        return regions_of_interest

    def get_frames_per_second(self):
        cap = cv2.VideoCapture(self.data_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def get_list_of_frames(self, as_grayscale=False, resize_options=None):
        cap = cv2.VideoCapture(self.data_path)
        frames = list()
        if resize_options is not None:
            max_size, scale_factor = resize_options
        while True:
            ret, frame = cap.read()
            if ret is False:
                cap.release()
                break
            if as_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_options is not None:
                while True:
                    frame_h = frame.shape[0]
                    frame_w = frame.shape[1]
                    if frame_h > max_size[0] or frame_w > max_size[1]:
                        frame = cv2.resize(frame, (int(frame_w*scale_factor), int(frame_h*scale_factor)), interpolation=cv2.INTER_AREA)
                    else:
                        break
            frames.append(frame)
        return np.array(frames)

    def is_labeled(self):
        return len(self.label_maps) > 0

    def get_label_files_of_index(self, idx):
        if str(idx) in self.label_maps.keys():
            return self.label_maps[str(idx)]
        else:
            return None

    def get_frame_of_index(self, idx, roi_only=False):
        empty_frame = np.zeros((self.frame_height, self.frame_width, 3))
        if 0 < idx < self.num_of_frames:
            cap = cv2.VideoCapture(self.data_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx-1)
            ret, frame = cap.read()
            cap.release()
            if frame is None:
                frame = empty_frame
            if not roi_only:
                return frame
            if str(idx) in self.regions_of_interest:
                roi = self.regions_of_interest[str(idx)]
                height, width = frame.shape[:2]
                return frame[int(roi[1]*height):int(roi[3]*height), int(roi[0]*width):int(roi[2]*width), :]
            else:
                return empty_frame
        else:
            return empty_frame

    def get_label_map_of_index(self, idx, obj_type="4", roi_only=False):
        empty_label_map = np.zeros((self.frame_height, self.frame_width, 3))
        if 0 < idx < self.num_of_frames:
            label_map_path = self.get_label_files_of_index(idx)
            if label_map_path is not None:
                if obj_type not in label_map_path.keys():
                    # raise KeyError("Label map with key {} for frame {} of video {} not existing".format(obj_type, idx, self.id))
                    return None
                lbm = cv2.imread(label_map_path[obj_type])
                if not roi_only:
                    return lbm
                if str(idx) in self.regions_of_interest:
                    roi = self.regions_of_interest[str(idx)]
                    height, width = lbm.shape[:2]
                    return lbm[int(roi[1]*height):int(roi[3]*height), int(roi[0]*width):int(roi[2]*width), :]
                else:
                    return None
            else:
                return None
        else:
            return None

    def get_segmentation_of_index(self, idx, roi_only=False):
        if str(idx) in self.segmentation:
            lbm = cv2.imread(self.segmentation[str(idx)])
            if lbm is not None:
                if not roi_only:
                    return lbm
                if str(idx) in self.regions_of_interest:
                    roi = self.regions_of_interest[str(idx)]
                    height, width = lbm.shape[:2]
                    return lbm[int(roi[1]*height):int(roi[3]*height), int(roi[0]*width):int(roi[2]*width), :]
                else:
                    return None
            else:
                return None
        else:
            return None

    def export(self, image_folder, label_map_folder, roi_only=True):
        check_n_make_dir(image_folder)
        check_n_make_dir(label_map_folder)
        print("Exporting {}".format(self))
        for i in tqdm(range(len(self))):
            image = self.get_frame_of_index(i, roi_only)
            label = self.get_label_map_of_index(i, roi_only=roi_only)
            i_file = os.path.join(image_folder, "{}_{}.jpg".format(self.id, i))
            l_file = os.path.join(label_map_folder, "{}_{}.png".format(self.id, i))
            if image is not None and label is not None:
                cv2.imwrite(i_file, image)
                cv2.imwrite(l_file, label)

    def export_frames(self, image_folder, roi_only=True):
        check_n_make_dir(image_folder)
        print("Exporting {}".format(self))
        for i in tqdm(range(len(self))):
            image = self.get_frame_of_index(i, roi_only)
            i_file = os.path.join(image_folder, "{}_{}.jpg".format(self.id, i))
            if image is not None:
                cv2.imwrite(i_file, image)

    def get_roi_of_index(self, idx, in_percent=False):
        labels_dict = self.get_label_files_of_index(idx)

        if labels_dict is None:
            return None

        label_files = list(labels_dict.values())

        tiff.imread(label_files).tolist()

        first = True
        label_img = None
        for label in label_files:
            if first:
                label_img = tiff.imread(label)
                first = False
            else:
                label_img = np.logical_or(label_img, tiff.imread(label))

        bboxes = self.get_bounding_boxes_for_single_label(label_img, in_percent=in_percent)

        x1 = None
        y1 = None
        x2 = None
        y2 = None

        for bbox in bboxes:
            if x1 is None:
                x1, y1, x2, y2 = bbox
            else:
                if x1 > bbox[0]:
                    x1 = bbox[0]
                if y1 > bbox[1]:
                    y1 = bbox[1]
                if x2 < bbox[2]:
                    x2 = bbox[2]
                if y2 < bbox[3]:
                    y2 = bbox[3]

        return x1, y1, x2, y2

    @staticmethod
    def get_bounding_boxes_for_single_label(label, obj_type=True, box_type='x1,y1,x2,y2', in_percent=False):
        boxes = list()

        label[label != obj_type] = 0

        regions = regionprops(sklabel(label))

        for region in regions:
            y1, x1, y2, x2 = region.bbox
            if in_percent:
                label_h, label_w = label.shape
                y1 = y1 / label_h
                x1 = x1 / label_w
                y2 = y2 / label_h
                x2 = x2 / label_w
            if box_type == 'x1,y1,x2,y2':
                boxes.append((x1, y1, x2, y2))
            elif box_type == 'xm,ym,w,h':
                w = x2 - x1
                h = y2 - y1
                xm = x1 + w // 2
                ym = y1 + h // 2
                boxes.append((xm, ym, w, h))
            else:
                # use 'x1,y1,x2,y2' if wrong format was given
                boxes.append((x1, y1, x2, y2))

        return boxes

    def get_tag_list(self, color_coding):
        tag_list = []
        for i in range(self.num_of_frames):
            lbm = self.get_label_map_of_index(i)
            if lbm is not None:
                img = self.get_frame_of_index(i)
                if img is not None:
                    tag_list.append(
                        VideoFrameTag(video=self, frame_no=i, color_coding=color_coding)
                    )
        return tag_list
