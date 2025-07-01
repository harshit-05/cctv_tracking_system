# trackers/bytetrack_wrapper.py

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
import argparse
import numpy as np
import torch

class TrackByDetection:
    def __init__(self, conf_thresh=0.3, img_size=640, iou_thresh=0.5, skip_interval=1, device='cuda'):
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.iou_thresh = iou_thresh
        self.skip_interval = skip_interval
        self.frame_id = 0
        self.timer = Timer()
        self.device = device

        args = argparse.Namespace()
        args.track_thresh = self.conf_thresh
        #args.thresh = self.conf_thresh
        args.track_buffer = 30
        args.match_thresh = self.iou_thresh
        args.mot20 = False
        """
        self.tracker = BYTETracker(
            track_thresh=conf_thresh,
            match_thresh=iou_thresh,
            frame_rate=30  # default FPS
        )
        """
        self.tracker = BYTETracker(args)

    def update(self, detections, frame):
        self.frame_id += 1
        if self.frame_id % self.skip_interval != 0:
            return []

        self.timer.tic()
        if detections:
            detections_tensor = torch.from_numpy(np.array(detections, dtype=float)).to(self.device)
            detections_tensor = torch.empty((0, 6)).to(self.device)
        #online_targets = self.tracker.update(empty_tensor, frame.shape[:2], (self.img_size, self.img_size))
        #detections_np = np.array(detections, dtype=float)
        online_targets = self.tracker.update(detections_tensor, frame.shape[:2], (self.img_size, self.img_size))
        self.timer.toc()

        tracks = []
        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            bbox = [int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])]
            tracks.append((tid, bbox))
        return tracks

    def get_fps(self):
        return self.timer.average_time
 






