# ================================================================
# neurapose_backend/tracker/modulos/botsort.py
# ================================================================

import numpy as np
from colorama import Fore
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT as BOTSORT_ORIGINAL
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.gmc import GMC

from config_master import BOT_SORT_CONFIG
from tracker.modulos.reid import CustomReID


class CustomBoTSORT(BOTSORT_ORIGINAL):
    def __init__(self, frame_rate=30):
        class Args:
            pass

        args = Args()
        for k, v in BOT_SORT_CONFIG.items():
            setattr(args, k, v)

        # Nao usamos o ReID nativo, usamos o nosso encoder
        args.with_reid = False

        super().__init__(args, frame_rate)
        self.args = args

        # GMC (motion compensation)
        self.gmc = GMC(method=args.gmc_method)

        # Custom ReID
        self.encoder = CustomReID(args.model)

    def get_kalmanfilter(self):
        return KalmanFilterXYWH()

    def init_track(self, results, img=None):
        if len(results) == 0:
            return []

        bboxes = results.xywh.cpu().numpy()
        dets = np.hstack([bboxes, np.arange(len(bboxes)).reshape(-1, 1)])

        feats = self.encoder(img, dets)

        tracks = [
            BOTrack(xywh, s, c, f)
            for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, feats)
        ]

        return tracks

    def get_dists(self, tracks, detections):
        emb = matching.embedding_distance(tracks, detections)
        iou = matching.iou_distance(tracks, detections)

        alpha = self.args.appearance_thresh
        dists = emb * alpha + iou * (1.0 - alpha)

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        return dists
