from os.path import join
import numpy as np

from .imprt import preset_import
from .vis.pt import scatter_on_img


class LucasKanadeTracker():
    """Lucas Kanade Tracker.

    Attributes:
        frames (list(numpy.array)): Grayscale.
        pts (numpy.array)
        lk_params (dict)
        backtrack_thres (float)
        tracks (list(numpy.array)): Positions of tracks from the :math:`i`-th
            to :math:`(i+1)`-th frame. Arrays are of shape N-by-2.

            .. code-block:: none

                +------------>
                |       tracks[:, 1]
                |
                |
                v tracks[:, 0]

        can_backtrack (list(numpy.array)): Whether each track can be
            back-tracked to the previous frame. Arrays should be Boolean.
        is_lost (list(numpy.array)): Whether each track is lost in this frame.
            Arrays should be Boolean.
    """
    def __init__(self, frames, pts, backtrack_thres=1, lk_params=None):
        """
        Args:
            frames (list(numpy.array)): Frame images in order. Arrays are either
                H-by-W or H-by-W-by-3, and will be converted to grayscale.
            pts (array_like): Points to track in the first frame. Of shape
                N-by-2.

                .. code-block:: none

                    +------------>
                    |       pts[:, 1]
                    |
                    |
                    v pts[:, 0]

            backtrack_thres (float, optional): Largest pixel deviation in the
                :math:`x` or :math:`y` direction of a successful backtrack.
            lk_params (dict, optional): Keyword parameters for
                :func:`cv2.calcOpticalFlowPyrLK`.
        """
        cv2 = preset_import('cv2', assert_success=True)
        frames_gs = []
        for img in frames:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames_gs.append(img)
        self.frames = frames_gs
        self.pts = np.array(pts)
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 12,
            'criteria': (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        if lk_params is not None:
            # Overwrite with whatever is user-provided
            for key, val in lk_params.items():
                self.lk_params[key] = val
        self.backtrack_thres = backtrack_thres
        self.tracks = []
        self.can_backtrack = []
        self.is_lost = []

    def run(self, constrain=None):
        """Runs tracking.

        Args:
            constrain (function, optional): Function applied to tracks before
                being fed to the next round. It should take in an N-by-2
                arrays as well as the current workspace (as a dictionary) and
                return another array.
        """
        cv2 = preset_import('cv2', assert_success=True)
        for fi in range(0, len(self.frames) - 1):
            f0, f1 = self.frames[fi], self.frames[fi + 1]
            if fi == 0:
                p0 = self._my2klt(self.pts)
            # Track with forward flow
            p1, not_lost, err = cv2.calcOpticalFlowPyrLK(
                f0, f1, p0, None, **self.lk_params)
            is_lost = (1 - not_lost.ravel()).astype(bool)
            err = err.ravel()
            # Check quality by back-tracking
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                f1, f0, p1, None, **self.lk_params)
            can_backtrack = \
                abs(p0 - p0r).reshape(-1, 2).max(-1) < self.backtrack_thres
            # Continue tracking these points or impose some constraints
            if constrain is None:
                p0 = p1
            else:
                pts = self._klt2my(p1)
                pts = constrain(pts, locals())
                p0 = self._my2klt(pts)
            self.tracks.append(self._klt2my(p0))
            self.can_backtrack.append(can_backtrack)
            self.is_lost.append(is_lost)

    def vis(self, out_dir, marker_bgr=(0, 0, 255)):
        """Visualizes results.

        Args:
            out_dir (str): Output directory.
            marker_bgr (tuple, optional): Marker BGR color.

        Writes
            - Each frame with tracked points marked out.
        """
        for fi in range(0, len(self.frames) - 1):
            im = self.frames[fi + 1]
            pts = self.tracks[fi]
            scatter_on_img(
                im, pts, size=6, bgr=marker_bgr,
                outpath=join(out_dir, '%04d.png' % (fi + 1)))

    @staticmethod
    def _my2klt(pts):
        """Reshapes

        .. code-block:: none

            +------------>
            |       pts[:, 1]
            |
            |
            v pts[:, 0]

        into

        .. code-block:: none

            +------------>
            |       pts[:, 0, 0]
            |
            |
            v pts[:, 0, 1]
        """
        return np.expand_dims(
            np.vstack((pts[:, 1],
                       pts[:, 0])).T, 1).astype(np.float32)

    @staticmethod
    def _klt2my(pts):
        """Inverse of :func:`_my2klt`"""
        return pts.reshape(-1, 2)[:, ::-1]
