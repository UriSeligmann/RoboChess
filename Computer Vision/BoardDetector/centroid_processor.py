import numpy as np
from typing import List, Tuple, Optional

from .debugger import Debugger
from .wrapper_decorators import debug_entry_exit_method

# =============================================
#      CENTROID FILTERING & ORDERING
# =============================================
class CentroidProcessor:
    """Filter, deduplicate and (when YOLO finds only three) infer chess‑board
    corner centroids so that the final four corners resemble a *square* as
    closely as the detections allow.
    """

    def __init__(self,
                 min_distance: float = 50.0,
                 debugger: Optional[Debugger] = None) -> None:
        self.min_distance = min_distance
        self.debugger = debugger

    # ------------------------------------------------------------------
    # 1     Non‑max suppression – keep at most four strong, well separated
    #       detections.
    # ------------------------------------------------------------------
    @debug_entry_exit_method(level=2)
    def filter_centroids(self,
                         centroids: List[Tuple[float, float]],
                         confidences: np.ndarray
                         ) -> List[Tuple[float, float]]:
        if not centroids:
            return []
        # sort by confidence (desc)
        idx = np.argsort(-confidences)
        cand = [centroids[i] for i in idx]

        picked: List[Tuple[float, float]] = []
        for c in cand:
            if all(np.linalg.norm(np.subtract(c, p)) > self.min_distance for p in picked):
                picked.append(c)
            if len(picked) == 4:
                break
        if self.debugger:
            self.debugger.log(f"Centroids kept after filtering: {len(picked)}", level=0)
        return picked

    # ------------------------------------------------------------------
    # 2     If only three corners are present, synthesise the fourth so the
    #       resulting quadrilateral is as close to a square (equal sides &
    #       90° angles) **in the image** as perspective allows.
    # ------------------------------------------------------------------
    def _infer_fourth_corner(self, pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pts) == 4:
            return pts
        if len(pts) != 3:
            raise ValueError("Need 3 or 4 points to infer the fourth corner.")

        arr = np.asarray(pts, dtype=np.float32)

        def _score_quad(quad: np.ndarray) -> float:
            """Lower is better –  combines side-length equality & right angles."""
            ordered = self._order_four(quad)
            tl, tr, br, bl = ordered
            # side vectors
            v_top = np.subtract(tr, tl)
            v_left = np.subtract(bl, tl)
            v_right = np.subtract(br, tr)
            v_bottom = np.subtract(br, bl)

            # side lengths
            lengths = [np.linalg.norm(v_top), np.linalg.norm(v_right),
                       np.linalg.norm(v_bottom), np.linalg.norm(v_left)]
            max_len, min_len = max(lengths), max(1e-6, min(lengths))
            ratio_penalty = abs(max_len / min_len - 1.0)          # 0 if perfectly equal

            # angle penalty – want adjacent sides ⟂ (cosθ≈0)
            cos_tl = abs(np.dot(v_top, v_left) / (max(1e-6, np.linalg.norm(v_top) * np.linalg.norm(v_left))))
            cos_tr = abs(np.dot(v_top, v_right) / (max(1e-6, np.linalg.norm(v_top) * np.linalg.norm(v_right))))
            cos_br = abs(np.dot(v_bottom, v_right) / (max(1e-6, np.linalg.norm(v_bottom) * np.linalg.norm(v_right))))
            cos_bl = abs(np.dot(v_bottom, v_left) / (max(1e-6, np.linalg.norm(v_bottom) * np.linalg.norm(v_left))))
            angle_penalty = (cos_tl + cos_tr + cos_br + cos_bl) / 4.0  # 0 if all 90°

            return ratio_penalty + angle_penalty  # equal weight

        best_quad = None
        best_score = 1e9

        # three parallelogram completions: treat every point as diagonal-opposite
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            new_pt = arr[j] + arr[k] - arr[i]
            quad = np.vstack([arr, new_pt])
            score = _score_quad(quad)
            if score < best_score:
                best_score, best_quad = score, quad

        if self.debugger:
            self.debugger.log(f"Synthetic corner chosen (score ≈ {best_score:.3f})", level=1)
        return self._order_four(best_quad)

    # ------------------------------------------------------------------
    # 3  Ordering helper used by both the public method and the
    #    square‑fitness test above.
    # ------------------------------------------------------------------
    def _order_four(self, pts: np.ndarray) -> List[Tuple[float, float]]:
        idx_by_y = np.argsort(pts[:, 1])
        top, bottom = pts[idx_by_y[:2]], pts[idx_by_y[2:]]
        top_left, top_right = top[np.argsort(top[:, 0])]
        bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
        return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]

    # ------------------------------------------------------------------
    # 4  Public entry – always returns exactly four corners ordered TL,TR,BR,BL
    # ------------------------------------------------------------------
    @debug_entry_exit_method(level=2)
    def order_centroids(self, centroids: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        centroids = self._infer_fourth_corner(list(centroids))
        pts = np.asarray(centroids, dtype=np.float32)
        return self._order_four(pts)