import numpy as np
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST,    R_WRIST    = 9, 10

def _kp(kps, i):
    return kps[i] if i < len(kps) else np.zeros(3)

def _vis(kp, c=0.3):
    return float(kp[2]) >= c

class PoseSOSDetector:
    def __init__(self, cfg):
        self.margin = cfg.get("raise_margin_px", 25)

    def detect(self, kps, h):
        ls,rs = _kp(kps,L_SHOULDER),_kp(kps,R_SHOULDER)
        lw,rw = _kp(kps,L_WRIST),   _kp(kps,R_WRIST)
        lu = _vis(lw) and _vis(ls) and lw[1] < ls[1]-self.margin
        ru = _vis(rw) and _vis(rs) and rw[1] < rs[1]-self.margin
        return {"is_sos": lu or ru,
                "left_arm_up": lu, "right_arm_up": ru}
