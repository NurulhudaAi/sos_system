import cv2, time

def preprocess(frame, w, h):
    if frame.shape[1]!=w or frame.shape[0]!=h:
        frame = cv2.resize(frame,(w,h))
    return frame

class Visualizer:
    def __init__(self):
        self._t = []

    def fps(self, frame):
        now = time.time()
        self._t = [t for t in self._t if now-t<1.0]
        self._t.append(now)
        cv2.putText(frame,f"FPS {len(self._t)}",(10,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)

    def banner(self, frame, atype):
        c = {"fall_warning":(0,165,255),"fall":(0,0,220),"hand_sos":(0,200,0)}.get(atype,(0,0,220))
        l = {"fall_warning":"FALL WARNING","fall":"FALL DETECTED","hand_sos":"SILENT SOS HAND"}.get(atype,"ALERT")
        h,w = frame.shape[:2]
        cv2.rectangle(frame,(0,h-50),(w,h),c,-1)
        cv2.putText(frame,f"!!! {l} !!!",(20,h-14),
                    cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)

    def hand_state(self, frame, state):
        s = ["—","open palm","thumb in","SOS!"]
        c = [(150,150,150),(255,200,0),(0,165,255),(0,220,0)]
        cv2.putText(frame,f"Hand: {s[state]}",(10,56),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,c[state],1)
