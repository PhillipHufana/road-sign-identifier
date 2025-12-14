class Track:
    def __init__(self, tid, tracker):
        self.tid = tid
        self.tracker = tracker

def rect_from_pos(pos, w, h):
    x1 = int(max(0, pos.left()))
    y1 = int(max(0, pos.top()))
    x2 = int(min(w - 1, pos.right()))
    y2 = int(min(h - 1, pos.bottom()))
    return x1, y1, x2, y2
