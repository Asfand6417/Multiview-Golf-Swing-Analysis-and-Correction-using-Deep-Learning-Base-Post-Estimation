import math

def is_bad_swing(landmarks, w, h):
    # Example rule: check elbow angle
    def get_angle(p1, p2, p3):
        x1, y1 = landmarks[p1]
        x2, y2 = landmarks[p2]
        x3, y3 = landmarks[p3]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        return angle

    # Example: check left elbow angle
    elbow_angle = get_angle(11, 13, 15)
    if elbow_angle > 160 or elbow_angle < 30:
        return True
    return False