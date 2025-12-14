import os
import dlib
from config import PREDICTOR_PATH

def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(PREDICTOR_PATH):
        raise FileNotFoundError(
            f"shape_predictor_68_face_landmarks.dat not found at:\n{PREDICTOR_PATH}\n"
            "Fix PREDICTOR_PATH or set env var DLIB_PREDICTOR."
        )
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    return detector, predictor
