import os

# Keep this early (before importing cv2 elsewhere)
os.environ.setdefault("QT_LOGGING_RULES", "qt.core.qmimedatabase=false;qt.qpa.*=false")

PREDICTOR_PATH = os.environ.get(
    "DLIB_PREDICTOR",
    r"C:\Users\Phillip\Downloads\shape_predictor_68_face_landmarks.dat"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}

DISPLAY_MAX_W = 960
APP_TITLE = "Mask + Anonymization (dlib)"
