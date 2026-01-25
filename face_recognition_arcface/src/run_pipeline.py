import os
from typing import Tuple

import cv2

from .camera import camera_stream
from .embed import ArcFaceEmbedder
from .recognize import load_identity_database, recognize_frame


def draw_label(frame: cv2.Mat, box: Tuple[int, int, int, int], text: str) -> None:
    """Draw a bounding box and label on the frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.rectangle(frame, (x1, y1 - 24), (x2, y1), (0, 200, 0), -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "arcface.onnx")
    identities_dir = os.path.join(base_dir, "data", "identities")

    embedder = ArcFaceEmbedder(model_path)
    database = load_identity_database(identities_dir)

    for frame in camera_stream():
        results = recognize_frame(frame, embedder, database)
        for box, name, score in results:
            draw_label(frame, box, f"{name} ({score:.2f})")

        cv2.imshow("ArcFace Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
