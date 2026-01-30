import os
import time
from typing import List, Tuple

import cv2

from .action_detector import ActionDetector
from .action_logger import ActionLogger
from .camera import camera_stream
from .embed import ArcFaceEmbedder
from .face_locker import FaceLocker
from .recognize import load_identity_database, recognize_frame


def draw_label(frame: cv2.Mat, box: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]) -> None:
    """Draw a bounding box and label on the frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1 - 24), (x2, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_lock_indicator(frame: cv2.Mat, identity: str, duration: float, actions: List[str]) -> None:
    """Draw lock status and recent actions on the frame."""
    height, width = frame.shape[:2]
    
    # Draw lock status banner
    cv2.rectangle(frame, (10, 10), (width - 10, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width - 10, 100), (0, 255, 0), 2)
    
    # Lock status text
    status_text = f"LOCKED: {identity}"
    cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Duration text
    duration_text = f"Duration: {duration:.1f}s"
    cv2.putText(frame, duration_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Recent actions
    if actions:
        actions_text = f"Actions: {', '.join(actions)}"
        cv2.putText(frame, actions_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)


def draw_instructions(frame: cv2.Mat) -> None:
    """Draw usage instructions on the frame."""
    height, width = frame.shape[:2]
    instructions = [
        "Press 'q' to quit",
        "Lock will activate when target face appears",
    ]
    
    y_offset = height - 60
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def select_target_identity(database: dict) -> str:
    """
    Allow user to select which identity to lock onto.
    
    Args:
        database: Dictionary of enrolled identities
        
    Returns:
        Selected identity name
    """
    if not database:
        print("No enrolled identities found!")
        print("Please enroll at least one identity first.")
        exit(1)
    
    print("\n" + "=" * 60)
    print("FACE LOCKING SYSTEM")
    print("=" * 60)
    print("\nEnrolled identities:")
    identities = list(database.keys())
    for i, name in enumerate(identities, 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect identity to lock (1-{len(identities)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(identities):
                selected = identities[idx]
                print(f"\n✓ Target identity set to: {selected}")
                print("Starting camera...\n")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(identities)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            exit(0)


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "arcface.onnx")
    identities_dir = os.path.join(base_dir, "data", "identities")
    history_dir = os.path.join(base_dir, "action_history")

    embedder = ArcFaceEmbedder(model_path)
    database = load_identity_database(identities_dir)
    
    # Select target identity
    target_identity = select_target_identity(database)
    
    # Initialize face locking components
    locker = FaceLocker(target_identity=target_identity, lock_threshold=0.45, unlock_timeout=2.0)
    action_detector = ActionDetector()
    action_logger = ActionLogger(output_dir=history_dir)
    
    logging_started = False
    recent_actions: List[str] = []
    action_display_frames = 0

    for frame in camera_stream():
        current_time = time.time()
        
        # Recognize all faces in frame
        results = recognize_frame(frame, embedder, database)
        
        # Update face lock
        locked_box = locker.update(results, current_time)
        
        # Draw all detected faces (non-locked faces in green)
        for box, name, score in results:
            if box != locked_box:
                draw_label(frame, box, f"{name} ({score:.2f})", (0, 200, 0))
        
        # Handle locked face
        if locked_box is not None:
            # Start logging if not already started
            if not logging_started:
                log_file = action_logger.start_logging(target_identity)
                action_logger.log_lock_event("LOCK ESTABLISHED", f"Target: {target_identity}")
                print(f"✓ Lock established on {target_identity}")
                print(f"✓ Logging to: {log_file}")
                logging_started = True
            
            # Detect actions
            detected_actions = action_detector.detect_actions(frame, locked_box)
            
            # Log actions
            if detected_actions:
                action_logger.log_multiple_actions(detected_actions)
                recent_actions = detected_actions
                action_display_frames = 30  # Display for 30 frames (~1 second)
            
            # Draw locked face with special color (red)
            lock_duration = locker.get_lock_duration(current_time)
            draw_label(frame, locked_box, f"LOCKED: {target_identity}", (0, 0, 255))
            
            # Draw lock indicator
            display_actions = recent_actions if action_display_frames > 0 else []
            draw_lock_indicator(frame, target_identity, lock_duration, display_actions)
            action_display_frames = max(0, action_display_frames - 1)
        
        else:
            # Lock released
            if logging_started:
                action_logger.log_lock_event("LOCK RELEASED", f"Target lost for {locker.unlock_timeout}s")
                action_logger.stop_logging()
                print(f"✗ Lock released on {target_identity}")
                print(f"✓ Log saved to: {action_logger.get_current_file()}")
                logging_started = False
                action_detector.reset()
                recent_actions = []
        
        # Draw instructions
        draw_instructions(frame)
        
        cv2.imshow("Face Locking System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    if logging_started:
        action_logger.stop_logging()
        print(f"\n✓ Final log saved to: {action_logger.get_current_file()}")
    
    cv2.destroyAllWindows()
    print("\nFace Locking System terminated.")


if __name__ == "__main__":
    main()
