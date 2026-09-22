"""Engine + Viewer + BBoxOverlay with simulated inference results.

Demonstrates:
- BBoxOverlay for drawing bounding boxes on event frames
- Simulating inference by generating random bboxes in a separate thread
- Thread-safe bbox updates via BBoxOverlay.update()
- Headless viewer for portability (no display window)

Run with: python examples/06_bbox_overlay.py [--duration 5]
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time

from frames2py import Engine, Viewer
from frames2py.bench.synthetic import profile_stream, PROFILES
from frames2py.display.overlays import BBox, BBoxOverlay


def simulate_inference(
    bbox_overlay: BBoxOverlay,
    sensor_size: tuple[int, int],
    stop_event: threading.Event,
    interval: float = 0.2,
) -> None:
    """Generate random bounding boxes and push to overlay (simulates inference thread)."""
    w, h = sensor_size
    colors = [
        (0, 255, 0),   # Green
        (0, 255, 255),  # Yellow
        (255, 0, 0),    # Blue
        (0, 165, 255),  # Orange
    ]
    labels = ["car", "person", "bike", "object"]

    while not stop_event.is_set():
        n_boxes = random.randint(1, 4)
        boxes = []
        for _ in range(n_boxes):
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = min(x1 + random.randint(40, 120), w)
            y2 = min(y1 + random.randint(40, 120), h)
            label = random.choice(labels)
            confidence = round(random.uniform(0.7, 0.99), 2)
            color = random.choice(colors)
            boxes.append(
                BBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, color=color, confidence=confidence)
            )
        bbox_overlay.update(boxes)
        stop_event.wait(timeout=interval)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run synthetic stream with BBoxOverlay (simulated inference)."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Stream duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="medium",
        help="Workload profile (default: medium)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    sensor_size = profile.sensor_size

    print("BBoxOverlay demo - simulated inference from separate thread")
    print(f"  Profile: {args.profile}, Duration: {args.duration}s")
    print("  Bboxes are pushed every ~200ms from a background thread")
    print()

    engine = Engine(sensor_size=sensor_size, kernel="event_count")
    bbox_overlay = BBoxOverlay()
    viewer = Viewer(engine, backend="headless", fps=30, colormap="viridis")
    viewer.add_overlay(bbox_overlay)
    viewer.start()

    stop_event = threading.Event()
    inference_thread = threading.Thread(
        target=simulate_inference,
        args=(bbox_overlay, sensor_size, stop_event),
        daemon=True,
    )
    inference_thread.start()

    start = time.monotonic()

    try:
        for batch in profile_stream(
            args.profile,
            duration_sec=args.duration,
            paced=True,
        ):
            engine.ingest(batch)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    stop_event.set()
    inference_thread.join(timeout=1.0)

    elapsed = time.monotonic() - start
    engine.stop()
    viewer.stop()

    stats = engine.stats
    print()
    print("=== Complete ===")
    print(f"  Duration:      {elapsed:.2f}s")
    print(f"  Events:        {stats.events_ingested:,}")
    print(f"  Frames shown:  {viewer.frames_shown}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
