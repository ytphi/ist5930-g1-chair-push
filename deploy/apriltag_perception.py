#!/usr/bin/env python3
"""AprilTag detection on the G1's head camera (RealSense D435), sent over UDP to the deploy PC.

Run on the robot's PC2 (where the camera is plugged in):
  python3 apriltag_perception.py --dest 192.168.123.222 --tag-size 0.15
  python3 apriltag_perception.py --dest 127.0.0.1 --show      # with a preview window

Each UDP packet: {"stamp": unix time of the frame, "lat": seconds from frame to send (this
machine's clock only), "tags": [{"id": int, "T": 16 floats (row-major 4x4)}]}
T = tag pose in the camera optical frame (x right, y down, z forward), tag frame x right,
y up (top edge of the print), z out of the printed face. Family tag36h11.
Needs: pyrealsense2, opencv-contrib-python (>= 4.7 for cv2.aruco.ArucoDetector), numpy.
The deploy side drops packets older than 0.3 s, measured as (time since the packet arrived)
+ "lat", so the two clocks do NOT need to be in sync (PC2 syncs to the robot's main board,
which was 94 s off the laptop on 2026-11-16).
"""

import argparse
import json
import signal
import socket
import sys
import time

import cv2
import numpy as np


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dest", default="127.0.0.1", help="IP of the PC running g1_chair_deploy.py")
  ap.add_argument("--port", type=int, default=9870)
  ap.add_argument("--tag-size", type=float, default=0.15, help="black square side, meters")
  ap.add_argument("--width", type=int, default=848)
  ap.add_argument("--height", type=int, default=480)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--show", action="store_true")
  ap.add_argument("--record", help="save the color stream to this mp4 (for checking the view)")
  args = ap.parse_args()

  import pyrealsense2 as rs
  pipe, cfg = rs.pipeline(), rs.config()
  cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
  prof = pipe.start(cfg)
  intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
  dist = np.array(intr.coeffs[:5])
  print(f"color {intr.width}x{intr.height} fx={intr.fx:.1f} fy={intr.fy:.1f}")

  params = cv2.aruco.DetectorParameters()
  params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
  det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
  s = args.tag_size / 2
  obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)  # TL TR BR BL

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  writer = None
  # SIGTERM (pkill) exits through the finally block, so a --record mp4 is finalized. Needed
  # because a nohup'd background process ignores SIGINT.
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  n, t0 = 0, time.time()
  try:
    while True:
      frames = pipe.wait_for_frames()
      img = np.asanyarray(frames.get_color_frame().get_data())
      stamp = time.time()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      corners, ids, _ = det.detectMarkers(gray)
      tags = []
      if ids is not None:
        for c, i in zip(corners, ids.flatten()):
          ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
          if not ok:
            continue
          R, _ = cv2.Rodrigues(rvec)
          T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.flatten()
          tags.append({"id": int(i), "T": T.reshape(-1).round(5).tolist()})
          if args.show or args.record:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, s)
            cv2.putText(img, f"{i}: {np.linalg.norm(tvec):.2f}m", tuple(c[0][0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      sock.sendto(json.dumps({"stamp": stamp, "lat": time.time() - stamp, "tags": tags}).encode(),
                  (args.dest, args.port))
      if args.record:
        if writer is None:
          writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, img.shape[1::-1])
        writer.write(img)
      if args.show:
        cv2.imshow("apriltags", img)
        if cv2.waitKey(1) == 27:
          break
      n += 1
      if n % 150 == 0:
        print(f"{n / (time.time() - t0):.1f} Hz, tags now: {[t['id'] for t in tags]}")
  finally:
    pipe.stop()
    if writer is not None:
      writer.release()


if __name__ == "__main__":
  main()
