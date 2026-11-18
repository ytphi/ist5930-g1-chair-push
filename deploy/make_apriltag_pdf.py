#!/usr/bin/env python3
"""Make a printable PDF of the AprilTags in tags.yaml (tag36h11, vector, exact size).

  python deploy/make_apriltag_pdf.py                  # -> deploy/apriltags/apriltags_tag36h11.pdf
  python deploy/make_apriltag_pdf.py --paper a4

Page 1 is a to-scale map of where the floor tags go. Then one tag per page, with mounting
instructions and a 100 mm scale bar for checking that the printer did not rescale.
Print at 100% / "Actual size" (never "Fit to page").
Needs: opencv-contrib-python (tag bit patterns), reportlab, pyyaml.
"""

import argparse
import os

import cv2
import yaml
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def tag_bits(tag_id):
  """8x8 cells (black border + 6x6 data), 0 = black, 255 = white, row 0 = top."""
  return cv2.aruco.generateImageMarker(DICT, tag_id, 8, borderBits=1)


def draw_tag(c, tag_id, cx, cy, side):
  bits = tag_bits(tag_id)
  cell = side / 8.0
  x0, y_top = cx - side / 2, cy + side / 2
  c.setFillColorRGB(0, 0, 0)
  c.setStrokeColorRGB(0, 0, 0)
  for r in range(8):
    for col in range(8):
      if bits[r, col] < 128:
        # Slight overlap avoids hairline gaps between cells in some PDF viewers/printers.
        c.rect(x0 + col * cell, y_top - (r + 1) * cell, cell + 0.05, cell + 0.05, stroke=0, fill=1)


def arrow_up(c, x, y, length, label, label_right=False):
  c.setLineWidth(2)
  c.line(x, y, x, y + length)
  p = c.beginPath(); p.moveTo(x, y + length + 6); p.lineTo(x - 5, y + length - 4); p.lineTo(x + 5, y + length - 4); p.close()
  c.drawPath(p, fill=1, stroke=0)
  c.setFont("Helvetica-Bold", 11)
  if label_right:
    c.drawString(x + 8, y + length / 2 - 2, label)
  else:
    c.drawCentredString(x, y + length + 10, label)


def scale_bar(c, x, y):
  c.setLineWidth(1.5)
  c.line(x, y, x + 100 * mm, y)
  for k in range(11):
    h = 4 * mm if k % 5 == 0 else 2 * mm
    c.line(x + k * 10 * mm, y, x + k * 10 * mm, y + h)
  c.setFont("Helvetica", 8)
  c.drawString(x, y - 10, "0")
  c.drawRightString(x + 100 * mm, y - 10, "100 mm  - measure this: if it is not 100 mm, the printer rescaled")


def map_page(c, W, H, cfg):
  c.setFont("Helvetica-Bold", 16)
  c.drawString(20 * mm, H - 20 * mm, "CGI-112 chair push: floor-tag layout (top view)")
  c.setFont("Helvetica", 9)
  lines = [
    "Table frame: origin = center of the table footprint, +x = into the table, +y = left when facing the table.",
    "Tape each floor tag flat, printed face up, with its TOP arrow pointing at the table (+x).",
    "Measure from the table's front-edge center: tag x = (distance in front of the edge) - 0.26 m.",
    "After taping, measure the real positions and update floor_tags in deploy/tags.yaml.",
  ]
  for i, t in enumerate(lines):
    c.drawString(20 * mm, H - (28 + 5 * i) * mm, t)
  # Map at 1:20. Table frame x (toward table) is drawn upward on the page.
  s = 1.0 / 20.0
  ox, oy = W / 2, H - 70 * mm          # table frame origin on the page
  to_page = lambda x, y: (ox - y * 1000 * s * mm, oy + x * 1000 * s * mm)  # noqa: E731
  # Table 0.52 deep x 1.5 wide.
  x0, y0 = to_page(-0.26, 0.75); x1, y1 = to_page(0.26, -0.75)
  c.setFillColorRGB(0.85, 0.75, 0.6); c.setStrokeColorRGB(0, 0, 0)
  c.rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0), fill=1, stroke=1)
  c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 9)
  cx, cy = to_page(0, 0); c.drawCentredString(cx, cy - 3, "TABLE (1.5 x 0.52 m)")
  ex, ey = to_page(-0.26, 0); c.circle(ex, ey, 2, fill=1)
  c.setFont("Helvetica", 7); c.drawString(ex + 4, ey - 9, "front-edge center (x = -0.26)")
  # Floor tags.
  ts = cfg["tag_size"]
  for tid, (x, y) in cfg["floor_tags"].items():
    px, py = to_page(x, y)
    half = ts * 1000 * s * mm / 2
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.rect(px - half, py - half, 2 * half, 2 * half, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0); c.setFont("Helvetica-Bold", 8)
    c.drawString(px + half + 2, py + 2, f"ID {tid}")
    c.setFont("Helvetica", 7)
    c.drawString(px + half + 2, py - 7, f"x={x:+.2f}  y={y:+.2f}")
    c.drawString(px + half + 2, py - 15, f"{-(x + 0.26):.2f} m in front of edge, {abs(y):.2f} m {'left' if y > 0 else 'right' if y < 0 else 'center'}")
  # Chair start area hint and robot side.
  c.setFont("Helvetica-Oblique", 8)
  rx, ry = to_page(-3.0, 0); c.drawCentredString(rx, ry, "robot / chair start area (this side)")
  arrow_up(c, 25 * mm, oy - 40 * mm, 25 * mm, "+x (toward table)")
  # Scale.
  c.setFont("Helvetica", 8)
  bx, by = 20 * mm, 25 * mm
  c.setLineWidth(1.5); c.line(bx, by, bx + 50 * mm, by)
  c.drawString(bx, by + 4, "1 m at 1:20 (= 50 mm)")
  c.drawString(20 * mm, 15 * mm, "Chair tag (ID 0): back face of the backrest, centered, top edge up, center "
               f"{cfg['chair_tags'][0][2] * 100:.0f} cm above the floor.")
  c.showPage()


def tag_page(c, W, H, tag_id, side, title, lines, top_label):
  # Layout from the top: header (to 37 mm), TOP arrow, cut box (tag + 20 mm white border),
  # notes, scale bar. Fits Letter and A4 with a 150 mm tag.
  c.setFont("Helvetica-Bold", 18)
  c.drawString(18 * mm, H - 14 * mm, f"AprilTag tag36h11  ID {tag_id}")
  c.setFont("Helvetica-Bold", 11)
  c.drawString(18 * mm, H - 21 * mm, title)
  c.setFont("Helvetica", 8.5)
  for i, t in enumerate(lines):
    c.drawString(18 * mm, H - (27 + 4.5 * i) * mm, t)
  quiet = 20 * mm                      # > 1 cell (18.75 mm) of white around the black square
  box = side + 2 * quiet
  top = H - 52 * mm
  cx, cy = W / 2, top - box / 2
  arrow_up(c, cx, top + 2 * mm, 9 * mm, top_label, label_right=True)
  c.setDash(4, 3); c.setLineWidth(0.5); c.setStrokeColorRGB(0.5, 0.5, 0.5)
  c.rect(cx - box / 2, cy - box / 2, box, box, stroke=1, fill=0)
  c.setDash(); c.setStrokeColorRGB(0, 0, 0)
  draw_tag(c, tag_id, cx, cy, side)
  c.setFont("Helvetica", 7.5)
  y = cy - box / 2 - 4 * mm
  c.drawString(cx - box / 2, y, "dashed line = cut here (keep the white border)")
  c.drawString(cx - box / 2, y - 4 * mm,
               f"black square = {side / mm:.0f} mm: measure it after printing and set tag_size in deploy/tags.yaml")
  scale_bar(c, cx - box / 2, y - 14 * mm)
  c.showPage()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tags", default=os.path.join(HERE, "tags.yaml"))
  ap.add_argument("--paper", choices=["letter", "a4"], default="letter")
  ap.add_argument("--out", default=os.path.join(HERE, "apriltags", "apriltags_tag36h11.pdf"))
  args = ap.parse_args()
  cfg = yaml.safe_load(open(args.tags))
  side = float(cfg["tag_size"]) * 1000 * mm
  size = letter if args.paper == "letter" else A4
  W, H = size
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  c = canvas.Canvas(args.out, pagesize=size)
  c.setTitle("AprilTags tag36h11 for the G1 chair push")
  map_page(c, W, H, cfg)
  for tid, (x, y, z) in cfg["chair_tags"].items():
    tag_page(c, W, H, int(tid), side, "CHAIR tag: back of the backrest",
             ["Tape flat on the BACK face of the backrest (the side the robot stands on), centered left-right.",
              f"Top edge UP. Tag center {z * 100:.0f} cm above the floor.",
              "Keep it flat: a curved tag gives a wrong angle. Glue it to a piece of cardboard first."],
             "TOP (up)")
  for tid, (x, y) in cfg["floor_tags"].items():
    tag_page(c, W, H, int(tid), side, f"FLOOR tag: table frame x = {x:+.2f} m, y = {y:+.2f} m",
             [f"Flat on the floor, face up, {-(x + 0.26):.2f} m in front of the table's front edge, "
              f"{abs(y):.2f} m {'left' if y > 0 else 'right' if y < 0 else 'of center (centered)'} of its center.",
              "TOP arrow points at the table. Tape all four edges down so nobody kicks it out of place.",
              "Laminate it or use matte paper: glare from ceiling lights hides the tag."],
             "TOP (point at the table)")
  c.save()
  print("wrote", args.out)


if __name__ == "__main__":
  main()
