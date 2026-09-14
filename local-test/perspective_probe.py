"""Offline perspective probe; this never calls Render, Modal, R2, or Vercel."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local-test" / "perspective-probe-image2.jpg"
source = cv2.imread(str(ROOT / "public/reference/image2.jpeg"))
height, width = source.shape[:2]
# Approximate floor-plane corners visible in the supplied angled flooring reference.
# These are deliberately only for a local diagnostic, not the production workflow.
points = np.float32([[0.19*width, 0.07*height], [0.97*width, 0.14*height], [0.99*width, 0.94*height], [0.28*width, 0.94*height]])
physical_width = max(np.linalg.norm(points[1]-points[0]), np.linalg.norm(points[2]-points[3]))
physical_height = max(np.linalg.norm(points[3]-points[0]), np.linalg.norm(points[2]-points[1]))
scale = 1000 / max(physical_width, physical_height)
out_w, out_h = round(physical_width*scale), round(physical_height*scale)
destination = np.float32([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]])
matrix = cv2.getPerspectiveTransform(points, destination)
warped = cv2.warpPerspective(source, matrix, (out_w,out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
original = cv2.resize(source, (640, 480), interpolation=cv2.INTER_AREA)
preview = cv2.resize(warped, (640, 480), interpolation=cv2.INTER_AREA)
canvas = Image.new('RGB',(1280,530),'white'); canvas.paste(Image.fromarray(cv2.cvtColor(original,cv2.COLOR_BGR2RGB)),(0,30)); canvas.paste(Image.fromarray(cv2.cvtColor(preview,cv2.COLOR_BGR2RGB)),(640,30))
d = ImageDraw.Draw(canvas); d.text((12,8),'Original reference',fill='black'); d.text((652,8),f'Local homography probe ({out_w} x {out_h})',fill='black')
canvas.save(OUT,quality=94)
print(OUT)
print('source=', source.shape[1], 'x', source.shape[0], 'output=', out_w, 'x', out_h)
