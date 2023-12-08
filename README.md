
# Zebrafish project!


Please download `3DZeF20Lables` and place it under `data`. 
Also download the video (or use the frame pngs) and place them under `data`.


To run the program, `python3 test.py`. During development, we use Python `3.8.10`



- tiles
  - 1x1
  - 2x2
  - 3x3
  - 4x4
 
- all are scaled down to 32x32

- types of data
  - "dataset" (original frames)
  - "removed_backgrounds" (current frame - averaged frame)
  - "prev_frames_export" (current frame - previous frame)

