
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import math
import numpy as np

## configs
numFish = 5
videoSource = "data/ZebraFish-04-raw.webm"
labelSource = "data/3DZeF20Lables/train/ZebraFish-04/gt/gt.txt"
exportFilename = "export/three-by-three-no-downscale"
M = 3
N = 3

def getFishColors(numFish): 
    cmap = cm.get_cmap('viridis', numFish)
    colors = []
    for i in range(numFish): 
        colors.append(tuple([cmap.colors[i][0] * 255, cmap.colors[i][1] * 255, cmap.colors[i][2] * 255]))

    return colors

def drawCircle(frame, center, color, radius=8, thickness=3): 
    cv2.circle(frame, center=center, radius=radius, color=color, thickness=thickness)

def readVideo(videoSource): 
    capture = cv2.VideoCapture(videoSource)
    totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    return capture, totalFrames, height, width



def readLables(labelSource, width):
    lines = None
    with open(labelSource) as file:
        lines = [line.rstrip() for line in file]

    camT = []
    camT.append((680, 135))
    camT.append((1962, 151))
    camT.append((1969, 1460))
    camT.append((626, 1431))

    for i, corner in enumerate(camT): 
        newX = (corner[0] + width) // 2
        newY = (corner[1]) // 2
        camT[i] = (newX, newY)

    return lines, camT

def createTiles(camT, originalImg, M=2, N=2): 
    # fix corners
    minX, maxX = math.inf, -math.inf
    minY, maxY = math.inf, -math.inf
    for corner in camT: 
        minX = min(minX, corner[0])
        maxX = max(maxX, corner[0])
        minY = min(minY, corner[1])
        maxY = max(maxY, corner[1])

    topLeft = (minX, minY)
    bottomRight = (maxX, maxY)
    cropped = originalImg[minY:maxY, minX:maxX]

    xStep = (maxX - minX - 1) // M
    yStep = (maxY - minY - 1) // N

    tiles = []
    tileInfo = []
    for j in range(minY, maxY - yStep, yStep):
        for i in range(minX, maxX - xStep, xStep):
            tiles.append(originalImg[j:j+yStep,i:i+xStep,:])
            # [(yMin, xMin), (yMax, xMax)]
            tileInfo.append([(j, i), (j + yStep, i + xStep)])

    #  print(len(tiles))
    assert(len(tiles) == M * N)
    return tiles, tileInfo, cropped


def visualize(frame, cropped, tiles, tileFishCount):
    fig1, axs1 = plt.subplots(2, 1)
    fig1.tight_layout(pad=3.0)
    axs1[0].imshow(frame)
    axs1[1].imshow(cropped)
    axs1[0].set_title("Original frame")
    axs1[1].set_title("Cropped frame")

    fig2, axs2 = plt.subplots(M, N)
    fig2.tight_layout(pad=3.0)
    ax2 = axs2.ravel()
    for i in range(len(tiles)):
        ax2[i].imshow(tiles[i])
        ax2[i].set_title(str(tileFishCount[i]) + " fish head(s) in tile")

    plt.show()

capture, totalFrames, height, width = readVideo(videoSource)
lines, camT = readLables(labelSource, width)

frameNr = 0
fishColors = getFishColors(numFish)
history = [[] for _ in range(numFish)]
videoWrite = cv2.VideoWriter('viz.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         10, (width, height))


print("read in lines: ", str(len(lines)))
print("total frames: ", totalFrames)
print("(height, width): ", height, ", ", width)

dataset = []
labels = []

while True:
    success, frame = capture.read()
    if success:
        #  cv2.imwrite("frames/frame" + str(frameNr) + ".png", frame)

        #  for dot in camT: 
        #      dot = (dot[0] // 2 + width // 2, dot[1] // 2)
        #      drawCircle(frame, center=dot, radius=10, thickness=3, color=(0, 0, 0))
        tiles, tileInfo, cropped = createTiles(camT, frame, M=M, N=N)

        # check if fish in tile
        tileFishCount = [0] * (M * N)
        for i in range(numFish): 

            annotatedData = lines[frameNr * numFish + i].split(",")
            x_top, y_top = int(annotatedData[5]) // 2 + width // 2, int(annotatedData[6]) // 2
            top_coor = (x_top, y_top)
            #  cv2.circle(frame, center=top_coor, radius=8, color=fishColors[i], thickness=5)

            for tileId, aTile in enumerate(tileInfo): 
                if aTile[0][0] <= y_top and aTile[1][0] >= y_top and aTile[0][1] <= x_top and aTile[1][1] >= x_top: 
                    tileFishCount[tileId] += 1

        #  cv2.imshow("Video", frame)
        #  videoWrite.write(frame)

        #  visualize(frame, cropped, tiles, tileFishCount)
        #  break
        #  if cv2.waitKey(20) & 0xFF == ord('q'):
        #      break

        for tileId, aTile in enumerate(tiles): 
            dataset.append(aTile)
            labels.append(tileFishCount[tileId])

        frameNr += 1

    if frameNr == totalFrames: 
        break

dataset = np.asarray(dataset)
labels = np.asarray(labels)

print("Saving data to ", exportFilename, "...")
np.savez(exportFilename, dataset=dataset, labels=labels)
print("Saved data!")

#  capture.release()
#  videoWrite.release()
cv2.destroyAllWindows()

