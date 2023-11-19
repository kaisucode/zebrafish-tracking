
import cv2
from matplotlib import cm
import json
import math

## configs
numFish = 5
videoSource = "data/ZebraFish-04-raw.webm"
labelSource = "data/3DZeF20Lables/train/ZebraFish-04/gt/gt.txt"

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



def readLables(labelSource):
    lines = None
    with open(labelSource) as file:
        lines = [line.rstrip() for line in file]

    camT = []
    camT.append((680, 135))
    camT.append((1962, 151))
    camT.append((1969, 1460))
    camT.append((626, 1431))

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

    #  topLeft = (minX, minY)
    #  bottomRight = (maxX, maxY)

    xStep = (maxX - minX) // M
    yStep = (maxY - minY) // N

    tiles = []
    tileInfo = []
    for i in range(minX, maxX - xStep, xStep):
        for j in range(minY, maxY - yStep, yStep):
            tiles.append(originalImg[j:j+yStep,i:i+xStep,:])
            # [(yMin, xMin), (yMax, xMax)]
            tileInfo.append([(j, i), (j + yStep, i + xStep)])

    assert(len(tiles) == M * N)
    return tiles, tileInfo


capture, totalFrames, height, width = readVideo(videoSource)
lines, camT = readLables(labelSource)

for i, corner in enumerate(camT): 
    newX = (corner[0] + width) // 2
    newY = (corner[1]) // 2
    camT[i] = (newX, newY)


frameNr = 0

fishColors = getFishColors(numFish)
history = [[] for _ in range(numFish)]
videoWrite = cv2.VideoWriter('viz.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         10, (width, height))


print("read in lines: ", str(len(lines)))
print("total frames: ", totalFrames)
print("height: ", height)
print("width: ", width)




while True:
    success, frame = capture.read()
    if success:
        #  cv2.imwrite("frames/frame" + str(frameNr) + ".png", frame)

        for dot in camT: 
            #  print(dot)
            dot = (dot[0] // 2 + width // 2, dot[1] // 2)
            drawCircle(frame, center=dot, radius=10, thickness=3, color=(0, 0, 0))
        tiles, tileInfo = createTiles(camT, frame, M=2, N=2)
        cv2.imshow("tiles[0]", tiles[0])


        # check if fish in tile
        tileFishCount = [0] * numFish
        for i in range(numFish): 

            annotatedData = lines[frameNr * numFish + i].split(",")
            x_top, y_top = int(annotatedData[5]) // 2 + width // 2, int(annotatedData[6]) // 2

            top_coor = (x_top, y_top)

            cv2.circle(frame, center=top_coor, radius=8, color=fishColors[i], thickness=3)

            for tileId, aTile in enumerate(tileInfo): 
                if aTile[0][0] <= y_top and aTile[1][0] >= y_top and aTile[0][1] <= x_top and aTile[1][1] >= x_top: 
                    tileFishCount[tileId] += 1

        print(tileFishCount)

        cv2.imshow("Video", frame)
        #  videoWrite.write(frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        frameNr += 1

    if frameNr == totalFrames: 
        break

#  capture.release()
#  videoWrite.release()
cv2.destroyAllWindows()

