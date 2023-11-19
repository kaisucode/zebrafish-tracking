
import cv2
from matplotlib import cm
import json


## configs
numFish = 5
videoSource = "data/ZebraFish-04-raw.webm"
labelSource = "data/3DZeF20Lables/train/ZebraFish-04/gt/gt.txt"

#  camTRefSource = "data/3DZeF20Lables/train/ZebraFish-04/camT_references.json"


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
    #  camT.append((680.0, 135.0))
    #  camT.append((1962.0, 151.0))
    #  camT.append((1969.0, 1460.0))
    #  camT.append((626.0, 1431.0))
    #  file = open(camTRefSource)
    #  data = json.load(file, cls=JSONWithCommentsDecoder)
    #  print(data)

    #  for i in range(4): 
    #      coor = (data[i]["camera"]["x"], data[i]["camera"]["x"])
    #      camT.append(coor)
    #  file.close()

    return lines, camT



capture, totalFrames, height, width = readVideo(videoSource)
lines, camT = readLables(labelSource)

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

# grab 



while True:
    success, frame = capture.read()
    if success:
        #  cv2.imwrite("frames/frame" + str(frameNr) + ".png", frame)


        #  data_tmp = lines[frameNr * numFish].split(",")
        #  t_left = int(data_tmp[7]) // 2
        #  t_top = int(data_tmp[8]) // 2
        #  t_width = int(data_tmp[9])
        #  t_height = int(data_tmp[10])
        #  upper_left = (668.0, 84.0)

        for dot in camT: 
            print(dot)
            dot = (dot[0] // 2 + width // 2, dot[1] // 2)
            drawCircle(frame, center=dot, radius=10, thickness=3, color=(0, 0, 0))

        for i in range(numFish): 

            # draw history
            for pastFrame in history[i]: 
                print(pastFrame)
                drawCircle(frame, pastFrame[0], fishColors[i], 1, 2)
                drawCircle(frame, pastFrame[1], fishColors[i], 1, 2)


            annotatedData = lines[frameNr * numFish + i].split(",")
            x_front, y_front = int(annotatedData[12]) // 2, int(annotatedData[13]) // 2
            x_top, y_top = int(annotatedData[5]) // 2 + width // 2, int(annotatedData[6]) // 2

            front_coor = (x_front, y_front)
            top_coor = (x_top, y_top)

            cv2.circle(frame, center=front_coor, radius=8, color=fishColors[i], thickness=3)
            cv2.circle(frame, center=top_coor, radius=8, color=fishColors[i], thickness=3)
            history[i].append([front_coor, top_coor])

        cv2.imshow("Video", frame)
        #  videoWrite.write(frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        frameNr += 1

#  capture.release()
#  videoWrite.release()
cv2.destroyAllWindows()

