
import cv2
from matplotlib import cm


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

capture = cv2.VideoCapture(videoSource)
totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

lines = None
with open(labelSource) as file:
    lines = [line.rstrip() for line in file]

frameNr = 0
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

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
            #  x, y = int(annotatedData[5]), int(annotatedData[6])
            #  print("coordiantes: ", x, ", ", y)
            #  cv2.circle(frame, center=(x_front, y_front), radius=10, color=(0, 0,255), thickness=3)
            #  cv2.circle(frame, center=(x_top, y_top), radius=10, color=(0, 0, 255), thickness=3)

            #  a_color = tuple([cmap.colors[i][0] * 255, cmap.colors[i][1] * 255, cmap.colors[i][2] * 255])
            cv2.circle(frame, center=front_coor, radius=8, color=fishColors[i], thickness=3)
            cv2.circle(frame, center=top_coor, radius=8, color=fishColors[i], thickness=3)
            history[i].append([front_coor, top_coor])

        cv2.imshow("Video", frame)
        videoWrite.write(frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        frameNr += 1

#  capture.release()
videoWrite.release()
cv2.destroyAllWindows()

