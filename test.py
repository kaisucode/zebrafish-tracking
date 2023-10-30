
import cv2

print("import done")

numFish = 5
videoSource = "data/ZebraFish-04-raw.webm"
labelSource = "data/3DZeF20Lables/train/ZebraFish-04/gt/gt.txt"

capture = cv2.VideoCapture(videoSource)
totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

lines = None
with open(labelSource) as file:
    lines = [line.rstrip() for line in file]

print("read in lines: ", str(len(lines)))
print("total frames: ", totalFrames)
frameNr = 0

while True:
    success, frame = capture.read()
    if success:
        #  cv2.imwrite("frames/frame" + str(frameNr) + ".png", frame)

        for i in range(frameNr, frameNr + numFish): 
            annotatedData = lines[frameNr * 5 + i].split(",")
            x, y = int(annotatedData[12]) // 2, int(annotatedData[13]) // 2
            #  x, y = int(annotatedData[5]), int(annotatedData[6])
            #  print("coordiantes: ", x, ", ", y)
            cv2.circle(frame, center = (x, y), radius = 10,color = (0, 0, 255),thickness=3)

        cv2.imshow("Video", frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        frameNr += 1

#  capture.release()

cv2.destroyAllWindows()



