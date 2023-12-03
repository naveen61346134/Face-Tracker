import cv2 as cv

face_cascade = cv.CascadeClassifier(
    "Trackers/haarcascade_frontalface_default.xml")


def changeRes(width, height, capture):
    capture.set(3, width)
    capture.set(4, height)
    return capture


def video(cascade, scale=.75):
    cam = cv.VideoCapture(0)
    capture = changeRes(1920, 1080, cam)
    while True:
        frame = capture.read()[1]
        frame = cv.flip(frame, 1)

        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimensions = (width, height)
        nFrame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

        smoothFrame = cv.medianBlur(nFrame, 3)

        grayFrame = cv.cvtColor(nFrame, cv.COLOR_BGR2GRAY)
        face_coordinates = cascade.detectMultiScale(grayFrame, 1.3, 4)
        print(f"FACE COORDINATES: {face_coordinates}")

        for x, y, w, h in face_coordinates:
            cv.putText(smoothFrame, "FACE", (x, y-5),
                       cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness=2)
            cv.rectangle(smoothFrame, (x, y), (x+w, y+h),
                         (0, 255, 0), thickness=3)
        cv.imshow("FACE TRACKER", smoothFrame)

        if cv.waitKey(20) & 0xFF == ord("d"):
            break
    cam.release()
    cv.destroyAllWindows()


video(face_cascade, 1)
