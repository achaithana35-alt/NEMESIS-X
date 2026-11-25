import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected")
    exit()

print("Camera connected successfully")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    print("Video frame received")

    cv2.imshow("NEMESIS-X Live Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()