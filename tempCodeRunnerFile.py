while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitkey(1)