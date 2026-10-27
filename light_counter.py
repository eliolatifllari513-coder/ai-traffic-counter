import cv2

cap = cv2.VideoCapture("traffik.mp4")
# detectShadows=True na ndihmon të mos numërojmë hijet e makinave si makina më vete
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zmadhojmë pak pamjen ose e lëmë siç është, marrim maskën e sfondit
    fgmask = fgbg.apply(frame)

    # Heqim hijet (hijet te MOG2 shënohen me vlerën 127, i bëjmë sfond/zi 0)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Pastrojmë zhurmat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    car_count = 0
    for cnt in contours:
        # Rritim kufirin e sipërfaqes (contourArea) që të kapë vetëm makina reale e jo objekte të vogla
        if cv2.contourArea(cnt) > 3000:
            car_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Makina (Prafic i filtruar): {car_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Accurate Traffic Counter", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

