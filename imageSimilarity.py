import cv2
import numpy as np


def slice_region_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 色彩空间转换为hsv，便于分离
    area_thresh = image.shape[0] * image.shape[1] * 0.1
    squares = []
    for clip in list(range(16, 256, 32)) + [255]:
        img = image.copy()
        img[np.where(img < clip)] = 0
        img[np.where(img >= clip)] = 255
        blur = cv2.medianBlur(img, 9)  # 奇数
        edge = cv2.Canny(blur, 5, 50)
        dilate = cv2.dilate(edge, kernel=cv2.UMat(), iterations=3)
        # findContours获取轮廓主要是用,返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
        # contourArea用来计算轮廓的面积
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for con in sorted(contours, key=cv2.contourArea, reverse=True):
            approx = cv2.approxPolyDP(con, cv2.arcLength(con, True) * 0.02, True)
            contour_area = cv2.contourArea(con)
            rect = np.resize(approx, (len(approx), 2))
            (y1, x1), (y2, x2) = ((min([j for _, j in rect]), min([i for i, _ in rect])),
                                  (max([j for _, j in rect]), max([i for i, _ in rect])))
            box_area = (x2 - x1) * (y2 - y1)
            if len(approx) >= 4 and contour_area >= area_thresh:  # and 2 * contour_area >= box_area:
                squares.append(((y1, x1, y2, x2), box_area, contour_area))
    if squares:
        y1, x1, y2, x2 = sorted(squares, key=lambda x: (x[1], x[2]), reverse=True)[0][0]
        return image[y1:y2, x1:x2]
    return None


def ORB_img_similarity(img1, img2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    if des2 is None:
        return 0
    matches = bf.match(des1, des2)
    if not matches:
        return 0
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # for d in matches:
    #     print(d.distance)

    # max_dis = matches[-1].distance
    min_dis = matches[0].distance
    good = []
    for m in matches:
        if m.distance < max(2 * min_dis, 50):
            good.append(m)

    F, mask = cv2.findFundamentalMat(np.array([kp1[m.queryIdx].pt for m in good]),
                                     np.array([kp2[m.trainIdx].pt for m in good]),
                                     cv2.FM_RANSAC)
    filter_num = np.count_nonzero(mask)
    score = filter_num * 1.0 / (len(kp1) + len(kp2) - filter_num)
    return score


