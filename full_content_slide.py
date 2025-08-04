import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import shutil
from glob import glob

def is_incremental_image(imgA, imgB, ssim_threshold=0.5, diff_threshold=0.01):
    # 灰度化处理
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # ORB 特征提取
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(grayA, None)
    kp2, des2 = orb.detectAndCompute(grayB, None)

    # Hamming距离匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test 过滤错误匹配
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return False  # 匹配点太少

    # 提取匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return False

    # 将图A配准到图B坐标系
    warpedA = cv2.warpPerspective(grayA, H, (grayB.shape[1], grayB.shape[0]))

    # 计算结构相似性差异图
    score, diff = ssim(warpedA, grayB, full=True)
    diff_area = np.sum((diff < 0.5).astype(np.uint8)) / diff.size

    # 判断是否为“增量图片”
    return score > ssim_threshold and diff_area > diff_threshold


def incremental_filter(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_paths = sorted(glob(os.path.join(image_folder, "*.*")))  # 支持任意格式

    if not image_paths:
        print("❌ No images found in folder:", image_folder)
        return

    keep_images = [image_paths[0]]  # 默认保留第一张

    for i in range(1, len(image_paths)):
        prev = cv2.imread(keep_images[-1])
        curr = cv2.imread(image_paths[i])

        if is_incremental_image(prev, curr):
            keep_images.append(image_paths[i])

    # 拷贝保留图像到输出文件夹
    for img in keep_images:
        shutil.copy(img, os.path.join(output_folder, os.path.basename(img)))

    print(f"✅ 共保留 {len(keep_images)} 张图像，保存到：{output_folder}")

# 示例调用：请根据你自己的路径修改下面两行
incremental_filter(
    image_folder=r"E:\python\python_study\SliTraNet_myfork\log\video_044",
    output_folder=r"E:\python\python_study\SliTraNet_myfork\log\video_044_new"
)
