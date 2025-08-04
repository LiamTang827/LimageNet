import os
import numpy as np
import cv2
from imageSimilarity import ORB_img_similarity  # 你定义的图像相似度函数
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def edit_distance(test_path, validate_path, threshold=0.9):
    seq1 = sorted([f for f in os.listdir(test_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                  key=natural_sort_key)

    seq2 = sorted([f for f in os.listdir(validate_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                  key=natural_sort_key)

    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    backtrack = np.full((m + 1, n + 1), '', dtype=object)

    for i in range(m + 1):
        dp[i][0] = i
        backtrack[i][0] = 'D'  # 删除
    for j in range(n + 1):
        dp[0][j] = j
        backtrack[0][j] = 'I'  # 插入

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            img1_path = os.path.join(test_path, seq1[i - 1])
            img2_path = os.path.join(validate_path, seq2[j - 1])
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            sim = ORB_img_similarity(img1, img2) if img1 is not None and img2 is not None else 0
            #print(f"{sim}")
            if sim >= threshold:
                dp[i][j] = dp[i - 1][j - 1]
                backtrack[i][j] = 'M'
            else:
                if dp[i - 1][j] <= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j] + 1
                    backtrack[i][j] = 'D'
                else:
                    dp[i][j] = dp[i][j - 1] + 1
                    backtrack[i][j] = 'I'
            #print(f"dp[{i}][{j}] = {dp[i][j]}  (由 {seq1[i - 1]} vs {seq2[j - 1]})")

    # 回溯路径，找出删除和插入的内容
    i, j = m, n
    deletions = []
    insertions = []

    while i > 0 or j > 0:
        op = backtrack[i][j]
        if op == 'M':
            i -= 1
            j -= 1
        elif op == 'D':
            deletions.append(seq1[i - 1])
            i -= 1
        elif op == 'I':
            insertions.append(seq2[j - 1])
            j -= 1

    deletions.reverse()
    insertions.reverse()

    return dp[m][n], deletions, insertions

def batch_process(test_root, validate_root, threshold=0.8):
    test_folders = sorted([f for f in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, f))])
    validate_folders = sorted([f for f in os.listdir(validate_root) if os.path.isdir(os.path.join(validate_root, f))])

    common_folders = sorted(set(test_folders) & set(validate_folders))

    for folder in common_folders:
        print(f"\n🔍 正在处理文件夹: {folder}")
        test_path = os.path.join(test_root, folder)
        validate_path = os.path.join(validate_root, folder)

        distance, deletions, insertions = edit_distance(test_path, validate_path, threshold=threshold)

        print(f"📏 编辑距离: {distance}")
        print(f"🗑 删除的帧（test中有但validate中没有）:")
        for f in deletions:
            print(f"  - {f}")
        print(f"➕ 插入的帧（validate中有但test中没有）:")
        for f in insertions:
            print(f"  + {f}")


if __name__ == '__main__':
    test_path = r"E:\python\python_study\SliTraNet_myfork\log"
    validate_path = r"E:\video_knowledge_extraction_distV1.5\keyframe_extraction_distV1.5\input_data\validate_dataset"
    distance, deletions, insertions = edit_distance(test_path, validate_path, threshold=0.8)
    batch_process(test_path,validate_path)

    print(f"\n编辑距离: {distance}")
    print(f"\n删除的帧（test中有但validate中没有）:")
    for f in deletions:
        print(f"  - {f}")
    print(f"\n插入的帧（validate中有但test中没有）:")
    for f in insertions:
        print(f"  + {f}")
