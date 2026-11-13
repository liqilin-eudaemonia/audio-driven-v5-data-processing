import numpy as np
import cv2
import matplotlib.pyplot as plt

# 替换为你的文件路径
image_path = '/data/pipline/human_centric_vw_model/benchmark/express/meta/face/pred_2_test.jpg'
keypoints_path = '/data/pipline/human_centric_vw_model/benchmark/express/meta/ldmks_98pt/pred_2_test.npy'
output_path = 'output.jpg'  # 保存路径

# 读取图像和关键点
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
keypoints = np.load(keypoints_path)  # shape (98, 2)

# 创建绘图
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image)
ax.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='red', marker='o')
for i, (x, y) in enumerate(keypoints):
    ax.text(x, y, str(i), fontsize=6, color='yellow')
ax.axis('off')

# 保存图像
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()
