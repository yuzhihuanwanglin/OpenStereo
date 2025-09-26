import cv2
import numpy as np

arr = np.load("/home/lc/share/mgz/compare_light_out/layerdump/disp_pred_quanted.npy")
# arr = np.load("/home/lc/share/datas/00000/Disp0/00001.npy")

arr = arr.squeeze()
print(arr.shape)
print(arr)

# arr = arr.transpose(1,2,0)
disp_vis = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = disp_vis.astype(np.uint8)

# 伪彩色显示
disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

cv2.imshow("Disparity Map", disp_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()