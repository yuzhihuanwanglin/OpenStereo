import numpy as np
import torch
from PIL import Image
import cv2
import onnxruntime as ort
from torchvision import transforms


# left_img_path = "/home/lc/share/datas/00000/Image0/00050.jpg"
# right_img_path = "/home/lc/share/datas/00000/Image1/00050.jpg"

left_img_path = "/home/lc/share/mgz/datas/left.png"
right_img_path = "/home/lc/share/mgz/datas/right.png"


onnx_model_path = "/home/lc/gaoshan/Workspace/OpenStereo/output/model_480x640.onnx"

left_img = Image.open(left_img_path).convert("RGB")
right_img = Image.open(right_img_path).convert("RGB")
normalize_mean=[0.485, 0.456, 0.406]
normalize_std=[0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.Resize((256, 512)), 
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
])        
        
left = transform(left_img).unsqueeze(0)
right = transform(right_img).unsqueeze(0)
    
left = left.cpu().numpy()
right = right.cpu().numpy()


# left.tofile("/home/lc/gaoshan/Workspace/OpenStereo/output/left1.npy")
# right.tofile("/home/lc/gaoshan/Workspace/OpenStereo/output/right1.npy")


# np.save("/home/lc/gaoshan/Workspace/OpenStereo/output/left.npy", left)
# np.save("/home/lc/gaoshan/Workspace/OpenStereo/output/right.npy", right)
    
session = ort.InferenceSession(onnx_model_path)
outputs = torch.tensor(session.run(['disp_pred'], ({'left_img': left,'right_img':right})))

disp_pred_onnx = outputs.squeeze().cpu().numpy()
np.save("./output/run_disparity.npy", disp_pred_onnx)

print(disp_pred_onnx)

cv2.imwrite('./output/run_disparity.png', disp_pred_onnx)