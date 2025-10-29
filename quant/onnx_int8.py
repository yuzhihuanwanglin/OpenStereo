import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
import os, glob

import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
import os 
import cv2

# ====== Step 1: KITTIStereoDataset ======
class KITTIStereoDataset(Dataset):
    def __init__(self, root, transform=None):
        self.left_images = sorted(glob.glob(os.path.join(root, "image_2", "*.png")))
        self.right_images = sorted(glob.glob(os.path.join(root, "image_3", "*.png")))
        assert len(self.left_images) == len(self.right_images), "左右图像数量不匹配"
        self.transform = transform

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = Image.open(self.left_images[idx]).convert("RGB")
        right_img = Image.open(self.right_images[idx]).convert("RGB")

        if self.transform:
            left_img = self.transform(left_img)   # [3,H,W]
            right_img = self.transform(right_img) # [3,H,W]

        return left_img, right_img


def build_dataloader(root, num_samples):
    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])

    dataset = KITTIStereoDataset(root=root, transform=transform)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            shuffle=False, num_workers=1, pin_memory=True)
    return dataloader

# ====== Step 2: CalibrationDataReader for 双输入 ======
class KITTICalibDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_names):
        self.enum_data = None
        self.dataloader = dataloader
        self.input_names = input_names  # [left_name, right_name]

    def get_next(self):
        if self.enum_data is None:
            np_batches = []
            for left, right in self.dataloader:
                left_np = left.numpy().astype(np.float32)   # [1,3,H,W]
                right_np = right.numpy().astype(np.float32) # [1,3,H,W]
                np_batches.append({self.input_names[0]: left_np,
                                   self.input_names[1]: right_np})
            self.enum_data = iter(np_batches)
        return next(self.enum_data, None)

# ====== Step 3: 量化 + 推理 ======
def quantize_and_infer(model_fp32, model_int8, calib_loader,quant_format):
    # 读取模型
    onnx_model = onnx.load(model_fp32)
    input_names = [inp.name for inp in onnx_model.graph.input]
    print("[模型输入]:", input_names)

    # 静态量化
    dr = KITTICalibDataReader(calib_loader, input_names)
    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=dr,
        quant_format = quant_format,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"[OK] 静态量化完成: {model_int8}")

    # 推理测试
    sess = ort.InferenceSession(model_int8, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]
    
    left_img_path = "/media/wanglin/Elements/datasets/MiddEval3/MiddEval3/trainingQ/Piano/im0.png"
    right_img_path = "/media/wanglin/Elements/datasets/MiddEval3/MiddEval3/trainingQ/Piano/im1.png"
    
    

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
    
    feed = {input_names[0]: left, input_names[1]: right}
    outputs = torch.tensor(sess.run(['disp_pred'], feed))
    print("[OK] INT8 推理成功，输出 shape:", [o.shape for o in outputs])
    disp_pred_onnx = outputs.squeeze().cpu().numpy()
    print(disp_pred_onnx)
    min_disparity = np.min(disp_pred_onnx)
    max_disparity = np.max(disp_pred_onnx)
    normalized = ((disp_pred_onnx - min_disparity) / (max_disparity - min_disparity) * 255).astype(np.uint8)
    disp_pred_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite('./output/quant/export_int8_2.png', disp_pred_color)


    # for left, right in calib_loader:
    #     left_np = left.numpy().astype(np.float32)
    #     right_np = right.numpy().astype(np.float32)
        
    #     feed = {input_names[0]: left_np, input_names[1]: right_np}
    #     outputs = torch.tensor(sess.run(['disp_pred'], feed))
    #     print("[OK] INT8 推理成功，输出 shape:", [o.shape for o in outputs])
    #     disp_pred_onnx = outputs.squeeze().cpu().numpy()
    #     min_disparity = np.min(disp_pred_onnx)
    #     max_disparity = np.max(disp_pred_onnx)
    #     normalized = ((disp_pred_onnx - min_disparity) / (max_disparity - min_disparity) * 255).astype(np.uint8)
    #     disp_pred_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    #     cv2.imwrite('./output/quant/export_int8_2.png', disp_pred_color)
    #     break  # 只跑一个 batch 测试


# ====== Step 4: 主函数 ======
if __name__ == "__main__":
    MODEL_FP32 = "/home/extra/share/mgz3/lightstereo_s_sceneflow_general_opt_256_512_sim_conv.onnx"
    
    # MODEL_FP32 = "./output/model_480x640.onnx"
    MODEL_INT8 = "./output/quant/model_int8.onnx"
    KITTI_ROOT = "/home/extra/share/datas/testing"  # 修改为你本地 KITTI 路径

    loader = build_dataloader(KITTI_ROOT, num_samples=100)
    
    quant_format="QDQ"
    # quant_format="QOperator"
    quantize_and_infer(MODEL_FP32, MODEL_INT8, loader,quant_format)
