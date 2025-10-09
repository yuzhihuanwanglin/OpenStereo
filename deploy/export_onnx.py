import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import torch
import torch.nn as nn
from easydict import EasyDict
from deploy_utils import load_model

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add ROOT to PATH
from stereo.utils.common_utils import config_loader
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from torchvision.transforms.functional import normalize

# 包装器：把 (H, W, 3) 转换成 (1, 3, H, W) 并归一化
class ExportWrapper(nn.Module):
    def __init__(self, model, device="cuda"):
        super().__init__()
        self.model = model
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1))


    def forward(self, left_img, right_img):
       # HWC -> CHW, float 0-1
        left = left_img.permute(2, 0, 1).float() / 255.0
        right = right_img.permute(2, 0, 1).float() / 255.0

        # 标准化
        # mean = torch.tensor([0.485, 0.456, 0.406], device=left.device).view(3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225], device=left.device).view(3, 1, 1)
        # left = (left - self.mean) / self.std
        # right = (right - self.mean) / self.std
        left[0] = (left[0] - 0.485) / 0.229
        left[1] = (left[1] - 0.456) / 0.224
        left[2] = (left[2] - 0.406) / 0.225

        right[0] = (right[0] - 0.485) / 0.229
        right[1] = (right[1] - 0.456) / 0.224
        right[2] = (right[2] - 0.406) / 0.225

        # 增加 batch 维度
        left = left.unsqueeze(0)
        right = right.unsqueeze(0)

        # 前向
        out = self.model({'left': left, 'right': right})
        return out


def export_onnx(model, dummy_left, dummy_right, weights, opset=12, dynamic=False, simplify=True):
    """ONNX 导出"""
    import onnx

    f = Path(weights).with_suffix('.onnx')
    input_names = ['left_img', 'right_img']
    output_names = ['disp_pred']

    torch.onnx.export(
        model,
        (dummy_left, dummy_right),   # 注意这里传 tuple，而不是 dict
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None 
    )

    # 校验
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)

    # 简化
    if simplify:
        try:
            import onnxsim
            model_opt, check = onnxsim.simplify(onnx_model)
            assert check, 'onnx-simplifier check failed'
            onnx.save(model_opt, f)
        except Exception as e:
            print(f"Simplifier failed: {e}")

    return f


def run(config, weights, imgsz=(256, 512), device='cpu',
        dynamic=False, simplify=True, opset=12):

    # 加载模型配置
    yaml_config = config_loader(config)
    cfgs = EasyDict(yaml_config)

    if not os.path.isfile(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")
    base_model = load_model(LightStereo(cfgs.MODEL), weights)

    # 包装
    model = ExportWrapper(base_model).eval().to(device)

    # 输入 (H, W, 3)
    h, w = imgsz
    dummy_left = torch.zeros(h, w, 3, dtype=torch.uint8).to(device)
    dummy_right = torch.zeros(h, w, 3, dtype=torch.uint8).to(device)


    print(dummy_left.shape)
    print(dynamic)

    # 导出
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
    t0 = time.time()
    f = export_onnx(model, dummy_left, dummy_right, weights, opset, dynamic, simplify)
    print(f"Export complete in {time.time()-t0:.1f}s\nSaved to: {f}")
    return f


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='模型 config.yaml')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[256, 512], help='输入尺寸 (h w)')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 或 cpu')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='是否用 onnx-simplifier 简化模型')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
