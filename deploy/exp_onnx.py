import argparse
import os
import sys
import time
import warnings
from pathlib import Path
import torch
from easydict import EasyDict
from deploy_utils import load_model
sys.path.append(str(Path(__file__).resolve().parent.parent))  # add ROOT to PATH
from stereo.utils.common_utils import config_loader
from stereo.modeling.models.lightstereo.lightstereo import LightStereo


def export_onnx(model, inputs, weights, opset=12, dynamic=False, simplify=True):
    """ONNX 导出"""
    import onnx

    f = Path(weights).with_suffix('.onnx')
    input_names = ['left_img', 'right_img']
    output_names = ['disp_pred']

    torch.onnx.export(
        model,
        {'data': inputs},
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None if not dynamic else {
            'left_img': {0: 'batch', 2: 'height', 3: 'width'},
            'right_img': {0: 'batch', 2: 'height', 3: 'width'},
            'disp_pred': {1: 'height', 2: 'width'}
        }
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


def run(config, weights, imgsz=(256, 512), batch_size=1, device='cpu',
        half=False, dynamic=False, simplify=True, opset=12):

    # 加载模型配置
    yaml_config = config_loader(config)
    cfgs = EasyDict(yaml_config)

    if not os.path.isfile(weights):
        raise FileNotFoundError(f"权重文件不存在: {weights}")
    model = load_model(LightStereo(cfgs.MODEL), weights)

    # 输入
    h, w = imgsz
    left = torch.zeros(batch_size, 3, h, w, dtype=torch.float).to(device)
    right = torch.zeros(batch_size, 3, h, w, dtype=torch.float).to(device)
    inputs = {'left': left, 'right': right}

    # 导出
    model.eval().to(device)
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)

    t0 = time.time()
    f = export_onnx(model, inputs, weights, opset, dynamic, simplify)
    print(f"Export complete in {time.time()-t0:.1f}s\nSaved to: {f}")
    return f


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='模型 config.yaml')
    parser.add_argument('--weights', type=str, required=True, help='模型权重路径')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[256, 512], help='输入尺寸 (h w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 或 cpu')
    parser.add_argument('--simplify', action='store_true', help='是否用 onnx-simplifier 简化模型')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
