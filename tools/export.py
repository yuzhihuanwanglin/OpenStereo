import sys
import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from PIL import Image
import cv2
import time
import torch.nn as nn
import onnx
import onnxruntime as ort
sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
from torchvision.transforms.functional import normalize
from torch.onnx import TrainingMode
# open env
device = 'cuda'#'cuda' if torch.cuda.is_available() else 'cpu'
# onnx_model_path = '/home/wanglin/workspace/OpenStereo/premodels/model_256x512.onnx'
onnx_model_path = '/home/wanglin/workspace/OpenStereo/premodels/model_640x1280.onnx'
input_shape=(1, 3, 640, 1280)
input_size = (1280,640)

class ExportWrapper(torch.nn.Module):
    def __init__(self,ori_mode):
        super().__init__()
        self.mode = ori_mode
        
    def forward(self,left,right):
        
        left = left.permute(2, 0, 1)
        right = right.permute(2, 0, 1)
        
        mean = [0.485, 0.456, 0.406]
        std =  [0.229, 0.224, 0.225]
        left = normalize(left / 255.0,mean, std)
        right = normalize(right / 255.0, mean, std)
        
        left = left.unsqueeze(0).to(device)
        right = right.unsqueeze(0).to(device)
        inputs = {'left':left,'right':right}
        return self.mode(inputs)
    
    def eval(self):
        self.mode.eval()
        
        
    def train(self, mode=True):
        self.training = mode  # 必须显式实现


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="/home/wanglin/workspace/OpenStereo/cfgs/lightstereo/lightstereo_l_sceneflow_general.yaml", help='specify the config for eval')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    
    args.dist_mode = False
    args.run_mode = 'infer'
    args.restore_ckpt = cfgs.MODEL.PRETRAINED_MODEL
    args.left_img_path = "/home/wanglin/workspace/sample/left/left.png"
    args.right_img_path = "/home/wanglin/workspace/sample/right/right.png"
    return args, cfgs
   
        
# 加载 RAFT-Stereo 模型
@torch.no_grad() 
def load_stereo_model():
    """
    加载 MSNet3D 模型
    :param weights_path: 预训练权重路径
    :return: 加载的模型
    """
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    # torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    # log
    logger = common_utils.create_logger(log_file=None, rank=local_rank)

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    trainer.model = trainer.model.to(device)
    model = ExportWrapper(trainer.model)
    
   

    print(device)
    # 加载pytorch模型的checkpoint
    checkpoint = torch.load(args.restore_ckpt,map_location=device)

    # 处理可能的'module.'前缀。当模型训练时采用了DataParallel，模型状态字典中会有'module.'前缀，我们将其去除，以便正确加载模型权重
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载模型权重
    model.load_state_dict(new_state_dict, strict=False)

    # 由于在导出过程中，我们需要跟踪一次模型的推理过程。
    # 为了加速这个过程，我们选择使用CUDA，这样使得整个导出过程更快。

    # 选择设备（GPU或CPU）
    
    
    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    transform = build_transform_by_cfg(transform_config)
    left_img = np.array(Image.open(args.left_img_path).convert('RGB'), dtype=np.float32)
    right_img = np.array(Image.open(args.right_img_path).convert('RGB'), dtype=np.float32)
    
    left_img = cv2.resize(left_img, input_size,cv2.INTER_LINEAR)
    right_img = cv2.resize(right_img, input_size,cv2.INTER_LINEAR)
    
    
    # sample = transform(sample)
    left_img = torch.from_numpy(left_img)
    right_img = torch.from_numpy(right_img)
    
    # left = left_img.transpose(2, 0, 1).astype(np.float32)#data['left'].cpu().numpy()
    # right = right_img.transpose(2, 0, 1).astype(np.float32)
        
    # left = torch.from_numpy(left).to(torch.float32)
    # right = torch.from_numpy(right).to(torch.float32)
        
    # mean = [0.485, 0.456, 0.406]
    # std =  [0.229, 0.224, 0.225]
    # left = normalize(left / 255.0,mean, std)
    # right = normalize(right / 255.0, mean, std)
    
    
    sample = {
        'left': left_img,
        'right': right_img,
    }
    
    print(sample['left'].shape)
    # sample['left'] = sample['left'].unsqueeze(0)
    # sample['right'] = sample['right'].unsqueeze(0)
    
    data = sample
    for k, v in data.items():
        data[k] = v.to(local_rank) if torch.is_tensor(v) else v


    print(data['left'])
    print(data['right'])
    
    
    print(data['left'].shape)
    print(data['right'].shape)
    model.eval()    
    # start = time.time()
    # with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
    #   
    #         model_pred = model(data['left'],data['right'])
    with torch.no_grad():
            model_pred = model(data['left'],data['right'])
        
    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()

    # 创建封装类实例
    return model,data,disp_pred

# 导出为 ONNX

def export_to_onnx(model, data,input_shape=input_shape):
    """
    将 RAFT-Stereo 模型导出为 ONNX 格式
    :param model: RAFT-Stereo 模型
    :param onnx_path: 导出的 ONNX 文件路径
    :param input_shape: 输入张量的形状 (batch_size, channels, height, width)
    """
    # 创建虚拟输入
   
    model.eval()    
    
    # model = torch.jit.script(model)
    
    
    # 显式强制所有 InstanceNorm 层为 eval 模式
    for module in model.modules():
        if isinstance(module, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d)):
            module.training = False  # 直接覆盖属性
            module.train(False)      # 调用方法确保状态更
    
    # 将模型导出为 ONNX 格式，支持固定输入和输出尺寸
    
    dumy_left = data['left']
    dumy_right = data['right']
    
    with torch.no_grad():
        torch.onnx.export(model,(dumy_left,dumy_right),onnx_model_path,training=TrainingMode.EVAL,export_params=True,opset_version=17,
                            do_constant_folding=True,input_names=['left','right'],output_names=['disp']
        )
    print(f"模型已导出为 {onnx_model_path}")

# 验证 ONNX 模型
def validate_onnx_model(data,disp_pred):
    """
    使用 ONNX Runtime 验证导出的 ONNX 模型
    :param onnx_path: ONNX 文件路径
    """
    

    # 加载 ONNX 模型
    
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # 创建虚拟输入
    dummy_left = data['left'].cpu().numpy().astype(np.float32)
    dummy_right = data['right'].cpu().numpy().astype(np.float32)

    # outputs = session.run(None, inputs)
    
    dummy_left =  np.round(dummy_left, decimals=4)
    dummy_right = np.round(dummy_right, decimals=4)
    
    
    # dumy_left = data['left']
    # dumy_right = data['right']
    
    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # providers = [('CUDAExecutionProvider',{"device_id":0})]
    # session = ort.InferenceSession(onnx_model_path, sess_options, providers=providers)
    
    session = ort.InferenceSession(onnx_model_path)
    outputs = torch.tensor(session.run(['disp'], ({'left': dummy_left,'right':dummy_right})))
    disp_pred_onnx = outputs.squeeze().cpu().numpy()
    
    # disp_pred_onnx = session.run(['disp'], {'left': dummy_left,'right':dummy_right})
    # disp_pred_onnx = disp_pred_onnx[0].squeeze()
    
    print('---------------------------\n\n\n')
    
    print(disp_pred_onnx)
    
    print('---------------------------\n\n\n')
    print(disp_pred)
    print('---------------------------\n\n\n')
    diff = np.abs(disp_pred_onnx - disp_pred)
    
    
    min_disparity = np.min(disp_pred)
    max_disparity = np.max(disp_pred)
    normalized = ((disp_pred - min_disparity) / (max_disparity - min_disparity) * 255).astype(np.uint8)
    disp_pred_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    
    cv2.imwrite('/home/wanglin/export_apt.png', disp_pred_color)
    cv2.imwrite('/home/wanglin/export_onnx.png', disp_pred_onnx)
    print("最大绝对误差:", diff.max())
    print("平均绝对误差:", diff.mean())
    
    print(f"差异超过阈值的比例: {(diff > 1e-5).mean():.2%}")

    # 定位差异最大的前10个位置
    top_indices = np.unravel_index(np.argsort(-diff.ravel())[:10], diff.shape)
    print("最大差异位置及值：")
    for idx in zip(*top_indices):
        print(f"PyTorch: {disp_pred[idx]:.4f} vs ONNX: {disp_pred_onnx[idx]:.4f} Δ={diff[idx]:.2e}")
    
    
    print("ONNX 模型推理成功！输出形状：", outputs[0].shape)
    np.testing.assert_allclose(disp_pred, disp_pred_onnx, rtol=1e-02, atol=1e-04)

# 主函数
'''
python tools/export.py --cfg_file /home/wanglin/workspace/OpenStereo/cfgs/lightstereo/lightstereo_l_sceneflow_general.yaml \n
--pretrained_model /home/wanglin/workspace/OpenStereo/premodels/StereoAnything-LightStereo_L.pt

'''


if __name__ == "__main__":
    # 加载 RAFT-Stereo 模型
    
    model,data,disp_pred = load_stereo_model()

    # 导出为 ONNX
    export_to_onnx(model,data)

    # # 验证 ONNX 模型
    validate_onnx_model(data,disp_pred)
    