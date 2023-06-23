import torch
import cv2
from PIL import Image
import numpy as np
from torch import autocast
from basicsr import tensor2img, img2tensor
from enum import Enum, unique
from einops import repeat


@unique
class ExtraCondition(Enum):
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5
    reference = 6
    

    
def get_cond_model(device, cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch:
        from annotator.sketch import pidinet
        model = pidinet()
        ckp = torch.load('checkpoints/image_to_guidance/table5_pidinet.pth', map_location='cpu')['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()}, strict=True)
        model.to(device)
        return model
    elif cond_type == ExtraCondition.seg:
        raise NotImplementedError
    elif cond_type == ExtraCondition.keypose:
        import mmcv
        from mmdet.apis import init_detector
        from mmpose.apis import init_pose_model
        det_config = 'configs/annotator/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        pose_config = 'configs/annotator/hrnet_w48_coco_256x192.py'
        pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        det_config_mmcv = mmcv.Config.fromfile(det_config)
        det_model = init_detector(det_config_mmcv, det_checkpoint, device=device)
        pose_config_mmcv = mmcv.Config.fromfile(pose_config)
        pose_model = init_pose_model(pose_config_mmcv, pose_checkpoint, device=device)
        return {'pose_model': pose_model, 'det_model': det_model}
    elif cond_type == ExtraCondition.depth:
        from annotator.depth.api import MiDaSInference
        model = MiDaSInference(model_type='dpt_hybrid').to(device)
        return model
    elif cond_type == ExtraCondition.canny:
        return None
    elif cond_type == ExtraCondition.style or cond_type == ExtraCondition.reference:
        from transformers import CLIPProcessor, CLIPVisionModel
        version = 'openai/clip-vit-large-patch14'
        processor = CLIPProcessor.from_pretrained(version)
        clip_vision_model = CLIPVisionModel.from_pretrained(version).to(device)
        return {'processor': processor, 'clip_vision_model': clip_vision_model}
    else:
        raise NotImplementedError
    
    
def get_cond_sketch(cfg, cond_image, device, cond_inp_type, cond_model=None):
    """
    Get the conditional sketch from the provided image or sketch. This function supports both fixed and non-fixed 
    resolution modes and can handle either direct sketch input or image input which is then transformed to a sketch.

    Args:
        cfg (config): The configuration parameters for the process.
        cond_image (str or PIL Image): The conditional image or path to it.
        device (torch.device): The device on which the calculations should be performed.
        cond_inp_type (str): The type of the input - 'sketch' or 'image'.
        cond_model (model, optional): If input is an image, this model will be used to convert it to a sketch.

    Returns:
        torch.Tensor: The resulting sketch.
    """

    if cfg.infer.fixed_resolution:
        # If cond_image is a path, read it, otherwise convert to BGR
        edge = cv2.imread(cond_image) if isinstance(cond_image, str) else cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
        edge = cv2.resize(edge, (512, 512))  # Resize the image
    else:
        edge = Image.open(cond_image).convert("RGB")  # Open and convert the image
        # ========outpainting========
        # w, h = edge.size
        # edge = edge.resize((int(w * (512 / h)), 512))
        # ===========================
        from modules.utils import pad_image
        edge = np.array(pad_image(edge))  # Pad the image

    if cond_inp_type == 'sketch':
        if cfg.infer.fixed_resolution:
            edge = img2tensor(edge)[0].unsqueeze(0).unsqueeze(0) / 255.  # Convert image to tensor and normalize
        else:
            edge = np.array(edge)[None].transpose(0, 3, 1, 2)[:, 0:1]  # Convert image to numpy array, transpose, and normalize
            edge = torch.from_numpy(edge).to(dtype=torch.float32) / 255  # Convert numpy array to tensor
        edge = edge.to(device)  # Move the tensor to the device
    elif cond_inp_type == 'image':
        edge = img2tensor(edge).unsqueeze(0) / 255.  # Convert image to tensor and normalize
        edge = cond_model(edge.to(device))[-1]  # Pass the image through the condition model
    else:
        raise NotImplementedError("Condition input type not implemented.")

    edge = (edge > 0.5).float()  # Binarize the sketch

    return edge



def get_cond_reference(cfg, cond_image, device, cond_inp_type='image', cond_model=None):
    assert cond_inp_type == 'image'
    if isinstance(cond_image, str):
        ref = Image.open(cond_image)
    else:
        # numpy image to PIL image
        ref = Image.fromarray(cond_image)

    ref_for_clip = cond_model['processor'](images=ref, return_tensors="pt")['pixel_values']
    ref_feat = cond_model['clip_vision_model'](ref_for_clip.to(device))['last_hidden_state']

    return ref_feat


def get_cond_seg(cfg, cond_image, device, cond_inp_type='image', cond_model=None):
    # if cond_inp_type == 'seg' and cond_image.endswith('jpg'): 
    #     import os
    #     cond_image = os.path.splitext(cond_image)[0] + '.png'
    if cfg.infer.fixed_resolution:
        if isinstance(cond_image, str):
            seg = cv2.imread(cond_image)
        else:
            seg = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
        seg = cv2.resize(seg, (512, 512))
    else:
        seg = Image.open(cond_image).convert('RGB')
        from modules.utils import pad_image
        seg = np.array(pad_image(seg))
    if cond_inp_type == 'seg':
        if cfg.infer.fixed_resolution:
            seg = img2tensor(seg).unsqueeze(0) / 255.
        else:
            seg = np.array(seg)[None].transpose(0, 3, 1, 2)
            seg = torch.from_numpy(seg).to(dtype=torch.float32) / 255
        seg = seg.to(device)
    else:
        raise NotImplementedError

    return seg


def get_cond_keypose(cfg, cond_image, device, cond_inp_type='image', cond_model=None):
    if cfg.infer.fixed_resolution:
        if isinstance(cond_image, str):
            pose = cv2.imread(cond_image)
        else:
            pose = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
        pose = cv2.resize(pose, (512, 512))
    else:
        pose = Image.open(cond_image).convert('RGB')
        from modules.utils import pad_image
        pose = np.array(pad_image(pose))
    if cond_inp_type == 'keypose':
        if cfg.infer.fixed_resolution:
            pose = img2tensor(pose).unsqueeze(0) / 255.
            pose = pose.to(device)
        else:
            pose = np.array(pose)[None].transpose(0, 3, 1, 2)
            pose = torch.from_numpy(pose).to(dtype=torch.float32) / 255
            pose = pose.to(device)
    elif cond_inp_type == 'image':
        from annotator.keypose import imshow_keypoints
        from mmdet.apis import inference_detector
        from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)

        # mmpose seems not compatible with autocast fp16
        with autocast("cuda", dtype=torch.float32):
            mmdet_results = inference_detector(cond_model['det_model'], pose)
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, 1)

            # optional
            return_heatmap = False
            dataset = cond_model['pose_model'].cfg.data['test']['type']

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            pose_results, returned_outputs = inference_top_down_pose_model(
                cond_model['pose_model'],
                pose,
                person_results,
                bbox_thr=0.2,
                format='xyxy',
                dataset=dataset,
                dataset_info=None,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

        # show the results
        pose = imshow_keypoints(pose, pose_results, radius=2, thickness=2)
        pose = img2tensor(pose).unsqueeze(0) / 255.
        pose = pose.to(device)
    else:
        raise NotImplementedError

    return pose


def get_cond_depth(cfg, cond_image, device, cond_inp_type='image', cond_model=None):
    # if cond_inp_type == 'depth' and cond_image.endswith('jpg'): 
    #     import os
    #     cond_image = os.path.splitext(cond_image)[0] + '.png'
    if cfg.infer.fixed_resolution:
        if isinstance(cond_image, str):
            depth = cv2.imread(cond_image)
        else:
            depth = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
        depth = cv2.resize(depth, (512, 512))
    else:
        depth = Image.open(cond_image).convert('RGB')
        from modules.utils import pad_image
        depth = np.array(pad_image(depth))
    if cond_inp_type == 'depth':
        if cfg.infer.fixed_resolution:
            depth = img2tensor(depth).unsqueeze(0) / 255.
        else:
            depth = np.array(depth)[None].transpose(0, 3, 1, 2)
            depth = torch.from_numpy(depth).to(dtype=torch.float32) / 255
        depth = depth.to(device)
    elif cond_inp_type == 'image':
        depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
        depth = cond_model(depth.to(device)).repeat(1, 3, 1, 1)
        depth -= torch.min(depth)
        depth /= torch.max(depth)
    else:
        raise NotImplementedError

    return depth


def get_cond_canny(cfg, cond_image, device, cond_inp_type='image', cond_model=None):
    if cfg.infer.fixed_resolution:
        if isinstance(cond_image, str):
            canny = cv2.imread(cond_image)
        else:
            canny = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
        canny = cv2.resize(canny, (512, 512))
    else:
        canny = Image.open(cond_image).convert('RGB')
        from modules.utils import pad_image
        canny = np.array(pad_image(canny))
    if cond_inp_type == 'canny':
        if cfg.infer.fixed_resolution:
            canny = img2tensor(canny)[0:1].unsqueeze(0) / 255.
        else:
            canny = np.array(canny)[None].transpose(0, 3, 1, 2)[:, 0:1]
            canny = torch.from_numpy(canny).to(dtype=torch.float32) / 255
        canny = canny.to(device)
    elif cond_inp_type == 'image':
        canny = cv2.Canny(canny, 100, 200)[..., None]
        canny = img2tensor(canny).unsqueeze(0) / 255.
        canny = canny.to(device)
    else:
        raise NotImplementedError

    return canny


def get_cond_ch(cond_type: ExtraCondition):
    if cond_type == ExtraCondition.sketch or cond_type == ExtraCondition.canny:
        return 1
    return 3


def get_weigted_guide_signal(guides, tau_nets, batch_size):
    ret_feat_map = None
    if not isinstance(guides, list):
        guides = [guides]
        tau_nets = [tau_nets]

    for guide, tau_net in zip(guides, tau_nets):
        cur_feature = tau_net['model'](guide)
        if ret_feat_map is None:
            ret_feat_map = list(map(lambda x: repeat(x, "1 ... -> n ...", n=batch_size) * tau_net['cond_weight'], cur_feature))
        else:
            ret_feat_map = list(map(lambda x, y: x + y * tau_net['cond_weight'], ret_feat_map, cur_feature))

    return ret_feat_map#, ret_feat_seq