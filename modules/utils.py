import os
import logging
import torch
import numpy as np
from PIL import Image
from einops import repeat
import cv2
import torch.nn.functional as F
from basicsr.utils import img2tensor
from omegaconf import OmegaConf

from modules.tau_net import StructuralTauNet
from annotator.api import ExtraCondition, get_cond_ch

from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config, read_state_dict


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    sd = read_state_dict(ckpt)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if 'anything' in ckpt.lower() and vae_ckpt is None:
        vae_ckpt = 'models/anything-v4.0.vae.pt'

    if vae_ckpt is not None and vae_ckpt != 'None':
        print(f"Loading vae model from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")
        if "global_step" in vae_sd:
            print(f"Global Step: {vae_sd['global_step']}")
        sd = vae_sd["state_dict"]
        m, u = model.first_stage_model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.cuda()
    model.eval()
    return model


def get_sd_models(opt, use_CMB, device):
    """
    build stable diffusion model, sampler
    """
    # SD
    print(os.path.dirname(__file__), opt.config)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, opt.sd_ckpt, opt.vae_ckpt)
    sd_model = model.to(device)

    # sampler
    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    elif opt.sampler == 'ddim':
        if use_CMB:
            from ldm.models.diffusion.ddim_infer import DDIMSampler
        else:
            from ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(model)
    else:
        raise NotImplementedError

    return sd_model, sampler


def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def get_tau_nets(cfg, device, cond_type: ExtraCondition):
    tau_net = {}
    cond_weight = getattr(cfg.infer.auxiliary, f'{cond_type.name}_weight', None)
    if cond_weight is None:
        cond_weight = getattr(cfg.infer.auxiliary, 'cond_weight')
    tau_net['cond_weight'] = cond_weight

    if cond_type == ExtraCondition.style or cond_type == ExtraCondition.reference:
        tau_net['model'] = ContextualTauNet(width=1024, context_dim=1024, num_head=8, n_layes=3, num_token=8).to(device)
    else:
        tau_net['model'] = StructuralTauNet(
            cin=64 * get_cond_ch(cond_type),
            channels=[320, 640, 1280, 1280][:4],
            nums_rb=2,
            ksize=1,
            sk=True,
            use_conv=False).to(device)
    ckpt_path = getattr(cfg.infer.auxiliary, f'{cond_type.name}_ckpt', None)
    if ckpt_path is None:
        ckpt_path = getattr(cfg.infer.auxiliary, '_ckpt')
    state_dict = read_state_dict(ckpt_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v

    tau_net['model'].load_state_dict(new_state_dict)

    return tau_net


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        batch_size=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=batch_size),
        "txt": batch_size * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=batch_size),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=batch_size),
    }
    return batch


def get_pad_img_mask(img_path, mask_path, prompt, device, batch_size):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    h, w = img.height, img.width
    img = pad_image(img)
    mask = pad_image(mask)
    batch = make_batch_sd(img, mask, txt=prompt, device=device, batch_size=batch_size)
    return batch, h, w


def get_init_img(device, img_path: str, n_samples: int = 1):
    img = cv2.imread(img_path)
    img = torch.stack([img2tensor(img)] * n_samples, 0) / 255.
    img = F.interpolate(img, (512, 512))
    img = img.to(device)

    return img * 2. - 1.


def get_mask(device, mask_path: str, n_samples: int = 1):
    dest_size, img_size = (64, 64), (512, 512)
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask) / 255

    masks_array = []
    masks_array.append(mask)
    masks_array = np.array(masks_array).astype(np.float32)
    masks_array = masks_array[:, np.newaxis, :]
    masks_array = torch.from_numpy(masks_array).to(device)

    org_mask = org_mask.resize(img_size, Image.LANCZOS)
    org_mask = np.array(org_mask).astype(np.float32) / 255.0
    org_mask = org_mask[None, None]
    org_mask[org_mask < 0.5] = 0
    org_mask[org_mask >= 0.5] = 1
    org_mask = torch.from_numpy(org_mask).to(device)

    return torch.cat([masks_array] * n_samples, 0), torch.cat([org_mask] * n_samples, 0)
