import os
import cv2
import torch
import json
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast

from basicsr.utils import tensor2img
from modules.utils import setup_logger
from annotator import api
from annotator.api import (
    ExtraCondition,
    get_cond_model,
    get_weigted_guide_signal
)
from ldm.data.utils import load_flist
from modules.utils import get_tau_nets, get_sd_models

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["DISABLE_XFORMERS"] = 'true'  # disable XFORMERS

@hydra.main(version_base=None, config_path="configs", config_name="example_config")
def main(cfg):
    supported_cond = [e.name for e in ExtraCondition]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Select auxiliary guidance
    activated_conds = []
    cond_flists = []
    cond_ckpts = []
    for cond_name in supported_cond:
        if not hasattr(cfg.infer.auxiliary, f'{cond_name}_path'):
            continue
        assert getattr(cfg.infer.auxiliary, f'{cond_name}_ckpt') is not None, f'you should specify the {cond_name}_ckpt'
        activated_conds.append(cond_name)
        cond_flists.append(load_flist(getattr(cfg.infer.auxiliary, f'{cond_name}_path')))
        cond_ckpts.append(getattr(cfg.infer.auxiliary, f'{cond_name}_ckpt'))
    assert len(activated_conds) <= 1 or cfg.infer.CMB.enable, f'please using CMB by editing config.yaml if input multiple {len(activated_conds)} condition'

    # Prepare models
    tau_nets = []
    cond_transformers = []  # Transform image into guidance
    cond_inp_types = []
    process_cond_modules = []
    for cond_name in activated_conds:
        tau_nets.append(get_tau_nets(cfg, device, getattr(ExtraCondition, cond_name)))
        cond_inp_type = getattr(cfg.infer.auxiliary, f'{cond_name}_inp_type', 'image')
        if cond_inp_type == 'image':
            cond_transformers.append(get_cond_model(device, getattr(ExtraCondition, cond_name)))
        else:
            cond_transformers.append(None)
        cond_inp_types.append(cond_inp_type)
        process_cond_modules.append(getattr(api, f'get_cond_{cond_name}'))
    sd_model, sampler = get_sd_models(cfg.general, cfg.infer.CMB.enable, device)

    if not cfg.infer.fixed_resolution:
        from modules.inference_base import diffusion_inference_pad as diffusion_inference
    else:
        from modules.inference_base import diffusion_inference

    # Prepare inference dataset
    img_flist = load_flist(cfg.infer.data.img_path)
    mask_flist = load_flist(cfg.infer.data.mask_path)
    with open(cfg.infer.caption_path, 'r', encoding='utf-8') as fp:  # Load annotations
        data = json.load(fp)['annotations']
    prompt_dic = {f"%012d.jpg" % file['image_id']: file['caption'] for file in data}

    # Prepare the save path
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    with sd_model.ema_scope(), autocast('cuda'):
        seed_everything(cfg.general.rand_seed)
        for img_path, mask_path, *cond_paths in tqdm(zip(img_flist, mask_flist, *cond_flists)):
            basename = os.path.basename(img_path)
            for idx, path in enumerate(cond_paths):
                cond_paths[idx] = os.path.dirname(path)
            prompt = prompt_dic[basename] if cfg.infer.load_prompt_text and basename in prompt_dic else ""
            logger.info(f'img_path: {img_path} \nprompt: {prompt}')

            guide_signals = []  # Multi-modal guide signals
            for cond_idx, cond_name in enumerate(activated_conds):
                cond = process_cond_modules[cond_idx](
                    cfg, os.path.join(cond_paths[cond_idx], basename), device, cond_inp_types[cond_idx], cond_transformers[cond_idx]
                )
                # save condition img
                # image_path = os.path.join(cfg.general.save_path, f"{basename.split('.')[0]}_cond.jpg")
                # cv2.imwrite(image_path, tensor2img(cond))
                singular_guide_signal = get_weigted_guide_signal(cond, tau_nets[cond_idx], cfg.infer.batch_size)
                guide_signals.append(singular_guide_signal)

            guide_signals = guide_signals if len(guide_signals) > 0 else None
            if not cfg.infer.CMB.enable and guide_signals is not None:
                guide_signals = guide_signals[0]  # when not use CMB

            for idx in range(cfg.infer.n_samples):
                # Inference
                result = diffusion_inference(cfg, (img_path, mask_path, sd_model, sampler, prompt, guide_signals))

                # Save example images
                for bs_idx, res in enumerate(result):
                    image_path = os.path.join(cfg.general.save_path, f"{basename.split('.')[0]}_{idx * cfg.infer.batch_size + bs_idx}.jpg")
                    cv2.imwrite(image_path, tensor2img(res))
                    logger.info('save example image to {}'.format(image_path))


if __name__ == '__main__':
    main()
