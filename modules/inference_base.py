import torch
from ldm.util import fix_cond_shapes
from modules.utils import get_pad_img_mask, get_init_img, get_mask


def diffusion_inference(cfg, inputs):
    img_path, mask_path, model, sampler, prompt, guide_signals = inputs

    # Get text embedding
    c = model.get_learned_conditioning(cfg.infer.batch_size * [prompt])
    uc = model.get_learned_conditioning(cfg.infer.batch_size * [cfg.neg_prompt])
    c, uc = fix_cond_shapes(model, c, uc)

    H = 512
    W = 512
    shape = [cfg.infer.init_C, H // cfg.infer.down_factor, W // cfg.infer.down_factor]

    init_img = get_init_img(model.device, img_path, cfg.infer.batch_size)
    down_mask, init_mask = get_mask(model.device, mask_path, cfg.infer.batch_size)
    encode_mask_img = model.get_first_stage_encoding(model.encode_first_stage((1 - init_mask) * init_img))

    c_cat = torch.cat([down_mask, encode_mask_img], dim=1)

    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc]}

    samples_latents, _ = sampler.sample(
        S=cfg.infer.steps,
        conditioning=cond,
        batch_size=cfg.infer.batch_size,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=cfg.infer.classifier_free_guidance,
        unconditional_conditioning=uc_full,
        x_T=None,
        guide_signals=guide_signals,
        P=cfg.infer.CMB.P,
        Q=cfg.infer.CMB.Q,
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = x_samples * init_mask + (1 - init_mask) * init_img

    return torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)


def diffusion_inference_pad(cfg, inputs):
    img_path, mask_path, model, sampler, prompt, guide_signals = inputs

    batch, ori_h, ori_w = get_pad_img_mask(img_path, mask_path, prompt, model.device, cfg.infer.batch_size)
    h, w = batch['image'].shape[-2:]

    c = model.get_learned_conditioning(batch['txt'])
    uc = model.get_learned_conditioning(cfg.infer.batch_size * [cfg.neg_prompt])
    c_cat = []
    for ck in model.concat_keys:
        cc = batch[ck].float()
        if ck != model.masked_image_key:
            bchw = [1, 4, h // cfg.infer.down_factor, w // cfg.infer.down_factor]
            cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
        else:
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
        c_cat.append(cc)
    c_cat = torch.cat(c_cat, dim=1)

    cond = {"c_concat": [c_cat], "c_crossattn": [c]}
    uc_full = {"c_concat": [c_cat], "c_crossattn": [uc]}

    shape = [model.channels, h // cfg.infer.down_factor, w // cfg.infer.down_factor]
    samples_latents, _ = sampler.sample(
        S=cfg.infer.steps,
        conditioning=cond,
        batch_size=cfg.infer.batch_size,
        shape=shape,
        verbose=False,
        unconditional_guidance_scale=cfg.infer.classifier_free_guidance,
        unconditional_conditioning=uc_full,
        x_T=None,
        guide_signals=guide_signals,
        P = cfg.infer.CMB.P,
        Q = cfg.infer.CMB.Q
    )

    x_samples = model.decode_first_stage(samples_latents)
    x_samples = (1 - batch['mask']) * batch['image'] + batch['mask'] * x_samples
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return x_samples[..., :ori_h, :ori_w]