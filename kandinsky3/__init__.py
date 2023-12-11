import os
from typing import List, Optional, Union, cast

import numpy as np
import omegaconf
import PIL
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from tqdm import tqdm

from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule
from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.utils import flush

from .inpainting_pipeline import Kandinsky3InpaintingPipeline
from .t2i_pipeline import Kandinsky3T2IPipeline


def get_T2I_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    fp16: bool = False,
    fp8: bool = False,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        projections_state_dict = state_dict["projections"]
        null_embedding = state_dict["null_embedding"]
        unet.load_state_dict(state_dict["unet"])

    if fp8:
        unet.eval().to(cast(torch.device, device), torch.float8_e4m3fn)
    elif fp16:
        unet.eval().to(cast(torch.device, device), torch.bfloat16)
    else:
        unet.eval().to(cast(torch.device, device))

    return unet, null_embedding, projections_state_dict


def get_T2I_nullemb_projections(
    weights_path: Optional[str] = None,
) -> (Optional[torch.Tensor], Optional[dict]):
    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        projections_state_dict = state_dict["projections"]
        null_embedding = state_dict["null_embedding"]

    return null_embedding, projections_state_dict


def get_T5encoder(
    device: Union[str, torch.device],
    weights_path: str,
    projections_state_dict: Optional[dict] = None,
    fp16: bool = True,
    low_cpu_mem_usage: bool = True,
    device_map: Optional[str] = None,
) -> (T5TextConditionProcessor, T5TextConditionEncoder):
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}
    context_dim = 4096
    model_dims = {"t5": 4096}
    processor = T5TextConditionProcessor(tokens_length, model_names)
    condition_encoders = T5TextConditionEncoder(
        model_names,
        context_dim,
        model_dims,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
    )

    if projections_state_dict:
        condition_encoders.projections.load_state_dict(projections_state_dict)

    condition_encoders = condition_encoders.eval().to(device)
    return processor, condition_encoders


def get_T5processor(
    weights_path: str,
) -> T5TextConditionProcessor:
    model_names = {"t5": weights_path}
    tokens_length = {"t5": 128}
    processor = T5TextConditionProcessor(tokens_length, model_names)

    return processor


def get_movq(
    device: Union[str, torch.device], weights_path: str, fp16: bool = False
) -> MoVQ:
    generator_config = {
        "double_z": False,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 256,
        "ch_mult": [1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [32],
        "dropout": 0.0,
    }
    movq = MoVQ(generator_config)
    movq.load_state_dict(torch.load(weights_path))
    movq = movq.eval().to(device)

    if fp16:
        movq = movq.bfloat16()
    return movq


def get_inpainting_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None,
    fp16: bool = False,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=9,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        projections_state_dict = state_dict["projections"]
        null_embedding = state_dict["null_embedding"]
        unet.load_state_dict(state_dict["unet"])

    if fp16:
        unet = unet.bfloat16()

    unet.eval().to(device)

    return unet, null_embedding, projections_state_dict


def get_T2I_pipeline(
    device: Union[str, torch.device],
    fp16: bool = False,
    fp8: bool = False,
    cache_dir: str = "/tmp/kandinsky3/",
    unet_path: str | None = None,
    text_encode_path: str | None = None,
    movq_path: str | None = None,
) -> Kandinsky3T2IPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/kandinsky3.pt",
            cache_dir=cache_dir,
        )
    if text_encode_path is None:
        text_encode_path = "google/flan-ul2"
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

    null_embedding, projections_state_dict = get_T2I_nullemb_projections(unet_path)
    processor = get_T5processor(text_encode_path)

    encoder_loader = lambda: get_T5encoder(
        device, text_encode_path, projections_state_dict, fp16=fp16
    )[1]
    unet_loader = lambda: get_T2I_unet(device, unet_path, fp16=fp16, fp8=fp8)[0]
    movq_loader = lambda: get_movq(device, movq_path, fp16=fp16)

    flush()

    return Kandinsky3T2IPipeline(
        device,
        unet_loader,
        null_embedding,
        processor,
        encoder_loader,
        movq_loader,
        fp16=fp16,
    )


def get_inpainting_pipeline(
    device: Union[str, torch.device],
    fp16: bool = False,
    cache_dir: str = "/tmp/kandinsky3/",
    unet_path: str = None,
    text_encode_path: str = None,
    movq_path: str = None,
) -> Kandinsky3InpaintingPipeline:
    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/kandinsky3_inpainting.pt",
            cache_dir=cache_dir,
        )
    if text_encode_path is None:
        text_encode_path = "google/flan-ul2"
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.0",
            filename="weights/movq.pt",
            cache_dir=cache_dir,
        )

    unet, null_embedding, projections_state_dict = get_inpainting_unet(
        device, unet_path, fp16=fp16
    )
    processor, condition_encoders = get_T5encoder(
        device, text_encode_path, projections_state_dict, fp16=fp16
    )
    movq = get_movq(
        device, movq_path, fp16=False
    )  # MoVQ ooesn't work properly in fp16 on inpainting
    return Kandinsky3InpaintingPipeline(
        device, unet, null_embedding, processor, condition_encoders, movq, fp16=fp16
    )
