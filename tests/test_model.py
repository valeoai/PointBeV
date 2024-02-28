"""
Check if the installation works fine and if the model runs.
"""

import os
import time

import hydra
import torch
from einops import repeat

NUM_REPEAT = 1
B, Tin, N, C, H, W = 1, 1, 6, 3, 224, 480
Nq = 1


def print_elapsed(start, num_repeat, message=""):
    elapsed_time = time.perf_counter() - start
    ms = elapsed_time * 1_000 / num_repeat
    print(message + f"{ms:.2f}ms")


def process(model, imgs, rots, trans, intrins, bev_aug, *args, **kwargs):
    message = "Forward:"
    start = time.perf_counter()
    for _ in range(NUM_REPEAT):
        out = model(imgs, rots, trans, intrins, bev_aug, *args, **kwargs)
    print_elapsed(start, NUM_REPEAT, message)
    torch.cuda.empty_cache()

    message = "Eval:"
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(NUM_REPEAT):
            out = model(imgs, rots, trans, intrins, bev_aug, *args, **kwargs)
    print_elapsed(start, NUM_REPEAT, message)
    torch.cuda.empty_cache()

    message = "Backward:"
    model.train()
    start = time.perf_counter()
    for _ in range(NUM_REPEAT):
        out = model(imgs, rots, trans, intrins, bev_aug, *args, **kwargs)
        out["bev"]["offsets"].mean().backward()
    print_elapsed(start, NUM_REPEAT, message)

    del model
    del imgs, rots, trans, intrins, bev_aug
    torch.cuda.empty_cache()
    return


def test_pointbevmodel(cfg_pointbev):
    model = hydra.utils.instantiate(cfg_pointbev.model.net)
    device = "cuda"
    model.to(device)

    imgs = torch.randn(B, Tin, N, C, H, W).to(device)
    rots = torch.randn(B, Tin, N, 3, 3).to(device)
    trans = torch.randn(B, Tin, N, 3, 1).to(device)
    intrins = torch.randn(B, Tin, N, 3, 3).to(device)
    bev_aug = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=Tin).to(device)
    egoTin_to_seq = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=Tin).to(device)
    process(model, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq)


def test_temporal_pointbevmodel(cfg_temporalpointbev):
    device = "cuda"
    cfg = cfg_temporalpointbev
    Tin = 2

    cfg.model.net.temporal.cam_T_P = [[i, 0] for i in range(Tin)]
    model = hydra.utils.instantiate(cfg.model.net)
    model.to(device)
    imgs = torch.randn(B, Tin, N, C, H, W).to(device)
    rots = torch.randn(B, Tin, N, 3, 3).to(device)
    trans = torch.randn(B, Tin, N, 3, 1).to(device)
    intrins = torch.randn(B, Tin, N, 3, 3).to(device)
    bev_aug = torch.randn(B, Tin, 4, 4).to(device)
    egoTin_to_seq = repeat(torch.eye(4), "i j -> b tin i j", b=B, tin=Tin).to(device)
    process(model, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq)
