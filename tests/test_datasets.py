""" 
Check if the dataset runs fine and if we get the desired outputs.
Allows for fast debugging of the dataset.
"""

import hydra
import pytest

B = 2
Tin = 4
Nq = 2


@pytest.fixture
def get_temporal_dataloader(cfg_pointbev_global):
    cfg = cfg_pointbev_global
    assert cfg
    assert cfg.data
    cfg.data.version = "mini"
    cfg.data.batch_size = B
    cfg.data.num_workers = 1
    cfg.trainer.strategy = "none"
    cfg.data.cam_T_P = [[-2, 1], [0, 0], [1, 0], [2, 0]]
    cfg.data.bev_T_P = [[0, 0], [3, 0]]
    cfg.data.keep_input_binimg = True
    cfg.data.keep_input_centr_offs = True
    cfg.data.keep_input_detection = True
    cfg.data.keep_input_hdmap = False
    cfg.data.keep_input_persp = True
    cfg.data.keep_input_sampling = True
    cfg.data.keep_input_offsets_map = True
    cfg.data.keep_input_lidar = True
    cfg.data.img_params.min_visibility = 2
    cfg.data.visualise_mode = True
    cfg = cfg.copy()

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("only_val")
    return datamodule


def test_temporal_dataloader(get_temporal_dataloader):
    val_data = get_temporal_dataloader.valdata
    assert len(val_data.choose_cams()) == 6

    dataloader = get_temporal_dataloader.val_dataloader()

    elem = next(iter(dataloader))
    elem = get_temporal_dataloader.on_after_batch_transfer(elem, 0)

    print(elem.keys())
    rots = elem["rots"]
    trans = elem["trans"]
    imgs = elem["imgs"]
    intrins = elem["intrins"]
    egoTin_to_seq = elem["egoTin_to_seq"]
    egoTout_to_seq = elem["egoTout_to_seq"]
    bev_aug = elem["bev_aug"]
    binimg = elem["binimg"]
    binimg_aug = elem["binimg_aug"]
    # B,Tin,N,3,3
    assert rots.shape == (B, Tin, 6, 3, 3)
    # B,Tin,N,3
    assert trans.shape == (B, Tin, 6, 3, 1)
    # B,Tin,N,C,H,W
    assert imgs.shape == (B, Tin, 6, 3, 224, 480)
    # B,Tin,N,3,3
    assert intrins.shape == (B, Tin, 6, 3, 3)
    # B,Tin,4,4
    assert bev_aug.shape == (B, Tin, 4, 4)
    # B,Nq,4,4
    assert egoTin_to_seq.shape == (B, Tin, 4, 4)
    # B,Nq,4,4
    assert egoTout_to_seq.shape == (B, Nq, 4, 4)
    # B,Nq,1,200,200
    assert binimg.shape == (B, Nq, 1, 200, 200)
    # B,Nq,1,200,200
    assert binimg_aug.shape == (B, Nq, 1, 200, 200)
