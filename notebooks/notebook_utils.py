import sys

sys.path.insert(0, "../")
import json
import math
import os
from copy import deepcopy
from itertools import chain
from pathlib import Path
from pprint import pprint
from typing import List

import hydra
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyrootutils
import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
from hydra import compose, initialize
from hydra.utils import instantiate
from nuscenes.utils.splits import create_splits_scenes
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms.functional import affine, resize
from torchvision.utils import make_grid
from tqdm import trange

from pointbev import utils
from pointbev.utils.imgs import update_intrinsics


def show(imgs, cmap=None, fig_size=(10, 10)):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=fig_size)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = np.asarray(F.to_pil_image(img))

        # Matplotlib expect 3 or 4 channels.
        if img.shape[2] == 2:
            raise ValueError("2 channels not supported")
        axs[0, i].imshow(img, cmap=cmap)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
