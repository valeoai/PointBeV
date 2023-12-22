# Official PyTorch Implementation of [PointBeV: A Sparse Approach to BeV Predictions](https://arxiv.org/abs/2312.00703)

> [**PointBeV: A Sparse Approach to BeV Predictions**](https://arxiv.org/abs/2312.00703)<br>
> [Loick Chambon](https://loickch.github.io/), [Eloi Zablocki](https://scholar.google.fr/citations?user=dOkbUmEAAAAJ&hl=fr), [Mickael Chen](https://sites.google.com/view/mickaelchen/), [Florent Bartoccioni](https://f-barto.github.io/), [Patrick Perez](https://ptrckprz.github.io/), [Matthieu Cord](https://cord.isir.upmc.fr/).<br> Valeo AI, Sorbonne University

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="imgs/iou_vs_mem.png" width="320">
    </td>
    <td align="center">
      <img src="imgs/iou_vs_mem2.png" width="320">
    </td>
  </tr>

  <tr>
    <td align="center">
      <em>PointBeV reaches state-of-the-art on several segmentation tasks (vehicle without filtering above) while allowing a trade-off between performance and memory consumption.</em>
    </td>
    <td align="center">
      <em>PointBeV reaches state-of-the-art on several segmentation tasks (vehicle with filtering above). It can also be used using different pattern strategies, for instance a LiDAR pattern.</em>
    </td>
  </tr>

  <tr>
  <td colspan="2" align="center">
    <img src="imgs/overview.png" alt="alt text" width="700">
  </td>

  <tr>
    <td colspan="2" align="center">
      <em>Illustration of different sampling pattern, respectively: a full, a regular, a drivable hdmap, a lane hdmap, a front camera and a LiDAR pattern. PointBeV is flexible to any pattern.</em>
    </td>
  </tr>

</tr>
</table>
</div>

# Abstract
*We propose PointBeV, a novel sparse BeV segmentation model operating on sparse BeV features instead of dense grids. This approach offers precise control over memory usage, enabling the use of long temporal contexts and accommodating memory-constrained platforms. PointBeV employs an efficient two-pass strategy for training, enabling focused computation on regions of interest. At inference time, it can be used with various memory/performance trade-offs and flexibly adjusts to new specific use cases. PointBeV achieves state-of-the-art results on the nuScenes dataset for vehicle, pedestrian, and lane segmentation, showcasing superior performance in static and temporal settings despite being trained solely with sparse signals.*

<table>
  <tr>
  <td align="center">
    <img src="imgs/archi.png">
  </td>
  </tr>
  
  <tr>
  <td align="center">
    <em>PointBeV architecture is an architecture dealing with sparse representations. It uses an efficient Sparse Feature Pulling module to propagate features from images to BeV and a Sparse Attention module for temporal aggregation.</em>
  </td>
  </tr>

</table>

# 🚀 Main results

## 🔥 Vehicle segmentation
PointBeV is originally designed for vehicle segmentation. It can be used with different sampling patterns and different memory/performance trade-offs. It can also be used with temporal context to improve the segmentation.
<div align="center">
<table border="1">
  <caption><i>Vehicle segmentation of various static models at 448x800 image resolution with visibility filtering. More details can be found in our paper.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2312.00703">PointBeV (ours)</a></th>
        <th><a href="https://openaccess.thecvf.com/content/CVPR2023/html/Pan_BAEFormer_Bi-Directional_and_Early_Interaction_Transformers_for_Birds_Eye_View_CVPR_2023_paper.html">BAEFormer</a></th>
        <th><a href="https://arxiv.org/abs/2206.07959">SimpleBeV</a></th>
        <th><a href="https://arxiv.org/abs/2203.17270">BEVFormer</a></th>
        <th><a href="https://arxiv.org/abs/2205.02833">CVT</a></th>
    </tr>
    <tr class="highlight-column">
        <td>IoU</td>
        <td>47.6</td>
        <td>41.0</td>
        <td>46.6</td>
        <td>45.5</td>
        <td>37.7</td>
    </tr>
</table>
</div>
Below we illustrate the model output. On the ground truth, we distinguish vehicle with low visibility (vis < 40%) in light blue from those with higher visibility (vis > 40%) in dark blue. We can see that PointBeV is able to segment vehicles with low visibility, which is a challenging task for other models. They often correspond to occluded vehicles.

<img src='./imgs/vehicle_segm.gif'>

## 🔥 Pedestrian and lane segmentation

PointBeV can also be used for different segmentation tasks such as pedestrians or hdmap segmentation.
<div align="center">
<table border="1">
  <caption><i>Pedestrian segmentation of various static models at 224x480 resolution. More details can be found in our paper.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2312.00703">PointBeV (ours)</a></th>
        <th>TBP-Former</th>
        <th>ST-P3</th>
        <th>FIERY</th>
        <th>LSS</th>
    </tr>
    <tr class="highlight-column">
        <td>IoU</td>
        <td>18.5</td>
        <td>17.2</td>
        <td>14.5</td>
        <td>17.2</td>
        <td>15.0</td>
    </tr>
</table>
</div>
<img src='./imgs/pedes_segm.gif'>

## 🔥 Lane segmentation
<div align="center">
<table border="1">
  <caption><i>Lane segmentation of various static models at different resolutions. More details can be found in our paper.</i></caption>
    <tr>
        <th>Models</th>
        <th><a href="https://arxiv.org/abs/2312.00703">PointBeV (ours)</a></th>
        <th>MatrixVT</th>
        <th>M2BeV</th>
        <th>PeTRv2</th>
        <th>BeVFormer</th>
    </tr>
    <tr class="highlight-column">
        <td>IoU</td>
        <td>49.6</td>
        <td>44.8</td>
        <td>38.0</td>
        <td>44.8</td>
        <td>25.7</td>
    </tr>
</table>
</div>

# 🔨 Setup <a name="setup"></a>

Coming soon.

# 🔄 Training <a name="training"></a>

Coming soon. 

# 🔄 Evaluation <a name="evaluating"></a>

Coming soon. 

# 👍 Acknowledgements

Many thanks to these excellent open source projects:
* https://github.com/nv-tlabs/lift-splat-shoot
* https://github.com/aharley/simple_bev
* https://github.com/fundamentalvision/BEVFormer

To structure our code we used:
https://github.com/ashleve/lightning-hydra-template

# 📝 BibTeX

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{chambon2023pointbev,
      title={PointBeV: A Sparse Approach to BeV Predictions}, 
      author={Loick Chambon and Eloi Zablocki and Mickael Chen and Florent Bartoccioni and Patrick Perez and Matthieu Cord},
      year={2023},
      eprint={2312.00703},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# 📜 License

This project is released under the MIT license