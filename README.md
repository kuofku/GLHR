![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

# Global-to-Local Hierarchical Representation Learning for Maritime Vessel Re-Identification

The repository contains the *official* pytorch implementations for "Global-to-Local Hierarchical Representation Learning for Maritime Vessel Re-Identification".

*Code and others will be released after the paper is accepted.*
## Pipeline
<!--
![framework](figs/framework.jpg)

In this paper, we propose a global-to-local hierarchical representation learning (GLHR) method for maritime vessel re-identification. Through hierarchical feature aggregation (HFA), we construct a descriptor that balances global robustness with local discriminability. In addition, we incorporate structural alignment re-ranking (GLSA) and Top-
𝑘 Patch Match (TPM) to mitigate mismatches caused by complex backgrounds and variations in viewpoint and scale.
-->
## Requirements

### Installation

```bash
pip install -r requirements.txt
(we use /torch 1.10.0 /torchvision 0.11.0 /timm 1.0.11 /cuda 11.3 / 24G 3090 for training and evaluation.
```

### Prepare Datasets

```bash
mkdir data
```

Download the vessel datasets [WarshipReID](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [VesselReID](https://arxiv.org/abs/1711.08565), Then unzip them and rename them under the directory like:
(We adopt the vesselreid normal split of the VesselReID benchmark. You can also use your own dataset and organize it into this format.)

```
data
├── WarshipReID
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query
├── VesselReID
│   └── bounding_box_train
│   └── bounding_box_test
│   └── query

```

### Prepare Biformer Pre-trained Models

You need to download the ImageNet pretrained biformer model [Biformer-Base](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBa0JiY3pkUmxadkNoR3NYRnFBQS1QVm5BLVI4P2U9SVBsT0NH&cid=C29B955137735B40&sb=name&sd=1&id=C29B955137735B40%21626&parId=C29B955137735B40%21620&o=OneUp)
or download from Biformer repository: [Biformer](https://github.com/rayleizhu/BiFormer)
## Training

We utilize 1  GPU for training.

```bash
python train.py --config_file configs/GLHR.yml MODEL.DEVICE_ID "('your device id')" MODEL.PRETRAIN_PATH ${MODEL_DIR} DATASETS.ROOT_DIR ${DATASETS_DIR} OUTPUT_DIR ${OUTPUT_DIR}
```

#### Arguments

- `${DATASETS_DIR}`: folder for saving datasets, e.g. `../data/datasets`
- `${MODEL_DIR}`: folder for saving biformer pretrained models, e.g. `../pretrained/biformer_base_best.pth`
- `${OUTPUT_DIR}`: folder for saving logs and checkpoints, e.g. `../logs/GLHR/`

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Examples:**

```bash
# VesselReID
python test.py --config_file configs/DukeMTMC/GLHR.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT './logs/GLHR_VesselReID.pth'
```

## Trained Models

<table>
  <thead>
    <tr>
      <th style="text-align:center;">Datasets</th>
      <th style="text-align:center;">mAP</th>
      <th style="text-align:center;">Rank-1</th>
      <th style="text-align:center;">Rank-5</th>
      <th style="text-align:center;">Rank-10</th>
      <th style="text-align:center;">.pth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;"><strong>WarshipReID</strong></td>
      <td style="text-align:center;">98.6</td>
      <td style="text-align:center;">100.0</td>
      <td style="text-align:center;">100.0</td>
      <td style="text-align:center;">100.0</td>
      <td style="text-align:center;">
        <!--
        <a href="https://drive.google.com/file/d/143jqspOD6oM4XoEZ62EHnAYeB6COBeWa/view?usp=sharing">model</a> |
        <a href="https://pan.baidu.com/s/1ZFNwjlQ6G1YHm4CePLT1Gw?pwd=pd5p">model</a>
        -->
      </td>
    </tr>
    <tr>
      <td style="text-align:center;"><strong>VesselReID</strong></td>
      <td style="text-align:center;">72.4</td>
      <td style="text-align:center;">77.0</td>
      <td style="text-align:center;">89.9</td>
      <td style="text-align:center;">93.5</td>
      <td style="text-align:center;">
        <!--
        <a href="https://drive.google.com/file/d/1er-dEZQBZwlPAUTL-PGz3LvjjoZ7SSI8/view?usp=sharing">model</a> |
        <a href="https://pan.baidu.com/s/1fnh4X9XFDj5rAAIWWi50jA?pwd=ctmu">model</a>
        -->
      </td>
    </tr>
    <tr>
      <td style="text-align:center;"><strong>LSDV</strong></td>
      <td style="text-align:center;">77.6</td>
      <td style="text-align:center;">88.9</td>
      <td style="text-align:center;">94.9</td>
      <td style="text-align:center;">97.0</td>
      <td style="text-align:center;"> ——
        <!--
        <a href="https://drive.google.com/file/d/19-Rnba2eJSCryTj8VQfoLJfDWin3ox3l/view?usp=sharing">test.txt</a>
        -->
      </td>
    </tr>
  </tbody>
</table>

## Visualization
<!--
![Visualization](figs/Visualization.jpg)-->
You can visualize the results of GLHR by use “show_pic_rank_2” function in processor.py

## Acknowledgement

Codebase from [TransReid](https://github.com/damo-cv/TransReID)

## Contact

If you have any question, please feel free to contact us. E-mail: [eamon_fan@hnu.edu.cn](mailto:eamon_fan@hnu.edu.cn) 

