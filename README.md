# Beyond Part Models: Person Retrieval with Refined Part Pooling

**Related Projects:** [Strong Triplet Loss Baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline)

This project implements PCB (Part-based Convolutional Baseline) of paper [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349) using [pytorch](https://github.com/pytorch/pytorch).


# Current Results

The reproduced PCB is as follows. 
- `(Shared 1x1 Conv)` and `(Independent 1x1 Conv)` means the last 1x1 conv layers for stripes are shared or independent, respectively;
- `(Paper)` means the scores reported in the paper; 
- `R.R.` means using re-ranking.


|                                   | Rank-1 (%) | mAP (%) | R.R. Rank-1 (%) | R.R. mAP (%) |
| ---                               | :---: | :---: | :---: | :---: |
| Market1501 (Shared 1x1 Conv)      | 90.86 | 73.25 | 92.58 | 88.02 |
| Market1501 (Independent 1x1 Conv) | 92.87 | 78.54 | 93.94 | 90.17 |
| Market1501 (Paper)                | 92.40 | 77.30 | -     | -     |
| | | | | |
| Duke (Shared 1x1 Conv)            | 82.00 | 64.88 | 86.40 | 81.77 |
| Duke (Independent 1x1 Conv)       | 84.47 | 69.94 | 88.78 | 84.73 |
| Duke (Paper)                      | 81.90 | 65.30 | -     | -     |
| | | | | |
| CUHK03 (Shared 1x1 Conv)          | 47.29 | 42.05 | 56.50 | 57.91 |
| CUHK03 (Independent 1x1 Conv)     | 59.14 | 53.93 | 69.07 | 70.17 |
| CUHK03 (Paper)                    | 61.30 | 54.20 | -     | -     |

We can see that independent 1x1 conv layers for different stripes are critical for the performance. The performance on CUHK03 is still worse than the paper, while those on Market1501 and Duke are better.


# Resources

This repository contains following resources

- A beginner-level dataset interface independent of Pytorch, Tensorflow, etc, supporting multi-thread prefetching (README file is under way)
- Three most used ReID datasets, Market1501, CUHK03 (new protocol) and DukeMTMC-reID
- Python version ReID evaluation code (Originally from [open-reid](https://github.com/Cysu/open-reid))
- Python version Re-ranking (Originally from [re_ranking](https://github.com/zhunzhong07/person-re-ranking/blob/master/python-version/re_ranking))
- PCB (Part-based Convolutional Baseline, performance stays tuned)


# Installation

It's recommended that you create and enter a python virtual environment, if versions of the packages required here conflict with yours.

I use Python 2.7 and Pytorch 0.3. For installing Pytorch, follow the [official guide](http://pytorch.org/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Then clone the repository:

```bash
git clone https://github.com/huanghoujing/beyond-part-models.git
cd beyond-part-models
```


# Dataset Preparation

Inspired by Tong Xiao's [open-reid](https://github.com/Cysu/open-reid) project, dataset directories are refactored to support a unified dataset interface.

Transformed dataset has following features
- All used images, including training and testing images, are inside the same folder named `images`
- Images are renamed, with the name mapping from original images to new ones provided in a file named `ori_to_new_im_name.pkl`. The mapping may be needed in some cases.
- The train/val/test partitions are recorded in a file named `partitions.pkl` which is a dict with the following keys
  - `'trainval_im_names'`
  - `'trainval_ids2labels'`
  - `'train_im_names'`
  - `'train_ids2labels'`
  - `'val_im_names'`
  - `'val_marks'`
  - `'test_im_names'`
  - `'test_marks'`
- Validation set consists of 100 persons (configurable during transforming dataset) unseen in training set, and validation follows the same ranking protocol of testing.
- Each val or test image is accompanied by a mark denoting whether it is from
  - query (`mark == 0`), or
  - gallery (`mark == 1`), or
  - multi query (`mark == 2`) set

## Market1501

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4) or [BaiduYun](https://pan.baidu.com/s/1nvOhpot). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## CUHK03

We follow the new training/testing protocol proposed in paper
```
@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```
Details of the new protocol can be found [here](https://github.com/zhunzhong07/person-re-ranking).

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1Ssp9r4g8UbGveX-9JvHmjpcesvw90xIF) or [BaiduYun](https://pan.baidu.com/s/1hsB0pIc). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Then download the training/testing partition file from [Google Drive](https://drive.google.com/open?id=14lEiUlQDdsoroo8XJvQ3nLZDIDeEizlP) or [BaiduYun](https://pan.baidu.com/s/1miuxl3q). This partition file specifies which images are in training, query or gallery set. Finally run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03
```


## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```


## Combining Trainval Set of Market1501, CUHK03, DukeMTMC-reID

Larger training set tends to benefit deep learning models, so I combine trainval set of three datasets Market1501, CUHK03 and DukeMTMC-reID. After training on the combined trainval set, the model can be tested on three test sets as usual.

Transform three separate datasets as introduced above if you have not done it.

For the trainval set, you can download what I have transformed from [Google Drive](https://drive.google.com/open?id=1hmZIRkaLvLb_lA1CcC4uGxmA4ppxPinj) or [BaiduYun](https://pan.baidu.com/s/1jIvNYPg). Otherwise, you can run the following script to combine the trainval sets, replacing the paths with yours.

```bash
python script/dataset/combine_trainval_sets.py \
--market1501_im_dir ~/Dataset/market1501/images \
--market1501_partition_file ~/Dataset/market1501/partitions.pkl \
--cuhk03_im_dir ~/Dataset/cuhk03/detected/images \
--cuhk03_partition_file ~/Dataset/cuhk03/detected/partitions.pkl \
--duke_im_dir ~/Dataset/duke/images \
--duke_partition_file ~/Dataset/duke/partitions.pkl \
--save_dir ~/Dataset/market1501_cuhk03_duke
```

## Configure Dataset Path

The project requires you to configure the dataset paths. In `bpm/dataset/__init__.py`, modify the following snippet according to your saving paths used in preparing datasets.

```python
# In file bpm/dataset/__init__.py

########################################
# Specify Directory and Partition File #
########################################

if name == 'market1501':
  im_dir = ospeu('~/Dataset/market1501/images')
  partition_file = ospeu('~/Dataset/market1501/partitions.pkl')

elif name == 'cuhk03':
  im_type = ['detected', 'labeled'][0]
  im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
  partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

elif name == 'duke':
  im_dir = ospeu('~/Dataset/duke/images')
  partition_file = ospeu('~/Dataset/duke/partitions.pkl')

elif name == 'combined':
  assert part in ['trainval'], \
    "Only trainval part of the combined dataset is available now."
  im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
  partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')
```

## Evaluation Protocol

Datasets used in this project all follow the standard evaluation protocol of Market1501, using CMC and mAP metric. According to [open-reid](https://github.com/Cysu/open-reid), the setting of CMC is as follows

```python
# In file bpm/dataset/__init__.py

cmc_kwargs = dict(separate_camera_set=False,
                  single_gallery_shot=False,
                  first_match_break=True)
```

To play with [different CMC options](https://cysu.github.io/open-reid/notes/evaluation_metrics.html), you can [modify it accordingly](https://github.com/Cysu/open-reid/blob/3293ca79a07ebee7f995ce647aafa7df755207b8/reid/evaluators.py#L85-L95).

```python
# In open-reid's reid/evaluators.py

# Compute all kinds of CMC scores
cmc_configs = {
  'allshots': dict(separate_camera_set=False,
                   single_gallery_shot=False,
                   first_match_break=False),
  'cuhk03': dict(separate_camera_set=True,
                 single_gallery_shot=True,
                 first_match_break=False),
  'market1501': dict(separate_camera_set=False,
                     single_gallery_shot=False,
                     first_match_break=True)}
```


# Examples


## Test PCB

My training log and saved model weights (trained with independent 1x1 conv) for three datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1G3mLsI1g8ZZkHyol6d3yHpygZeFsENqO?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1zfjeiePvr1TlBtu7yGovlQ).

Specify
- a dataset name (one of `market1501`, `cuhk03`, `duke`)
- an experiment directory for saving testing log
- the path of the downloaded `model_weight.pth`

in the following command and run it.

```bash
python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test true \
--dataset DATASET_NAME \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
```

## Train PCB

You can also train it by yourself. The following command performs training, validation and finally testing automatically.

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- training on `trainval` set or `train` set (for tuning parameters)
- an experiment directory for saving training log

in the following command and run it.

```bash
python script/experiment/train_pcb.py \
-d '(0,)' \
--only_test false \
--dataset DATASET_NAME \
--trainset_part TRAINVAL_OR_TRAIN \
--exp_dir EXPERIMENT_DIRECTORY \
--steps_per_log 20 \
--epochs_per_val 1
```

### Log

During training, you can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) and access port `6006` to watch the loss curves etc. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir YOUR_EXPERIMENT_DIRECTORY/tensorboard
```

For more usage of TensorBoard, see the website and the help:

```bash
tensorboard --help
```


## Visualize Ranking List

Specify
- a dataset name (one of `['market1501', 'cuhk03', 'duke']`)
- either `model_weight_file` (the downloaded `model_weight.pth`) OR `ckpt_file` (saved `ckpt.pth` during training)
- an experiment directory for saving images and log

in the following command and run it.

```bash
python script/experiment/visualize_rank_list.py \
-d '(0,)' \
--num_queries 16 \
--rank_list_size 10 \
--dataset DATASET_NAME \
--exp_dir EXPERIMENT_DIRECTORY \
--model_weight_file '' \
--ckpt_file ''
```

Each query image and its ranking list would be saved to an image in directory `EXPERIMENT_DIRECTORY/rank_lists`. As shown in following examples, green boundary is added to true positive, and red to false positve.

![](example_rank_lists_on_Market1501/00000156_0003_00000009.jpg)

![](example_rank_lists_on_Market1501/00000305_0001_00000001.jpg)

![](example_rank_lists_on_Market1501/00000492_0005_00000001.jpg)

![](example_rank_lists_on_Market1501/00000881_0002_00000006.jpg)


# Time and Space Consumption


Test with CentOS 7, Intel(R) Xeon(R) CPU E5-2618L v3 @ 2.30GHz, GeForce GTX TITAN X.

**Note that the following time consumption is not gauranteed across machines, especially when the system is busy.**

### GPU Consumption in Training

For following settings

- ResNet-50 `stride=1` in last block
- `batch_size = 64`
- image size `h x w = 384 x 128`

it occupies ~11000MB GPU memory.

If not having a 12 GB GPU, you can decrease `batch_size` or use multiple GPUs.


### Training Time

Taking Market1501 as an example, it contains `31969` training images; each epoch takes ~205s; training for 60 epochs takes ~3.5 hours.

### Testing Time

Taking Market1501 as an example
- With `images_per_batch = 32`, extracting feature of whole test set (12936 images) takes ~160s.
- Computing query-gallery global distance, the result is a `3368 x 15913` matrix, ~2s
- Computing CMC and mAP scores, ~15s
- Re-ranking requires computing query-query distance (a `3368 x 3368` matrix) and gallery-gallery distance (a `15913 x 15913` matrix, most time-consuming), ~90s


# References & Credits

- [Beyond Part Models: Person Retrieval with Refined Part Pooling](https://arxiv.org/abs/1711.09349)
- [open-reid](https://github.com/Cysu/open-reid)
- [Re-ranking Person Re-identification with k-reciprocal Encoding](https://github.com/zhunzhong07/person-re-ranking)
- [Market1501](http://www.liangzheng.org/Project/project_reid.html)
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation)
