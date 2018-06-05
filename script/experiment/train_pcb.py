from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from bpm.dataset import create_dataset
from bpm.model.PCBModel import PCBModel as Model

from bpm.utils.utils import time_str
from bpm.utils.utils import str2bool
from bpm.utils.utils import may_set_mode
from bpm.utils.utils import load_state_dict
from bpm.utils.utils import load_ckpt
from bpm.utils.utils import save_ckpt
from bpm.utils.utils import set_devices
from bpm.utils.utils import AverageMeter
from bpm.utils.utils import to_scalar
from bpm.utils.utils import ReDirectSTD
from bpm.utils.utils import set_seed
from bpm.utils.utils import adjust_lr_staircase


class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(384, 128))
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1)

    parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
    # When the stride is changed to 1, we can compensate for the receptive field
    # using dilated convolution. However, experiments show dilated convolution is useless.
    parser.add_argument('--last_conv_dilation', type=int, default=1, choices=[1, 2])
    parser.add_argument('--num_stripes', type=int, default=6)
    parser.add_argument('--local_conv_out_channels', type=int, default=256)

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--new_params_lr', type=float, default=0.1)
    parser.add_argument('--finetuned_params_lr', type=float, default=0.01)
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(41,))
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.1)
    parser.add_argument('--total_epochs', type=int, default=60)

    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.train_mirror_type = 'random' if args.mirror else None
    self.train_batch_size = args.batch_size
    self.train_final_batch = False
    self.train_shuffle = True

    self.test_mirror_type = None
    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      batch_size=self.train_batch_size,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # The last block of ResNet has stride 2. We can set the stride to 1 so that
    # the spatial resolution before global pooling is doubled.
    self.last_conv_stride = args.last_conv_stride
    # When the stride is changed to 1, we can compensate for the receptive field
    # using dilated convolution. However, experiments show dilated convolution is useless.
    self.last_conv_dilation = args.last_conv_dilation
    # Number of stripes (parts)
    self.num_stripes = args.num_stripes
    # Output channel of 1x1 conv
    self.local_conv_out_channels = args.local_conv_out_channels

    #############
    # Training  #
    #############

    self.momentum = 0.9
    self.weight_decay = 0.0005

    # Initial learning rate
    self.new_params_lr = args.new_params_lr
    self.finetuned_params_lr = args.finetuned_params_lr
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.
    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')
    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()

    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    try:
      local_feat_list, logits_list = self.model(ims)
    except:
      local_feat_list = self.model(ims)
    feat = [lf.data.cpu().numpy() for lf in local_feat_list]
    feat = np.concatenate(feat, axis=1)

    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return feat


def main():
  cfg = Config()

  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  train_set = create_dataset(**cfg.train_set_kwargs)
  num_classes = len(train_set.ids2labels)
  # The combined dataset does not provide val set currently.
  val_set = None if cfg.dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(
    last_conv_stride=cfg.last_conv_stride,
    num_stripes=cfg.num_stripes,
    local_conv_out_channels=cfg.local_conv_out_channels,
    num_classes=num_classes
  )
  # Model wrapper
  model_w = DataParallel(model)

  #############################
  # Criteria and Optimizers   #
  #############################

  criterion = torch.nn.CrossEntropyLoss()

  # To finetune from ImageNet weights
  finetuned_params = list(model.base.parameters())
  # To train from scratch
  new_params = [p for n, p in model.named_parameters()
                if not n.startswith('base.')]
  param_groups = [{'params': finetuned_params, 'lr': cfg.finetuned_params_lr},
                  {'params': new_params, 'lr': cfg.new_params_lr}]
  optimizer = optim.SGD(
    param_groups,
    momentum=cfg.momentum,
    weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  if cfg.resume:
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  TMO(modules_optims)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=True,
        verbose=True)

  def validate():
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n===== Test on validation set =====\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=True,
      to_re_rank=False,
      verbose=True)
    print()
    return mAP, cmc_scores[0]

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, cfg.total_epochs):

    # Adjust Learning Rate
    adjust_lr_staircase(
      optimizer.param_groups,
      [cfg.finetuned_params_lr, cfg.new_params_lr],
      ep + 1,
      cfg.staircase_decay_at_epochs,
      cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    # For recording loss
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))
      labels_var = Variable(TVT(torch.from_numpy(labels).long()))

      _, logits_list = model_w(ims_var)
      loss = torch.sum(
        torch.cat([criterion(logits, labels_var) for logits in logits_list]))

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        log = '\tStep {}/Ep {}, {:.2f}s, loss {:.4f}'.format(
          step, ep + 1, time.time() - step_st, loss_meter.val)
        print(log)

    #############
    # Epoch Log #
    #############

    log = 'Ep {}, {:.2f}s, loss {:.4f}'.format(
      ep + 1, time.time() - ep_st, loss_meter.avg)
    print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
      mAP, Rank1 = validate()

    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
        ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  test(load_model_weight=False)


if __name__ == '__main__':
  main()
