# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Union

from mmengine.model import BaseModel
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.hooks import RuntimeInfoHook as _RuntimeInfoHook
from mmengine.runner import Runner



@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, *args, **kwargs):
        return {'loss': torch.tensor(0.0)}


def update_params_step(self, loss):
    pass


def runtimeinfo_step(self, runner, batch_idx, data_batch=None):
    runner.message_hub.update_info('iter', runner.iter)
    lr_dict = runner.optim_wrapper.get_lr()
    for name, lr in lr_dict.items():
        runner.message_hub.update_scalar(f'train/{name}', lr[0])

    momentum_dict = runner.optim_wrapper.get_momentum()
    for name, momentum in momentum_dict.items():
        runner.message_hub.update_scalar(f'train/{name}', momentum[0])


@patch('mmengine.optim.optimizer.OptimWrapper.update_params',
       update_params_step)
@patch('mmengine.hooks.RuntimeInfoHook.before_train_iter', runtimeinfo_step)
def fake_run(cfg):
    from mmengine.runner import Runner
    cfg.pop('model')
    cfg.pop('visualizer')
    cfg.pop('val_dataloader')
    cfg.pop('val_evaluator')
    cfg.pop('val_cfg')
    cfg.pop('test_dataloader')
    cfg.pop('test_evaluator')
    cfg.pop('test_cfg')
    extra_cfg = dict(
        model=dict(type='ToyModel'),
        visualizer=dict(
            type='Visualizer',
            vis_backends=[
                dict(type='TensorboardVisBackend', save_dir='temp_dir')
            ]),
    )
    cfg.merge_from_dict(extra_cfg)
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


class RuntimeInfoHook(_RuntimeInfoHook):

    def before_train_iter(self, runner, batch_idx, data_batch = None) -> None:
        runner.message_hub.update_info('iter', runner.iter)

        lr_dict = runner.optim_wrapper.get_lr()
        for name, lr in lr_dict.items():
            runner.message_hub.update_scalar(f'train/{name}', lr[0])

        momentum_dict = runner.optim_wrapper.get_momentum()
        for name, momentum in momentum_dict.items():
            runner.message_hub.update_scalar(f'train/{name}', momentum[0])


def visualize_scheduler(
    cfg: Union[str, dict, Config, None] = None,
    *,
    by_epoch: bool = True,
    epochs: int = 1,

    param_list: tuple = ('lr', 'momentum'),
    dataset_size: Optional[str] = None,
    num_gpus: int = 1,
    window_size: tuple = (12, 7),
    work_dir: Optional[str] = None,
):
    """Visualize parameter scheduler.
    
    Args:
        cfg (str, optional):
        param_list
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)
    
    dataloader = DataLoader()
    runner = Runner(
        model = ToyModel(),
        work_dir = work_dir,
        train_dataloader = dataloader,
    )