# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from typing import Dict, List, Optional, Union

import deepspeed
import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper, _ParamScheduler
from mmengine.registry import MODEL_WRAPPERS, STRATEGIES
from .base_strategy import BaseStrategy


@STRATEGIES.register_module()
class DeepSpeedStrategy(BaseStrategy):
    """

    Args:
        zero_optimization (dict, optional):
        fp16 (dict, optional):
    """

    def __init__(
        self,
        *,
        # the following args are for deepspeed
        config: Union[str, dict, None] = None,
        zero_optimization: Optional[dict] = None,
        fp16: Optional[dict] = None,
        inputs_to_half: Optional[List[Union[int, str]]] = None,
        bf16: Optional[dict] = None,
        amp: Optional[dict] = None,
        activation_checkpointing: Optional[dict] = None,
        aio: Optional[dict] = None,
        steps_per_print:
        int = 10000000000000,  # disable the log printed by deepseed
        # the following args are for BaseStrategy
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = self._parse_config(config)
        if zero_optimization is not None:
            self.config['zero_optimization'] = zero_optimization
        if fp16 is not None:
            self.config['fp16'] = fp16
        if bf16 is not None:
            self.config['bf16'] = bf16
        if amp is not None:
            self.config['amp'] = amp
        if activation_checkpointing is not None:
            self.config['activation_checkpointing'] = activation_checkpointing
        if aio is not None:
            self.config['aio'] = aio

        self.config['steps_per_print'] = steps_per_print

        self._inputs_to_half = inputs_to_half

    def _parse_config(self, config):
        if config is None:
            config = dict()
        elif isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        return config

    def setup_distributed(self, launcher=None, backend='nccl', **kwargs):
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        deepspeed.init_distributed(dist_backend=backend)

    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Optional[Union[OptimWrapper, dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        compile_target: str = 'forward',
        checkpoint: Optional[dict] = None,
        train_batch_size: Optional[int] = None,
        num_batches_per_epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
        max_iters: Optional[int] = None,
        cur_iter: Optional[int] = None,
        **kwargs,
    ):
        """Prepare model and some components.

        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It
                can be a dict used for build a model.

        Kwargs:
            optim_wrapper (OptimWrapper or dict, optional):
                Computing gradient of model parameters. If specified,
                :attr:`train_dataloader` should also be specified. If automatic
                mixed precision or gradient accmulation
                training is required. The type of ``optim_wrapper`` should be
                AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
                examples. Defaults to None.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optimizer` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
            compile_target (str): The method of model to be compiled.
                Defaults to 'forward'.
            checkpoint (dict, optional): Checkpoint to load strategy state.
                Defaults to None.
            train_batch_size (int, optional): Batch size of training. It will
                be used to scale the learning rate. Defaults to None.
            num_batches_per_epoch (int, optional): Number of batches per epoch.
                Defaults to None.
            max_epochs (int, optional): Number of epochs. Defaults to None.
            max_iters (int, optional): Number of iterations. Defaults to None.
            cur_iter (int, optional): Current iteration. Defaults to None.
        """
        return_items = []

        self.model = self.build_model(model)
        self.model = self._init_model_weights(self.model)
        return_items.append(self.model)

        self.config['train_batch_size'] = train_batch_size

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper)

            self.model = self.wrap_model(self.model)
            self.optim_wrapper.optimizer = self.model.optimizer
            return_items.clear()
            return_items.append(self.model)
            return_items.append(self.optim_wrapper)

        if param_scheduler is not None:
            _default_args = {}
            if num_batches_per_epoch is not None:
                _default_args['epoch_length'] = num_batches_per_epoch
            if max_epochs is not None:
                _default_args['max_epochs'] = max_epochs
            if max_iters is not None:
                _default_args['max_iters'] = max_iters

            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, _default_args)
            return_items.append(self.param_schedulers)

        if checkpoint is None:
            checkpoint = self.load_or_resume()
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

        return return_items[0] if len(return_items) == 1 else return_items

    def wrap_model(self, model: nn.Module) -> nn.Module:
        wrapper_cfg = dict(
            type='MMDeepSpeedEngine',
            model=model,
            optimizer=self.optim_wrapper.optimizer,
            config=self.config,
            inputs_to_half=self._inputs_to_half,
        )
        model = MODEL_WRAPPERS.build(wrapper_cfg)
        return model
