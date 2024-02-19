# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import tempfile
from collections import OrderedDict
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from mmengine.checkpoint.io import (_load_checkpoint_with_prefix,
                                    get_state_dict, load_checkpoint,
                                    load_state_dict, save_checkpoint)
from mmengine.fileio.file_client import PetrelBackend
from mmengine.registry import MODEL_WRAPPERS
from mmengine.testing import assert_tensor_equal


@MODEL_WRAPPERS.register_module()
class DDPWrapper:

    def __init__(self, module):
        self.module = module


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv = nn.Conv2d(3, 3, 1)


def test_get_state_dict():
    if torch.__version__ == 'parrots':
        state_dict_keys = {
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'conv.weight', 'conv.bias'
        }
    else:
        state_dict_keys = {
            'block.conv.weight', 'block.conv.bias', 'block.norm.weight',
            'block.norm.bias', 'block.norm.running_mean',
            'block.norm.running_var', 'block.norm.num_batches_tracked',
            'conv.weight', 'conv.bias'
        }

    model = Model()
    state_dict = get_state_dict(model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys

    assert_tensor_equal(state_dict['block.conv.weight'],
                        model.block.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'], model.block.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        model.block.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'], model.block.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        model.block.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        model.block.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(state_dict['block.norm.num_batches_tracked'],
                            model.block.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'], model.conv.weight)
    assert_tensor_equal(state_dict['conv.bias'], model.conv.bias)

    wrapped_model = DDPWrapper(model)
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict['block.conv.weight'],
                        wrapped_model.module.block.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        wrapped_model.module.block.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        wrapped_model.module.block.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'],
                        wrapped_model.module.block.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        wrapped_model.module.block.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        wrapped_model.module.block.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(
            state_dict['block.norm.num_batches_tracked'],
            wrapped_model.module.block.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'],
                        wrapped_model.module.conv.weight)
    assert_tensor_equal(state_dict['conv.bias'],
                        wrapped_model.module.conv.bias)

    # wrapped inner module
    for name, module in wrapped_model.module._modules.items():
        module = DataParallel(module)
        wrapped_model.module._modules[name] = module
    state_dict = get_state_dict(wrapped_model)
    assert isinstance(state_dict, OrderedDict)
    assert set(state_dict.keys()) == state_dict_keys
    assert_tensor_equal(state_dict['block.conv.weight'],
                        wrapped_model.module.block.module.conv.weight)
    assert_tensor_equal(state_dict['block.conv.bias'],
                        wrapped_model.module.block.module.conv.bias)
    assert_tensor_equal(state_dict['block.norm.weight'],
                        wrapped_model.module.block.module.norm.weight)
    assert_tensor_equal(state_dict['block.norm.bias'],
                        wrapped_model.module.block.module.norm.bias)
    assert_tensor_equal(state_dict['block.norm.running_mean'],
                        wrapped_model.module.block.module.norm.running_mean)
    assert_tensor_equal(state_dict['block.norm.running_var'],
                        wrapped_model.module.block.module.norm.running_var)
    if torch.__version__ != 'parrots':
        assert_tensor_equal(
            state_dict['block.norm.num_batches_tracked'],
            wrapped_model.module.block.module.norm.num_batches_tracked)
    assert_tensor_equal(state_dict['conv.weight'],
                        wrapped_model.module.conv.module.weight)
    assert_tensor_equal(state_dict['conv.bias'],
                        wrapped_model.module.conv.module.bias)


def test_load_checkpoint():

    class PrefixModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.backbone = Model()

    pmodel = PrefixModel()
    model = Model()
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')

    # add prefix
    torch.save(model.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(
        pmodel, checkpoint_path, revise_keys=[(r'^', 'backbone.')])
    for key in pmodel.backbone.state_dict().keys():
        assert torch.equal(pmodel.backbone.state_dict()[key], state_dict[key])
    # strip prefix
    torch.save(pmodel.state_dict(), checkpoint_path)
    state_dict = load_checkpoint(
        model, checkpoint_path, revise_keys=[(r'^backbone\.', '')])

    for key in state_dict.keys():
        key_stripped = re.sub(r'^backbone\.', '', key)
        assert torch.equal(model.state_dict()[key_stripped], state_dict[key])
    os.remove(checkpoint_path)


def test_load_checkpoint_metadata():

    class ModelV1(nn.Module):

        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv1 = nn.Conv2d(3, 3, 1)
            self.conv2 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv1.weight)
            nn.init.normal_(self.conv2.weight)

    class ModelV2(nn.Module):
        _version = 2

        def __init__(self):
            super().__init__()
            self.block = Block()
            self.conv0 = nn.Conv2d(3, 3, 1)
            self.conv1 = nn.Conv2d(3, 3, 1)
            nn.init.normal_(self.conv0.weight)
            nn.init.normal_(self.conv1.weight)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  *args, **kwargs):
            """load checkpoints."""

            # Names of some parameters in has been changed.
            version = local_metadata.get('version', None)
            if version is None or version < 2:
                state_dict_keys = list(state_dict.keys())
                convert_map = {'conv1': 'conv0', 'conv2': 'conv1'}
                for k in state_dict_keys:
                    for ori_str, new_str in convert_map.items():
                        if k.startswith(prefix + ori_str):
                            new_key = k.replace(ori_str, new_str)
                            state_dict[new_key] = state_dict[k]
                            del state_dict[k]

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          *args, **kwargs)

    model_v1 = ModelV1()
    model_v1_conv0_weight = model_v1.conv1.weight.detach()
    model_v1_conv1_weight = model_v1.conv2.weight.detach()
    model_v2 = ModelV2()
    model_v2_conv0_weight = model_v2.conv0.weight.detach()
    model_v2_conv1_weight = model_v2.conv1.weight.detach()
    ckpt_v1_path = os.path.join(tempfile.gettempdir(), 'checkpoint_v1.pth')
    ckpt_v2_path = os.path.join(tempfile.gettempdir(), 'checkpoint_v2.pth')

    # Save checkpoint
    save_checkpoint(model_v1.state_dict(), ckpt_v1_path)
    save_checkpoint(model_v2.state_dict(), ckpt_v2_path)

    # test load v1 model
    load_checkpoint(model_v2, ckpt_v1_path)
    assert torch.allclose(model_v2.conv0.weight, model_v1_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v1_conv1_weight)

    # test load v2 model
    load_checkpoint(model_v2, ckpt_v2_path)
    assert torch.allclose(model_v2.conv0.weight, model_v2_conv0_weight)
    assert torch.allclose(model_v2.conv1.weight, model_v2_conv1_weight)


def test_load_checkpoint_with_prefix():

    class FooModule(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 2)
            self.conv2d = nn.Conv2d(3, 1, 3)
            self.conv2d_2 = nn.Conv2d(3, 2, 3)

    model = FooModule()
    nn.init.constant_(model.linear.weight, 1)
    nn.init.constant_(model.linear.bias, 2)
    nn.init.constant_(model.conv2d.weight, 3)
    nn.init.constant_(model.conv2d.bias, 4)
    nn.init.constant_(model.conv2d_2.weight, 5)
    nn.init.constant_(model.conv2d_2.bias, 6)

    with TemporaryDirectory() as tmp_dir:
        path = osp.join(tmp_dir, 'model.pth')
        torch.save(model.state_dict(), path)
        prefix = 'conv2d'
        state_dict = _load_checkpoint_with_prefix(prefix, path)
        assert torch.equal(model.conv2d.state_dict()['weight'],
                           state_dict['weight'])
        assert torch.equal(model.conv2d.state_dict()['bias'],
                           state_dict['bias'])

        # test whether prefix is in pretrained model
        with pytest.raises(AssertionError):
            prefix = 'back'
            _load_checkpoint_with_prefix(prefix, path)


def test_save_checkpoint(tmp_path):
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # meta is not a dict
    with pytest.raises(TypeError):
        save_checkpoint(model, '/path/of/your/filename', meta='invalid type')

    # 1. save to disk
    filename = str(tmp_path / 'checkpoint1.pth')
    save_checkpoint(model.state_dict(), filename)

    filename = str(tmp_path / 'checkpoint2.pth')
    checkpoint = dict(
        model=model.state_dict(), optimizer=optimizer.state_dict())
    save_checkpoint(checkpoint, filename)

    filename = str(tmp_path / 'checkpoint3.pth')
    save_checkpoint(
        model.state_dict(), filename, backend_args={'backend': 'local'})

    filename = str(tmp_path / 'checkpoint4.pth')
    save_checkpoint(
        model.state_dict(), filename, file_client_args={'backend': 'disk'})

    # 2. save to petrel oss
    with patch.object(PetrelBackend, 'put') as mock_method:
        filename = 's3://path/of/your/checkpoint1.pth'
        save_checkpoint(model.state_dict(), filename)
    mock_method.assert_called()

    with patch.object(PetrelBackend, 'put') as mock_method:
        filename = 's3://path//of/your/checkpoint2.pth'
        save_checkpoint(
            model.state_dict(),
            filename,
            file_client_args={'backend': 'petrel'})
    mock_method.assert_called()


def test_load_state_dict_post_hooks():
    module = Block()

    state_dict = {
        'conv.weight': torch.empty((3, 3, 1, 1), dtype=torch.float32),
        'conv.bias': torch.empty((3, ), dtype=torch.float32),
        'norm.weight': torch.empty([3], dtype=torch.float32),
        'norm.bias': torch.empty([3], dtype=torch.float32),
        'norm.running_mean': torch.empty([3], dtype=torch.float32),
        'norm.running_var': torch.empty([3], dtype=torch.float32),
    }
    state_dict.pop('norm.running_var')

    with patch('mmengine.checkpoint.io.print_log') as mock:
        load_state_dict(module, state_dict, strict=False)
        mock.assert_called_once()

    def post_hook(_, incompatible_keys):
        incompatible_keys.missing_keys.remove('norm.running_var')

    module._load_state_dict_post_hooks = {0: post_hook}

    with patch('mmengine.checkpoint.io.print_log') as mock:
        load_state_dict(module, state_dict, strict=False)
        mock.assert_not_called()
