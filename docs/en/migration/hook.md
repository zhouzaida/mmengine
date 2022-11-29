# Migrate Hook from MMCV to MMEngine

## Introduction

Due to the upgraded architecture and increasing user requirements, MMCV's hook points are no longer sufficient for the needs, so they are redesigned and the hook functions are adjusted in MMEngine. Before starting the migration, read [Hook Design](../design/hook.md) would be helpful.

This article compares the hooks of [MMCV v1.6.0](https://github.com/open-mmlab/mmcv/tree/v1.6.0) and [MMEngine v0.1.0](https://github.com/open-mmlab/mmengine/tree/v0.1.0) in terms of function, point, usage and implementation.

## Differences in functionaliy

<table class="docutils">
<thead>
  <tr>
    <th></th>
    <th>MMCV</th>
    <th>MMEngine</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Perform back propagation and update gradient</td>
    <td>OptimizerHook</td>
    <td rowspan="2">将反向传播以及梯度更新的操作抽象成 <a href="../tutorials/optim_wrapper.html">OptimWrapper</a> 而不是钩子</td>
  </tr>
  <tr>
    <td>GradientCumulativeOptimizerHook</td>
  </tr>
  <tr>
    <td>Learning rate scheduler</td>
    <td>LrUpdaterHook</td>
    <td rowspan="2">ParamSchdulerHook 以及 <a href="../tutorials/param_scheduler.html">_ParamScheduler</a> 的子类完成优化器超参的调整</td>
  </tr>
  <tr>
    <td>Momentum scheduler</td>
    <td>MomentumUpdaterHook</td>
  </tr>
  <tr>
    <td>Save checkpoints periodicall</td>
    <td>CheckpointHook</td>
    <td rowspan="2">CheckpointHook 除了保存权重，还有保存最优权重的功能，而 EvalHook 的模型评估功能则交由 ValLoop 或 TestLoop 完成</td>
  </tr>
  <tr>
    <td>Evaluate model and save best checkpoint</td>
    <td>EvalHook</td>
  </tr>
  <tr>
    <td>Print log</td>
    <td rowspan="3">LoggerHook and its subclasses to print (save) logs and visualization</td>
    <td>LoggerHook</td>
  </tr>
  <tr>
    <td>Visualization</td>
    <td>NaiveVisualizationHook</td>
  </tr>
  <tr>
    <td>添加运行时信息</td>
    <td>RuntimeInfoHook</td>
  </tr>
  <tr>
    <td>Apply Exponential Moving Average (EMA) on the model during training</td>
    <td>EMAHook</td>
    <td>EMAHook</td>
  </tr>
  <tr>
    <td>Ensure the shuffle of distributed Sampler is active</td>
    <td>DistSamplerSeedHook</td>
    <td>DistSamplerSeedHook</td>
  </tr>
  <tr>
    <td>Synchronize model buffers</td>
    <td>SyncBufferHook</td>
    <td>SyncBufferHook</td>
  </tr>
  <tr>
    <td>Releases all unoccupied cached GPU memory</td>
    <td>EmptyCacheHook</td>
    <td>EmptyCacheHook</td>
  </tr>
  <tr>
    <td>Time consumption during iteration</td>
    <td>IterTimerHook</td>
    <td>IterTimerHook</td>
  </tr>
  <tr>
    <td>Collect the performance metrics during training and inference</td>
    <td>ProfilerHook</td>
    <td>Not available</td>
  </tr>
  <tr>
    <td>提供注册方法给钩子点位的功能</td>
    <td>ClosureHook</td>
    <td>Not available</td>
  </tr>
</tbody>
</table>

## 点位差异

<table class="docutils">
<thead>
  <tr>
    <th colspan="2"></th>
    <th class="tg-uzvj">MMCV</th>
    <th class="tg-uzvj">MMEngine</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">全局位点</td>
    <td>执行前</td>
    <td>before_run</td>
    <td>before_run</td>
  </tr>
  <tr>
    <td>执行后</td>
    <td>after_run</td>
    <td>after_run</td>
  </tr>
  <tr>
    <td rowspan="2">Checkpoint 相关</td>
    <td>加载 checkpoint 后</td>
    <td>无</td>
    <td>after_load_checkpoint</td>
  </tr>
  <tr>
    <td>保存 checkpoint 前</td>
    <td>无</td>
    <td>before_save_checkpoint</td>
  </tr>
  <tr>
    <td rowspan="6">训练相关</td>
    <td>训练前触发</td>
    <td>无</td>
    <td>before_train</td>
  </tr>
  <tr>
    <td>训练后触发</td>
    <td>无</td>
    <td>after_train</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>before_train_epoch</td>
    <td>before_train_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>after_train_epoch</td>
    <td>after_train_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>before_train_iter</td>
    <td>before_train_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>after_train_iter</td>
    <td>after_train_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td rowspan="6">验证相关</td>
    <td>验证前触发</td>
    <td>无</td>
    <td>before_val</td>
  </tr>
  <tr>
    <td>验证后触发</td>
    <td>无</td>
    <td>after_val</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>before_val_epoch</td>
    <td>before_val_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>after_val_epoch</td>
    <td>after_val_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>before_val_iter</td>
    <td>before_val_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>after_val_iter</td>
    <td>after_val_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td rowspan="6">测试相关</td>
    <td>测试前触发</td>
    <td>无</td>
    <td>before_test</td>
  </tr>
  <tr>
    <td>测试后触发</td>
    <td>无</td>
    <td>after_test</td>
  </tr>
  <tr>
    <td>每个 epoch 前</td>
    <td>无</td>
    <td>before_test_epoch</td>
  </tr>
  <tr>
    <td>每个 epoch 后</td>
    <td>无</td>
    <td>after_test_epoch</td>
  </tr>
  <tr>
    <td>每次迭代前</td>
    <td>无</td>
    <td>before_test_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td>每次迭代后</td>
    <td>无</td>
    <td>after_test_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
</tbody>
</table>

## Differences in usage

In MMCV, to register hooks to the Runner, you need to call the `register_training_hooks` method of the Runner to register hooks, while in MMEngine, you can register hooks by passing hooks to the initialization method of the Runner.

- MMCV

```python
model = ResNet18()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr_config = dict(policy='step', step=[2, 3])
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=5)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
runner = EpochBasedRunner(
    model=model,
    optimizer=optimizer,
    work_dir='./work_dir',
    max_epochs=3,
    xxx,
)
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config,
    custom_hooks_config=custom_hooks,
)
runner.run([trainloader], [('train', 1)])
```

- MMEngine

```python
model=ResNet18()
optim_wrapper=dict(
    type='OptimizerWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9))
param_scheduler = dict(type='MultiStepLR', milestones=[2, 3]),
default_hooks = dict(
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
)
custom_hooks = [dict(type='NumClassCheckHook')]
runner = Runner(
    model=model,
    work_dir='./work_dir',
    optim_wrapper=optim_wrapper,
    param_scheduler=param_scheduler,
    train_cfg=dict(by_epoch=True, max_epochs=3),
    default_hooks=default_hooks,
    custom_hooks=custom_hooks,
    xxx,
)
runner.train()
```

For more usage of MMEngine hooks, please refer to \[Hook Usage\](. /tutorials/hook.md).

## Differences in implementation

以 `CheckpointHook` 为例，MMEngine 的 [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py) 相比 MMCV 的 [CheckpointHook](https://github.com/open-mmlab/mmcv/blob/v1.6.0/mmcv/runner/hooks/checkpoint.py)（新增保存最优权重的功能，在 MMCV 中，保存最优权重的功能由 EvalHook 提供），因此，它需要实现 `after_val_epoch` 点位。

- MMCV

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""
```

- MMEngine

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""

    def after_val_epoch(self, runner, metrics):
        """根据 metrics 保存最优权重"""
```
