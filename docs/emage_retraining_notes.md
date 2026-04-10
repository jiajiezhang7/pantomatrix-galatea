# 新版 PantoMatrix 下的 EMAGE 主模型重训练备忘

日期：2026-04-10

本文档用于固定当前工作区关于 `EMAGE` 重训练的关键结论、边界和注意事项。当前已经确认：后续若要继续提升 `EMAGE`，默认采用 **新版 `PantoMatrix`** 路线，而不是回到 `legacy EMAGE_2024`。

## 1. 当前决策

- 后续 `EMAGE` 相关工作，默认以上游快照 `third_party/PantoMatrix/` 为主。
- 在当前 `hybrid/lam-emage-pipeline` 分支中，`EMAGE` 的职责是 **body baseline**。
- 当前分支的人脸驱动是 provider 化的，生产路径不是 `EMAGE face`，而是 `lam` / `a2f-3d-sdk` 等独立 face provider。
- 因此，当前语境下的 “重训 EMAGE”，核心目标是：
  - 提升 body / full-motion baseline
  - 保持与当前 hybrid 推理链路兼容
  - 不以恢复 `EMAGE face` 为主要目标

## 2. 新版代码到底能不能训练

可以。

新版 `PantoMatrix` **不是只有推理和部署代码**，它已经包含：

- 训练入口：`third_party/PantoMatrix/train_emage_audio.py`
- 训练配置：`third_party/PantoMatrix/configs/emage_audio.yaml`
- `BEAT2` 数据加载器
- validation / test 循环
- 评估指标接入
- `save_pretrained(...)` 风格的模型导出

因此，如果目标是训练一个新的 **audio-only EMAGE 主模型**，新版代码可以直接承担这件事，不需要为了“只是重训主模型”而回退到 legacy。

## 3. 新版代码训练的到底是什么

这是最重要的边界。

新版代码训练的不是“完整原始论文版 EMAGE 全套”，而是：

- **新版定义下的 audio-only EMAGE 主模型**
- 其上层模型名可视为 `EmageAudioModel`

从当前代码事实来看，新版训练流程会：

- 先加载预训练的 `VQ / VAE` 子模块
- 将这些子模块冻结
- 在其上训练音频到动作的主模型

所以，“用新版重训 EMAGE”在当前工作区里应理解为：

- 重训 **主模型**
- 复用已有 latent motion 相关子模块

而不应理解为：

- 完整复现原始论文的训练配方
- 在新版代码中从零重训所有 `VQ / VAE`
- 恢复老版的 `Audio + Text` 训练路线

## 4. 新版路线适合什么目标

当目标是下面这些时，应优先使用新版：

- 提升当前 hybrid 分支的 `EMAGE` body baseline
- 训练一个新的 audio-only 模型并接回当前推理链路
- 保持和 `tools/run_emage_audio_inference.py` 的兼容
- 保持当前工作区已经成型的工程基础设施
- 尽量减少和 legacy 训练栈重新耦合的成本

这也是当前工作区的默认推荐路线。

## 5. 什么情况下才需要回退到 legacy

只有在目标明显超出新版能力边界时，才考虑回到 `third_party/PantoMatrix_legacy/scripts/EMAGE_2024/`。

典型情况包括：

- 需要尽量忠实复现原始 EMAGE 论文路线
- 需要 `Audio + Text` 训练路径
- 需要重训底层 `VQ / VAE` 组件
- 需要恢复老版研究代码中独有的训练行为

如果目标只是：

- “给当前 hybrid body lane 训一个更强的主模型”

那么 legacy 不应是第一选择。

## 6. 当前工作区下的训练契约

### 训练目标

- 主训练目标：`EmageAudioModel`
- 主代码路径：`third_party/PantoMatrix/`
- 默认训练入口：`train_emage_audio.py`

### 推理兼容目标

未来任何重训产物，都应优先保证兼容以下入口：

- `tools/run_emage_audio_inference.py`
- `tools/hybrid_pipeline.py`

这意味着最终模型包应尽量保持当前加载契约不变：

- 一个主模型目录
- 配套的 `emage_vq/face`
- 配套的 `emage_vq/upper`
- 配套的 `emage_vq/lower`
- 配套的 `emage_vq/hands`
- 配套的 `emage_vq/global`

如果我们只重训主模型、继续复用现有预训练 latent 子模块，那么这一兼容性最好维护。

## 7. 数据和外部资产前提

新版训练链路依赖若干仓库外资产；代码本身不是全自包含的。

至少需要准备：

- `BEAT2` 数据集
- `emage_audio` 预训练资产
- `emage_evaltools`
- `foot_contact` 预处理产物

当前工作区中可用的辅助材料：

- `tools/download_emage_assets.py`
- `tools/bootstrap_emage_env.sh`
- `docs/external-assets-redownload.md`

务必注意：

- 仅有源码仓库，无法直接完成完整训练
- 数据、评估工具、预训练 latent 资产必须先到位

## 8. 推荐的重训策略

### 第一步：先冻结问题范围

开始前，先明确这次实验属于哪一种：

- `仅重训主模型`
- `调整数据集 / split`
- `调整 loss / scheduler`
- `修改模型结构`

第一轮实验不要把这些变量同时混在一起。

### 第二步：第一轮尽量保守

最稳妥的第一轮重训建议是：

- 保持新版现有架构不变
- 保持 `VQ / VAE` 子模块冻结
- 保持当前数据格式和 loader 路径
- 只调整数据选择、训练超参、实验命名和 checkpoint 管理

这样最有机会训出一个仍然能接回当前 hybrid pipeline 的可用模型。

### 第三步：先跑通短程训练验证

在正式长跑前，先确认：

- dataset metadata JSON 可正常解析
- `BEAT2` motion / expression / trans / foot-contact 文件都可访问
- `emage_evaltools` 能正常加载
- debug 模式或短程训练能真正写出 checkpoint

### 第四步：保存成可复用的模型包

一轮成功的重训应至少保留：

- `last/`
- `best/`
- 原始 optimizer / scheduler checkpoint
- 本次实验实际使用的 config

目标是让后续工具可以直接通过类似 `--model-root` 的方式切换到新模型。

## 9. 评估要求

新版代码已经有现成评估钩子，因此评估应视为默认流程，而不是“以后再看”。

至少应记录：

- `FGD`
- `BC`
- `L1div`
- `LVDFace`
- `MSEFace`
- train / val 曲线

但在当前工作区里，数值指标还不够。

还应验证：

- 生成的 `.npz` 是否可正常加载
- body 序列是否仍能在 `Viser` 中正确播放
- 输出是否仍能被当前 hybrid 编排链路消费

如果一个新模型指标更好，但破坏了 `.npz` 兼容性或 hybrid 接入能力，那么它不算当前分支的合格升级。

## 10. 主要风险

### 风险 1：把“新版主模型重训”误当成“完整原版 EMAGE 复现”

这是两个不同目标。

当前新版路线服务的是前者，不是后者。

### 风险 2：无意中破坏推理兼容性

即使只是小幅结构改动，也可能破坏：

- 当前 `from_pretrained(...)` 加载方式
- 当前 `tools/run_emage_audio_inference.py`
- 当前 hybrid body 生成契约

因此，前几轮实验优先做“兼容性保守型”修改。

### 风险 3：误以为新版已经具备完整 VQ 重训基础设施

当前新版 README 对 `train the vqvae` 仍是未完备状态。

在没有额外独立方案之前，不应把“基于新版完整从底层 latent stack 开始重训”当作默认计划。

### 风险 4：忘记当前分支上下文

当前分支中，`EMAGE face` 不是生产 face 路径。

因此，不应为了 EMAGE face 指标或表现去主导当前重训决策，除非分支目标本身发生变化。

## 11. 当前推荐默认策略

在需求没有变化前，默认执行以下策略：

- 默认重训路线：**新版 PantoMatrix**
- 默认目标：**提升 audio-only EMAGE 主模型**
- 默认兼容目标：**当前 hybrid 分支推理契约**
- 默认非目标：**完整 legacy 论文复现**
- 只有在明确需要 text-conditioning 或完整 latent-stack 重训时，才升级为 legacy 子项目

## 12. 快速决策规则

后续讨论时，可直接套用下面这条判断：

- “我们要给当前 hybrid body lane 训一个更强的模型。”
  走新版 `PantoMatrix` 训练路径。

- “我们要做完整原始 EMAGE 训练栈，包括旧版 text-conditioned 行为或更深的 latent 模块重训。”
  重新打开 legacy 路线，并将其视为一个独立子项目。
