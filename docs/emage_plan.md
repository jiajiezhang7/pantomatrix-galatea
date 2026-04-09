**PantoMatrix / EMAGE 完整实施文档**  
**目标**：实现 **streaming-ready**、**显式可解释参数** 输出（Body: SMPL-X → SMPL-H 兼容；Face: FLAME → ARKit blendshapes），支持 wav（+可选 text）输入，直接接入 **Viser player**（验证 body 参数）和 **Unreal Engine ARKit facial**（下游驱动）。  

**文档版本**：2026.4.8（基于官方 GitHub README、HF Space、arXiv 2401.00374v5、BEAT2 数据集及工具全网验证）  
**适用场景**：你的 code agent 自动化 pipeline（安装 → 推理 → post-process → Viser/Unreal 验证）

### 1. 方案概述（为什么选择 EMAGE / PantoMatrix）
- **模型**：EMAGE（CVPR 2024，主推）是 PantoMatrix 项目中最 holistic 的模型，支持 **full-body + face** 联合生成。
- **输入**：单声道 wav（16kHz 推荐）；text 可选（提升语义手势，但非必须）。
- **输出**（显式参数，完全满足你的核心要求）：
  - **Body**：SMPL-X 参数（pose θ ∈ ℝ^{T×55×3} axis-angle / Rot6D、shape β、global translation γ）。**可直接转 SMPL-H**（忽略 face/hand 额外维度）。
  - **Face**：FLAME 参数（expression 100维 + jaw 3维）。**官方提供 ARKit → FLAME 映射**（51 ARKit blendshapes → FLAME params，含 mat_final.npy 和 handcrafted templates）。
- **格式**：单个 `.npz` 文件（BEAT-compatible layout），可被 Viser 直接加载。
- **Streaming 适配**：**非原生 streaming**（non-autoregressive Transformer + VQ-VAE），但：
  - 支持 windowed / frame-by-frame 推理（Colab 示例已演示）。
  - 可导出 **ONNX**（官方未直接提供，但 setup.sh 环境支持 PyTorch → ONNX 转换）。
  - 低延迟路径：用 `test_emage_audio.py` + 短音频 chunk 实现伪 streaming（~实时在 GPU 上）。
- **Demo 成熟度**：HF Space 一键试跑 + Colab + Blender addon + ARKit 转换脚本全开源。

**核心优势**（匹配你的需求）：
- 显式 SMPL-X + FLAME → Viser/Unreal 零成本接入。
- ARKit 映射官方提供（ARkit_FLAME.zip + mat_final.npy）。
- 社区活跃（2025 年仍有更新）。

### 2. 完整 ToDo List（给 code agent 的自动化 Checklist）
**Phase 0: 准备环境（一次性）**
- [ ] Clone 仓库：`git clone https://github.com/PantoMatrix/PantoMatrix.git && cd PantoMatrix`
- [ ] 运行 `bash setup.sh`（自动创建 py39 venv + 安装依赖）
- [ ] （可选）下载 BEAT2_Tools（HF）：`ARKit_FLAME.zip` + `mat_final.npy` + SMPLX 模型
- [ ] 下载 Blender addon（可视化验证）：`smplx_blender_addon_20230921.zip`

**Phase 1: 推理 Pipeline（核心自动化脚本）**
- [ ] 准备输入文件夹：`my_audio/`（放入 .wav 文件，推荐 16kHz mono）
- [ ] 运行 EMAGE 推理（优先）：
  ```bash
  source py39/bin/activate
  python test_emage_audio.py \
    --audio_folder ./my_audio \
    --save_folder ./my_results \
    --visualization   # 可选（需 pytorch3d，否则加 --nopytorch3d）
  ```
- [ ] 输出检查：每个 wav → 对应 `.npz`（含 `motion_axis_angle` 等键）
- [ ] （可选）用其他模型测试：`test_camn_audio.py` / `test_disco_audio.py`（轻量版）

**Phase 2: Post-Processing（显式参数转换）**
- [ ] Body → Viser / SMPL-H：
  - 用 `beat_format_save` 或 np.load 提取 SMPL-X pose。
  - 转换脚本（仓库 emage_utils/format_transfer）：SMPL-X → SMPL-H（忽略 face/hand 维度）。
- [ ] Face → ARKit blendshapes（Unreal）：
  - 加载 FLAME params。
  - 用官方 ARKit2FLAME 脚本 + `mat_final.npy` 映射 → 51 维 ARKit blendshape weights。
  - 输出 `.json` / `.csv` 供 Unreal ARKit 驱动。
- [ ] 保存格式标准化：为每个 chunk 生成 timestamped params（便于 streaming）。

**Phase 3: 验证 & 可视化**
- [ ] Viser player：加载 `.npz` → 播放 body 动画（确认 SMPL-X 参数正确）。
- [ ] Blender addon：加载 SMPL-X + FLAME → 视频渲染验证。(暂时不做)
- [ ] Unreal Engine：导入 ARKit blendshapes → 实时 facial 测试。（暂时不做）

**Phase 4: Streaming 适配（伪实时 / 生产化）**
- [ ] 实现 audio chunking（每 1-2 秒一个 window）。
- [ ] 导出 ONNX（PyTorch → ONNX，Colab 有示例）。
- [ ] 部署：Edge GPU / ONNX Runtime → 帧级推理。
- [ ] 测试 latency（目标 <500ms）。

**Phase 5: 自动化 & 错误处理**
- [ ] 写 wrapper script（input wav → output npz + ARKit json）。
- [ ] 添加 logging / retry（pytorch3d 安装失败时 fallback）。
- [ ] CI 测试：用 examples/audio 跑端到端验证。

### 3. 方案注意细节 & 风险（必须告知 code agent）
- **模型局限**：
  - 训练于英文 Speaker 2 数据；非英语或强口音可能手势/表情弱。
  - Face 质量略低于 mocap 基准（但 body 强）。
  - 下肢运动语义较弱（依赖 audio rhythm）。
- **依赖坑**：
  - pytorch3d 安装易失败 → 始终加 `--nopytorch3d`。
  - CUDA 版本必须匹配 PyTorch 1.x / 2.x（setup.sh 会处理）。
  - 内存：EMAGE 全模型 ~8-12GB GPU。
- **输出细节**：
  - `.npz` 键：`motion_axis_angle`（body）、FLAME expression/jaw。
  - 帧率：默认 30fps（可配置）。
- **Streaming 真实性**：
  - 当前是非自回归 → windowed 推理可模拟 streaming。
  - 若需真 streaming，考虑 ONNX + 滑动窗口（未来可蒸馏为 autoregressive）。
- **法律/许可**：MIT / 研究用途（检查 HF 数据集许可）。
- **可视化**：
  - Blender addon 必须安装才能看 3D mesh。
  - Viser 直接支持 SMPL-X .npz。

### 4. 关键资源链接（直接复制给 code agent）
- 主仓库：https://github.com/PantoMatrix/PantoMatrix
- EMAGE 官方页 + 论文：https://pantomatrix.github.io/EMAGE/ + arXiv 2401.00374
- HF Inference Space（快速验证）：https://huggingface.co/spaces/H-Liu1997/EMAGE
- Colab（一键跑通）：https://colab.research.google.com/drive/1MeuZtBv8yUUG9FFeN8UGy78Plk4gzxT4
- ARKit2FLAME 工具：HF `BEAT2_Tools`（ARkit_FLAME.zip + mat_final.npy）
- Blender Addon：仓库 assets/

### 5. 最终输出格式建议（code agent 交付标准）
- 每个 wav → `{audio_name}.npz`（SMPL-X + FLAME）
- `{audio_name}_arkit.json`（51 维 blendshapes + timestamp）
- 日志文件 + 可视化视频