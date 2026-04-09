**Hybrid Pipeline Technical Specification**  
**目的**：为 code agent 提供一份完整、清晰、可执行的技术文档，用于在已完全跑通的 EMAGE 基础上实现 hybrid 方案，最终生成一套时间戳对齐的参数文件，可直接驱动 Viser（body 验证）和 Unreal Engine ARKit（face + eye 实时对话）。

### 1. Hybrid Pipeline 总体架构
整个方案采用模块化设计，将数字人驱动分解为三个独立但可同步的模块：

1. **Body 模块**：使用 EMAGE 生成的 SMPL-X 参数（包含 body、hands 和全局平移），保留 EMAGE 在全身手势与肢体连贯性上的最大优势。  
2. **Face Expression 模块**：切换到 LAM_Audio2Expression 生成的 51/52 个标准 ARKit blendshapes 系数，实现直接兼容 Unreal ARKit 且具备原生实时能力。  
3. **Eye/Gaze + Blink 模块**：叠加 TalkingEyes 生成的显式 eyeball rotation（左右眼 3D 旋转参数）和 blink 信息。若 LAM_Audio2Expression 的 eye 相关系数在实际测试中已足够自然，可先省略 TalkingEyes，只保留前两个模块。

所有模块共享同一段单声道 wav 音频作为输入（推荐 16kHz 单声道）。最终通过一个融合层按帧时间戳对齐并合并参数，输出统一的结果文件夹。

### 2. 各模块详细职责与输出要求

**Body 模块（EMAGE）**  
- 保持现有 EMAGE 推理流程不变。  
- 输出要求：完整的 SMPL-X 参数序列（pose、shape、global translation），帧率保持 30fps。  
- 作用：提供全身自然手势和肢体运动，确保 body 与手势高度同步。

**Face Expression 模块（LAM_Audio2Expression）**  
- 替换 EMAGE 原生的 FLAME face 参数。  
- 输出要求：每帧输出 51/52 个标准 ARKit blendshapes 系数（0~1 范围），包含 lip-sync 和 upper face 表情。  
- 作用：提供直接可用于 Unreal ARKit 的面部表情系数，实现零转换成本和实时驱动。

**Eye/Gaze + Blink 模块（TalkingEyes）**  
- 可选叠加模块。  
- 输出要求：每帧输出左右眼的显式 eyeball rotation 参数（3D 旋转）以及 blink 概率/强度。  
- 作用：补充真实眼球注视方向和眨眼运动，提升对话时的眼神交流自然度。  
- 测试策略：先单独运行 LAM_Audio2Expression，检查其 eye 相关 blendshapes 系数的效果；若注视和眨眼已足够逼真，则无需引入 TalkingEyes；若不足，再叠加 TalkingEyes 并仅提取其 eye 部分。

### 3. 整体执行流程

1. **准备阶段**：确保 EMAGE 已完全跑通，使用同一段测试 wav 文件分别运行三个模块，生成各自独立的输出文件。  
2. **推理阶段**：三个模块可并行处理同一段音频（推荐使用多进程方式实现并行推理）。  
3. **融合阶段**：编写融合逻辑，按音频总时长和统一帧率（默认 30fps）对齐三个模块的输出参数，处理可能的时序偏移。  
4. **输出阶段**：生成标准结果文件夹，包含 body 参数文件、face ARKit 参数文件和 eye 参数文件（可选）。  
5. **验证阶段**：使用 Viser 播放 body 参数，使用 Unreal Engine 加载 ARKit blendshapes + eye 参数进行端到端视觉验证。

### 4. 融合层核心逻辑要求
融合层需完成以下工作：
- 以音频总时长为基准，将三个模块的输出序列强制对齐到同一帧序列（默认 30fps）。  
- 处理可能的帧数差异（通过线性插值或截断实现）。  
- 保留 EMAGE 的 body 参数完整性，仅替换 face 和 eye 部分。  
- 可选添加平滑处理（例如轻量滤波），避免融合后出现抖动。  
- 输出统一格式的结果文件夹，方便后续 Viser 和 Unreal 直接读取。

### 5. ToDo Checklist（code agent 执行顺序）

- [ ] 下载并完成 LAM_Audio2Expression 的环境搭建与预训练权重准备。  
- [ ] 下载并完成 TalkingEyes 的环境搭建与预训练权重准备。  
- [ ] 使用同一段短测试 wav 文件分别运行 EMAGE、LAM_Audio2Expression 和 TalkingEyes，确认每个模块均能正常输出参数文件。  
- [ ] 实现并行推理逻辑，让三个模块同时处理同一段音频。  
- [ ] 实现融合逻辑，按时间戳对齐并合并三个模块的参数。  
- [ ] 生成标准结果文件夹并进行 Viser + Unreal 端到端验证。  
- [ ] 测试端到端延迟，目标控制在合理范围内（适合实时对话场景）。  
- [ ] （可选）若 LAM_Audio2Expression 的 eye 效果已足够，则移除 TalkingEyes 模块以简化 pipeline。

### 6. 注意事项与风险控制

- **时序对齐**：三个模型默认输出帧率可能存在细微差异，必须严格以音频时长为基准进行对齐。  
- **Eye 部分处理**：优先测试 LAM_Audio2Expression 的 eye 相关系数效果，确认不足后再引入 TalkingEyes，仅提取其 eye 部分以避免 face 重复。  
- **风格一致性**：LAM_Audio2Expression 具有风格嵌入能力，可通过调整使其表情风格尽量接近 EMAGE 的输出。  
- **GPU 资源**：推荐使用单卡 24GB 以上显存的机器，或采用分卡并行推理方式。  
- **测试流程**：先用 10 秒左右的短音频跑通完整 pipeline，再逐步扩展到完整对话音频。  
- **下游兼容**：最终输出的 ARKit blendshapes 可直接用于 Unreal Engine，body 参数可直接用于 Viser 验证，无需额外转换。  

此文档已固定所有方法细节和逻辑流程。code agent 可根据实际环境自行实现具体代码和脚本。  

---

## 7. Retarget 暂停说明（2026-04-09）

当前 `EMAGE .npz -> BEAT_Avatars(HumGen)` 这条高品质数字人渲染链已暂停，不继续自动实现。

暂停原因：
- 已确认 Blender headless、官方 SMPL-X addon、官方 `BEAT_Avatars.zip`、官方 `rendervideo.zip` 都可以获取并部分验证。
- 但未找到官方公开的、现成可复用的 `EMAGE .npz -> BEAT_Avatars(HumGen)` 自动 retarget 代码/脚本/流水线。
- `BEAT_Avatars` 内的 `HG_*` 角色骨架与 `SMPL-X` 55 关节骨架并非同构，不能安全假设可直接自动绑定。
- 继续推进需要人工参与的骨骼映射、偏移调节和视觉验收，这超出当前可靠自动化范围。

当前建议：
- 先将 `EMAGE -> BEAT_Avatars` 视为后续同事接手的 Blender/rigging 子任务。
- 若需要继续自动出片，建议改走：
  - `EMAGE .npz -> 官方 SMPL-X Blender addon -> Blender 中的 SMPL-X角色渲染`

当前进展：
- 这条 `SMPL-X Blender addon` 降级路线已经在本工作区落地并验证。
- 当前推荐的自动渲染方案见：
  - `docs/blender-smplx-render.md`

详细交接信息见：
- `docs/retarget-handoff-2026-04-09.md`
