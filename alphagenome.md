# AlphaGenome 微调指南与硬件优化策略

## 1. 什么是 AlphaGenome？
AlphaGenome 是由 Google DeepMind 开发的基础基因组大模型（发布于 2026 年初）。
*   **核心突破**：能够同时处理 **1 Mb (一百万碱基)** 的超长 DNA 上下文，并保持 **1 bp (单碱基)** 的极高分辨率。
*   **多模态预测**：通过 1D 和 2D 混合架构（CNN + Transformer + U-Net），同时预测基因表达、染色质可及性、3D 结构和剪接变异等 11 种模态。

## 2. 整合 AlphaGenome 与 gReLU
我们已经通过原生的 PyTorch 移植版 `alphagenome-pytorch` 子模块，将 AlphaGenome 无缝集成到了 gReLU 基础设施中。

**整合实现细节：**
1.  **Trunk 封装 (`AlphaGenomeTrunk`)**：在 `src/grelu/model/trunks/alphagenome.py` 中实现了特定的 Trunk 层。它负责处理 gReLU 标准 `(Batch, 4, Length)` 到 AlphaGenome `(Batch, Length, 4)` 的张量维度转换。
2.  **Model 封装 (`AlphaGenomeModel`)**：在 `src/grelu/model/models.py` 中，通过继承 gReLU 的 `BaseModel`，将 `AlphaGenomeTrunk` 嵌入。
3.  **多模态支持**：模型实例化时支持通过 `output_key` 和 `resolution` 参数自由切换预期的输出模态（例如 `atac`, `dnase`, `rna_seq`），并能在推断时自动应用 AlphaGenome 特有的上采样与激活机制（通过 `model.predict`）。
4.  **使用方法**：可以直接向 `grelu.lightning.LightningModel` 传入 `model_params={"model_type": "AlphaGenomeModel", "output_key": "atac"}`，从而像使用内置的 `Borzoi` 一样调用 gReLU 强大的数据加载器和下游变异预测等分析流程（具体可参考 `run_alphagenome_inference.py`）。

## 3. 针对 4 x RTX 6000 Ada (192GB VRAM) 的终极微调策略
当前硬件配置（4 张 48GB 显卡）极为强悍，但 1Mb 序列的计算仍易引发 OOM。必须采用以下极端的显存优化组合策略：

| 优化维度 | 核心技术 | 配置建议 (基于 PyTorch Lightning / gReLU) | 收益 / 原理 |
| :--- | :--- | :--- | :--- |
| **多卡并行** | **FSDP** (完全分片数据并行) | `strategy=FSDPStrategy(sharding_strategy="FULL_SHARD")` | 将模型参数、梯度、优化器状态打碎分布到 4 张卡上，单卡显存压力骤降。 |
| **数据精度** | **BF16 混合精度** | `precision="bf16-mixed"` | RTX Ada 架构原生加速，显存减半且梯度稳定，避免 fp16 的溢出问题。 |
| **计算引擎** | **Flash Attention 2** | 模型代码底层必须调用 `flash_attn_func` | 消除 Transformer 层 $O(N^2)$ 的显存与计算时间暴增。 |
| **显存置换** | **Gradient Checkpointing** (梯度检查点) | `model.gradient_checkpointing_enable()` | 丢弃前向传播的中间激活值，反向传播时重算，用时间换空间。 |
| **收敛稳定** | **Gradient Accumulation** (梯度累加) | `accumulate_grad_batches=8` | 长序列下单卡 Batch Size 极小，通过累积多次 Step 的梯度进行更新，平滑训练。 |

**执行预期：**
在上述策略加持下，配合 LoRA，该 4 卡硬件环境有极大可能支持在 **256kb 甚至 512kb** 的超长上下文窗口下完成 AlphaGenome 级别的微调任务。