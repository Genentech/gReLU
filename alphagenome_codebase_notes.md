# AlphaGenome-PyTorch 子模块架构分析与集成指南

## 1. 源码库结构
`alphagenome-pytorch` 位于 `src/alphagenome_pytorch/` 目录，其核心 PyTorch 模型定义在 `src/alphagenome_pytorch/src/alphagenome_pytorch/` 下。

关键文件及其职责：
*   **`model.py`**: 定义了核心的 `AlphaGenome` 类。它包含一个 U-Net 类型的卷积主干（`dna_embedder`, `down_blocks`, `up_blocks`）以及预测头。
*   **`attention.py`**: 包含 Transformer / 注意力机制的实现。AlphaGenome 具有 9 个 Transformer blocks，并且针对部分 block 存在针对 Pair (2D) 交互的更新逻辑 (`PairUpdateBlock`, `AttentionBiasBlock`)。
*   **`convolutions.py`**: 包含 1D 卷积模块（如 `DnaEmbedder`），负责最初的序列特征提取。
*   **`heads.py`**: 定义了多种基因组预测任务的输出头（ATAC, CAGE, Splicing, Contact Maps 等 11 种模态）。
*   **`config.py`**: 类型和精度策略 (`DtypePolicy`) 配置，以严格对齐原始 JAX 版本的 bf16 计算精度。

## 2. 核心架构特性与显存优化支持

在浏览源码后，发现该移植库已经内置了非常多我们在微调时必须用到的显存优化特性，这极大地降低了我们整合的难度：

1.  **内置梯度检查点 (Gradient Checkpointing)**：
    *   `AlphaGenome` 和 Transformer blocks 的初始化函数中都暴露了 `self.gradient_checkpointing` 标志。只需将其设为 `True`，模型即可在反向传播时自动应用检查点，这解决了我们在长序列微调时最大的显存痛点。
2.  **输入维度要求 (NLC vs NCL)**：
    *   gReLU 默认的输入格式是 `(Batch, 4, Length)` (NCL 格式)。
    *   `AlphaGenome` 的 `forward` 函数接收 `dna_sequence` 形状为 `(B, S, 4)` (NLC 格式)，并在内部第一步执行了 `x.transpose(1, 2)`。
    *   **注意**：在做 gReLU 的 Wrapper 时，我们需要把 gReLU 传进来的 `(Batch, 4, Length)` 用 `permute(0, 2, 1)` 转换为模型期望的 `(Batch, Length, 4)`。
3.  **多物种索引 (Organism Index)**：
    *   AlphaGenome 前向传播强制需要 `organism_index` 参数（0 为 Human，1 为 Mouse）。在使用 gReLU 的单数据流时，我们需要在 Wrapper 中自动注入这个常量 Tensor。
4.  **精度策略控制 (DtypePolicy)**：
    *   该库专门实现了 `DtypePolicy` 来控制混合精度。在 RTX Ada 显卡上，我们可以直接传入 `DtypePolicy.mixed_precision()` 以利用原生的 bf16 加速。

## 3. gReLU 适配器 (Wrapper) 设计原型

为了使 `AlphaGenome` 能够作为 `grelu.lightning.LightningModel` 的引擎运行，我们需要编写一个接口对齐类：

```python
import torch
import torch.nn as nn
from alphagenome_pytorch.src.alphagenome_pytorch.model import AlphaGenome
from alphagenome_pytorch.src.alphagenome_pytorch.config import DtypePolicy

class AlphaGenomeGreluWrapper(nn.Module):
    def __init__(self, organism_idx=0, gradient_checkpointing=True):
        super().__init__()
        # 初始化 AlphaGenome，启用梯度检查点和 bf16 混合精度
        self.model = AlphaGenome(
            num_organisms=2,
            dtype_policy=DtypePolicy.mixed_precision(),
            gradient_checkpointing=gradient_checkpointing
        )
        self.organism_idx = organism_idx
        
    def forward(self, x):
        # 1. 维度对齐: gReLU 传入 (B, 4, L)，转换成 AlphaGenome 需要的 (B, L, 4)
        x = x.permute(0, 2, 1)
        
        # 2. 调用前向传播，注入 organism_index
        # 注意: 预测头会输出一个字典或元组，可能需要进一步处理以对齐 gReLU 的损失函数格式
        outputs = self.model(dna_sequence=x, organism_index=self.organism_idx)
        
        # 3. 抽取所需的 task_logits (例如只取 'atac' 或提取全部 1D 轨道拼接成 Tensor)
        # 根据微调目标自定义提取逻辑
        return outputs
```

## 4. 下一步行动计划
1.  **连通性测试**：编写一个小型 Python 脚本，实例化 `AlphaGenomeGreluWrapper`，喂入一个 dummy 张量（如 64kb 序列），验证 `forward` 能够无报错运行并输出预测字典。
2.  **权重加载验证**：尝试下载并加载转换后的 AlphaGenome `.pth` 预训练权重。
3.  **微调损失对齐**：由于 AlphaGenome 输出的是 11 种模态的字典，而 gReLU 的 LightningModel 默认期望一个单一的 Tensor 以计算 Loss，需要编写一个特化的 Head 解析器来对齐损失函数。
