# gReLU 模型架构与外部模型整合原理

gReLU 主要基于 **PyTorch** 构建。为了使其训练、数据处理和下游分析（如变异效应预测、序列设计等）能够无缝工作，模型在底层必须是 PyTorch 的 `nn.Module` 对象。

## 1. gReLU 预置支持的模型
gReLU 内置了丰富的基因组序列分析常用架构（位于 `grelu.model` 模块中）：

*   **基础积木模型**：
    *   `ConvModel` (全卷积网络)
    *   `DilatedConvModel` (空洞卷积网络，适用于捕捉远距离特征，类似 ChromBPNet)
    *   `ConvGRUModel` (CNN + 循环神经网络 GRU)
    *   `ConvTransformerModel` (CNN + 局部/全局 Transformer 注意力机制)
    *   `ConvMLPModel` (CNN + 多层感知机)
*   **经典/大型基石架构 (Trunks)**：
    内置了学术界经典的生物序列大模型结构，可以直接实例化它们：
    *   `AlphaGenomeModel` (DeepMind 的百万碱基超长上下文预测模型，通过 `alphagenome-pytorch` 子模块接入)
    *   `Enformer` (DeepMind 的远距离序列预测模型)
    *   `Borzoi` (Calico 的 RNA 测序读段覆盖度预测模型)
    *   `ExplaiNN` (可解释性 CNN 模型)

## 2. 整合外部（其他）模型的原理
gReLU 提供了一个极其灵活的框架来整合任何第三方的 PyTorch 模型。整合的核心原理是 **“接口对齐”** 和 **“LightningModel 封装”**：

### 第一步：输入和输出对齐
gReLU 引擎的标准数据流期望输入张量的维度是 `(Batch_Size, 4, Sequence_Length)`（4 代表 A, C, G, T 四种碱基的 One-hot 编码）。
如果你的外部 PyTorch 模型（比如从 Kipoi 下载的模型或自己用 PyTorch 写的模型）输入维度不同（例如它是 `Batch x Length x 4` 或增加了一个维度 `Batch x 4 x Length x 1`），你只需要写一个简单的 Wrapper（包装类）在 `forward` 函数里对张量进行 `reshape` 或 `permute` 操作：

```python
import torch.nn as nn

class MyModelWrapper(nn.Module):
    def __init__(self, external_model):
        super().__init__()
        self.model = external_model
        
    def forward(self, x):
        # 假设 gReLU 传入的是 (N, 4, L)
        # 但外部模型需要 (N, L, 4)
        x = x.permute(0, 2, 1) 
        return self.model(x)
```

### 第二步：架构拆分 (可选，但推荐)
gReLU 倾向于将模型分为两段：
*   **Embedding (Trunk)**：负责从序列提取特征（如 CNN 层）。
*   **Head**：负责将特征映射到最终的任务预测（如最后的全连接层）。

如果你的模型可以拆分，这有助于 gReLU 在特定的层提取特征进行可视化（如 Motif 发现）。如果不可拆分，你可以将整个外部模型视为 `Embedding`，并将 `Head` 设为 `nn.Identity()`。

### 第三步：使用 LightningModel 包装
将上述调整好的 PyTorch `nn.Module` 放入 `grelu.lightning.LightningModel` 中。这是最关键的一步，它将原生的 PyTorch 模型转换为一个包含了 gReLU 所有高级功能（计算 Loss、自动优化、支持多 GPU 训练、直接挂载解释和突变分析函数等）的超级对象：

```python
from grelu.lightning import LightningModel
from grelu.model.models import BaseModel
import torch.nn as nn

# 假设 external_model 是外部模型实例
wrapped_model = BaseModel(embedding=MyModelWrapper(external_model), head=nn.Identity())

# 整合进 gReLU 生态
grelu_model = LightningModel(
    model_params={
        'model_type': 'BaseModel', 
        'embedding': wrapped_model.embedding, 
        'head': wrapped_model.head
    }
)
```

完成这一步后，这个模型就可以像内置模型一样，调用 gReLU 强大的下游分析流程了（详情参考 `docs/tutorials/8_custom_models.ipynb`）。