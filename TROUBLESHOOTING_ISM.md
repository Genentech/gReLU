# gReLU 巨型模型（Borzoi/AlphaGenome）ISM 运行故障排查与指南

## 1. 核心问题描述
在 4x NVIDIA RTX 6000 Ada (48GB) 环境下，尝试在同一个 Python 进程中先后运行 **单卡推理 (Inference)** 和 **多卡并行 ISM (In Silico Mutagenesis)** 时，频繁触发 `torch.OutOfMemoryError`。

### 现象：
* 单卡推理 `SRSF11` 基因（Brain vs Liver）可以成功运行并产生对比图。
* 进入 `run_ism` 调用分布式并行（DDP）时，子进程尝试在 GPU 0 上加载模型副本，但 GPU 0 仍被主进程残留的 CUDA Context 占据，导致 512MB-1GB 的微小分配请求失败，进而导致整个分布式环境崩溃。

## 2. 深度原因分析

### 2.1 显存残留与碎片化
* **Borzoi 的体量**：Borzoi 每次前向传播涉及 524,288 bp 的序列和庞大的 Transformer 层。即使推理结束，Python 的垃圾回收（GC）无法立即强制显存返回给操作系统。
* **CUDA Context 独占性**：主进程只要在 GPU 0 上初始化过任何 Tensor，就会持有一个 Context。在 DDP 模式下，子进程 0 同样需要请求 GPU 0 的资源，两者冲突。

### 2.2 多进程 DDP 陷阱
* `grelu.interpret.score.ISM_predict` 使用了 PyTorch Lightning 的 `DDPStrategy`。
* 当子进程被 `spawn` 或 `fork` 出来时，如果主进程已经“污染”了 GPU 0，那么子进程在初始化模型权重时会因为 GPU 0 剩余容量不足而失败。

## 3. 已验证的生物学逻辑
* **目标基因**：`SRSF11` (剪接因子)。
* **特异性指标**：`Brain RNA-seq / Liver RNA-seq`。
* **初步结果**：
  * Borzoi 预测比值：~1.58
  * AlphaGenome 预测比值：~1.74
  * 结论：两个模型均准确预测了该基因在脑部的更高表达水平。

## 4. 推荐的稳健执行方案

为了确保显存绝对纯净，**必须进行进程级隔离**。

### 方案 A：分步执行（推荐）
不要在同一个脚本中同时使用 `--inference` 和 `--compute_ism`。

1. **执行推理并生成对比图**：
   ```bash
   python tutorial_1_ism_modular.py --inference --gene SRSF11 --devices 0
   ```
2. **在干净的环境中运行 ISM 扫描**：
   ```bash
   python tutorial_1_ism_modular.py --compute_ism both --gene SRSF11 --devices 0,1,2,3 --num_workers 4
   ```

### 方案 B：代码层面的极端清理
如果必须在脚本内连续运行，需在两个任务之间加入以下逻辑：
```python
def _cleanup(self):
    if hasattr(self, 'model'): del self.model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```
*注：即使如此，进程隔离依然是解决 Borzoi OOM 最可靠的工程手段。*

## 5. 后续调优建议
* **AlphaGenome**：由于其序列较短（131kb），通常可以在 GPU 0 上平稳运行。
* **Borzoi**：由于其极长的 context，建议将 `batch_size` 严格限制在 1，并使用 `precision="bf16-mixed"` 来进一步降低显存压力。
