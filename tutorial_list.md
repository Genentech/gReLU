# gReLU 教程概览 (Tutorials Overview)

项目提供的入门与进阶教程位于 `docs/tutorials/` 目录下，均为 Jupyter Notebook 格式。按照推荐的学习顺序，包含以下内容：

1. **`1_inference.ipynb` - 推理 (Inference)**
   介绍如何使用预训练模型进行基础的序列预测和推理。

2. **`2_finetune.ipynb` - 微调 (Fine-tuning)**
   演示如何使用自定义数据对现有模型进行微调。

3. **`3_train.ipynb` - 训练 (Training)**
   讲解如何从头开始训练一个新的模型。

4. **`4_design.ipynb` - 序列设计 (Design)**
   介绍如何使用模型来设计或优化 DNA/RNA 序列。

5. **`5_variant.ipynb` - 变异分析 (Variant)**
   讲解如何评估基因变异（如 SNP）对模型预测结果的影响（变异效应预测）。

6. **`6_model_zoo.ipynb` - 模型库 (Model Zoo)**
   介绍如何浏览、加载和使用 gReLU 提供的预训练模型库。

7. **`7_simulations.ipynb` - 模拟预测 (Simulations)**
   展示如何利用模型进行各种模拟实验（如基序插入、突变扫描等）。

8. **`8_custom_models.ipynb` - 自定义模型 (Custom Models)**
   进阶教程，指导如何将你自己定义的 PyTorch 模型架构集成到 gReLU 的训练和评估流程中。
