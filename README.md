# ICSTFT - 对抗样本攻击与防御项目

## 项目概述

ICSTFT (Intelligent Computing Security Techniques for AI Frameworks) 是一个专注于对抗样本攻击与防御的综合项目集合。本项目涵盖了多种经典和前沿的对抗样本生成算法，包括基于梯度的方法、基于优化的方法、黑盒攻击方法以及针对大语言模型的提示词攻击。项目旨在深入研究对抗样本的生成机制，评估深度学习模型的鲁棒性，并探索有效的防御策略。

## 项目结构

项目包含6个主要子项目：

### 1. FGSM (Fast Gradient Sign Method)
快速梯度符号方法，是一种基于梯度的对抗样本生成算法。该方法通过计算损失函数相对于输入的梯度，然后沿梯度符号方向添加扰动来生成对抗样本。该方法简单高效，是许多后续对抗攻击算法的基础。

**使用方法：**
1. 确保已安装Python 3.12, PyTorch 2.9.1+cu126
2. 运行 `1_FGSM/FGSM-MNIST.ipynb` Jupyter笔记本
3. 笔记本将自动下载MNIST数据集并使用预训练模型(model.pth)进行FGSM攻击
4. 观察对抗样本的生成过程和攻击效果

### 2. C&W (Carlini & Wagner Attack)
Carlini & Wagner攻击，是一种高级对抗攻击算法，使用L2距离作为优化目标。该方法通过优化一个复杂的损失函数来生成对抗样本，通常能产生更小扰动的对抗样本。该算法在多种模型和数据集上表现出强大的攻击能力。

**使用方法：**
1. 确保已安装torchvision, torchsummary等依赖
2. 运行 `2_C&W/C&W-MNIST.ipynb` Jupyter笔记本
3. 笔记本将使用预训练的LeNet模型(cw_mnist_lenet.pth)对MNIST数据集进行C&W攻击
4. 可调整c参数、kappa参数、最大迭代次数等超参数来优化攻击效果

### 3. ZOO (Zeroth Order Optimization)
零阶优化攻击，是一种基于黑盒优化的对抗攻击方法。该方法不需要目标模型的梯度信息，通过零阶优化技术生成对抗样本，适用于无法获取模型内部结构和梯度的情况。

**使用方法：**
- **无目标攻击 (Adam优化器):**
  1. 运行 `3_ZOO/adam_cifar10_untargeted.py` 或 `3_ZOO/adam_mnist_untargeted.py`
  2. 确保预训练模型(`models/cifar10_model.pt`或`models/mnist_model.pt`)存在
  3. 程序将生成无目标对抗样本并保存结果

- **有目标攻击 (Newton优化器):**
  1. 运行 `3_ZOO/newton_cifar10_targeted.py` 或 `3_ZOO/newton_mnist_targeted.py`
  2. 确保预训练模型存在
  3. 程序将生成有目标对抗样本并保存结果

- **模型训练:**
  1. 运行 `3_ZOO/setup_cifar10_model.py` 或 `3_ZOO/setup_mnist_model.py` 训练模型
  2. 需要安装numba, scipy等依赖

### 4. Boundary Attack
边界攻击，是一种基于决策边界的黑盒攻击方法。该方法从初始对抗样本开始，沿着决策边界进行随机游走，逐步接近目标类别。该方法在保持较小扰动的同时，有效生成对抗样本。

**使用方法：**
1. 确保已安装Keras, matplotlib, seaborn等依赖
2. 运行 `4_BoundaryAttack/ba_with_visual.py`
3. 程序将使用Keras的ResNet50模型进行攻击
4. 生成的对抗样本和可视化结果将保存在images目录中
5. 确保`images/original/`目录中有原始图片文件

### 5. Defensive Distillation
蒸馏防御，是一种对抗样本防御技术。通过在训练过程中使用软标签(softmax输出)而不是硬标签(one-hot编码)来训练模型，以提高模型对对抗样本的鲁棒性。

**使用方法：**
1. 运行 `5_DefensiveDistillation/Adversarial_Example_(Attack_and_defense).ipynb` Jupyter笔记本
2. 笔记本将展示对抗攻击和蒸馏防御的对比
3. 比较原始模型和蒸馏模型在对抗样本上的表现

### 6. Prompt Attack
提示词攻击，是针对大语言模型(LLM)的新型攻击方法。通过精心设计提示词来生成对抗性输入，使LLM产生错误的输出。

**使用方法：**
1. 安装依赖：`pip install openai nltk bert_score torch transformers`
2. 配置LLM API接口（如OpenAI API）
3. 运行 `6_PromptAttack/PromptAttack.py` 进行提示词攻击
4. 使用 `6_PromptAttack/Demo.ipynb` 查看使用示例
5. 可调整多个参数，如词修改比例阈值、BERTScore阈值等

## 环境需求

- Python 3.12
- PyTorch 2.9.1+cu126
- CUDA 12.6
- 其他依赖包根据各子项目需求安装

## 运行说明

1. **虚拟环境**: 所有项目共享1_FGSM的虚拟环境
2. **GPU支持**: 某些项目需要GPU支持以获得最佳性能
3. **数据集**: 大部分项目会自动下载所需数据集
4. **预训练模型**: 部分项目包含预训练模型，需要确保模型文件存在

## 项目特点

1. **全面性**: 涵盖了从白盒到黑盒、从传统机器学习到深度学习的多种对抗攻击方法
2. **实用性**: 提供完整的代码实现和使用示例，便于复现和扩展
3. **可视化**: 包含丰富的可视化功能，便于分析攻击效果
4. **前沿性**: 包含最新的对抗样本研究方向，如大语言模型对抗攻击

## 应用场景

- 深度学习模型的鲁棒性评估
- 对抗样本防御技术研究
- 安全AI系统的构建
- 大语言模型安全研究

## 注意事项

1. 部分攻击方法可能需要较长的计算时间
2. 某些方法需要大量GPU显存
3. 在使用API接口时需注意费用问题
4. 对抗样本研究应在合法合规的范围内进行