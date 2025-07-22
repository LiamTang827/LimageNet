# LimageNet

LimageNet 是一个用于教学视频分析的关键帧提取与幻灯片切换检测系统。该项目专注于识别视频中具有语义代表性的幻灯片帧，并过滤掉由于鼠标、动画等因素引起的干扰，从而提升视频内容结构化处理的准确性。

## 🌟 项目亮点

- 🎯 **精确幻灯片检测**：利用 ResNet2D 提取静态帧，结合 ORB 筛选去除冗余。
- 🧠 **3D CNN 判别模型**：对候选过渡帧进行时序建模，识别真实的幻灯片切换。
- 🖱️ **抗干扰机制**：针对鼠标移动等噪声进行了特别处理，提升准确率。
- 📊 **支持评估指标**：支持编辑距离、召回率、准确率等多种评估方式。

## 🧩 模块组成

## 🚀 快速开始

### 1. 安装依赖

建议使用 Conda 环境（`limagenet`）：

```bash
conda create -n limagenet python=3.10
conda activate limagenet
pip install -r requirements.txt

