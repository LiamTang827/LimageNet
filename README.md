# LimageNet
## 🤯 厌倦了漫长、低效的教学视频？

你是否还在为期末周一堆水课教学视频而烦恼？  
或者在视频网站里被 AI 配音和幻灯片堆砌的视频折磨得头昏脑胀？

在这个信息爆炸的时代，视频内容越来越多，我们的**时间却越来越宝贵**。尤其是在大学课堂中，那些机械地读着 PPT 的录播视频，常常让人抓狂。

🎉 是时候试试 **LimageNet** 了！

> LimageNet 是一个开源工具，专为提取教学视频中的关键帧设计，能够快速捕捉重要画面、跳过无效内容，帮你节省时间、提高效率，让学习更轻松！

# LimageNet 🧠🎥  
> Extract keyframes from educational videos using deep learning and visual filters.

## 🔍 Features
- 🎞️ Frame extraction from raw videos
- 🧠 CLIP-based semantic filtering
- 🔍 Sharpness + transition detection
- 🧪 Easy-to-run pipeline scripts
- 📦 Extensible modular design

## 🚀 Quick Start
```bash
git clone https://github.com/LiamTang827/LimageNet.git
cd LimageNet
pip install -r requirements.txt
python scripts/run_pipeline.py --input data/raw/sample.mp4 --output data/keyframes/

