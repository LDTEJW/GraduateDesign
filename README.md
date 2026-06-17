# 航班延误预测系统

基于 LSTM + XGBoost 集成模型的航班到达延误预测系统。

## 🌟 项目简介

本项目旨在通过机器学习技术预测航班到达延误时间。系统采用**混合深度学习架构**，结合长短期记忆网络（LSTM）和梯度提升树（XGBoost），实现高精度的航班延误预测。

## ✨ 功能特性

- 🚀 **高精度预测**：LSTM + XGBoost 集成模型，兼顾时序特征和非线性关系
- 📊 **可视化展示**：训练过程可视化、性能对比图表、误差分析
- 🌐 **Web 界面**：简洁易用的交互式预测界面
- 📁 **完整流程**：数据预处理、特征工程、模型训练、评估与部署
- ⚡ **实时推理**：支持批量 CSV 数据预测

## 🛠️ 技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| 框架 | Flask | ^2.0 |
| 深度学习 | PyTorch | ^2.0 |
| 机器学习 | XGBoost | ^2.0 |
| 数据处理 | Pandas / NumPy | - |
| 可视化 | Matplotlib / Seaborn | - |
| 前端 | HTML5 / CSS3 | - |

## 项目结构

```
Air-recongnition/
├── main/
│   ├── train/                    # 训练模块
│   │   ├── main.py               # 主训练脚本
│   │   ├── info.py               # 信息工具
│   │   ├── text.py               # 文本处理工具
│   │   ├── data_set/             # 数据集目录
│   │   │   └── flights_sample_3m.csv  # 示例数据集（300万行）
│   │   ├── picture/              # 可视化图表输出
│   │   └── saved_models/         # 训练好的模型文件
│   └── Web/                      # Web 应用模块
│       ├── app.py                # Flask 应用入口
│       ├── templates/            # HTML 模板
│       ├── static/               # 静态资源（CSS、字体）
│       └── utils/                # 工具函数
├── .gitignore
└── README.md
```

## 安装说明

### 环境要求

- Python 3.9+
- CUDA 11.8+（推荐，用于 GPU 加速）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/Air-recongnition.git
cd Air-recongnition
```

2. **创建虚拟环境**
```bash
conda create -n air_delay python=3.9
conda activate air_delay
```

3. **安装依赖**
```bash
cd main/Web
pip install -r requirement.txt
```

4. **安装额外依赖**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

##  使用方法

### 1. 模型训练

```bash
cd main/train
python main.py
```

训练过程将自动完成：
- 数据加载与预处理
- 特征工程（时间特征、滞后特征、统计特征）
- LSTM 模型训练
- XGBoost 模型训练
- 集成模型构建
- 结果可视化与保存

### 2. 启动 Web 服务

```bash
cd main/Web
python app.py
```

访问 `http://localhost:5000` 即可使用预测界面。

### 3. 数据格式要求

上传的 CSV 文件需包含以下必要列：
- `FL_DATE`: 航班日期
- `AIRLINE_CODE`: 航空公司代码
- `ORIGIN`: 出发机场代码
- `DEST`: 目的机场代码
- `CRS_DEP_TIME`: 计划出发时间

可选列（用于模型评估）：
- `ARR_DELAY`: 实际到达延误时间（分钟）

## 训练过程可视化

训练完成后，系统会自动生成以下图表：

| 图表文件 | 说明 |
|----------|------|
| `training_process.png` | LSTM损失/MAE曲线、XGBoost训练曲线、学习率变化 |
| `model_comparison.png` | 各模型性能对比（MAE、RMSE、R²、MSE） |
| `error_distributions.png` | 误差分布分析 |
| `improvement_analysis.png` | 性能改进分析 |
| `uncertainty_analysis.png` | 不确定性分析 |

## 评估指标

系统使用以下指标评估模型性能：

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **R²**: 决定系数
- **MSE** (Mean Squared Error): 均方误差

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发规范

1. 代码风格遵循 PEP 8
2. 提交信息使用语义化格式
3. 新增功能需附带测试用例

##  许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

⭐ 如果本项目对您有帮助，请给予 Star 支持！
