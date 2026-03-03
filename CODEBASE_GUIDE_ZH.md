# TF-KF 代码库完整导读（中文）

## 1. 项目目标与核心思路

TF-KF 是一个 **3D 人体姿态估计** 项目：
- 输入：图像（先经 CNN 提取特征）
- 输出：17 个关节的 3D 坐标（共 51 维）
- 思路：将深度时序模型（LSTM/Transformer）与卡尔曼滤波更新融合，得到更加稳定的序列预测。

项目同时提供：
- 纯 CNN（InceptionResNetV2）训练入口
- 多个 KF-RNN 变体（kfl_K / kfl_QRf / kfl_QRFf）
- Transformer + KF 变体（kfl_QRFf_transformer）
- 按 MPJPE 的评估脚本和可视化对比脚本

---

## 2. 顶层目录速览

- `helper/`：参数配置、数据组织、batch/state 管理、checkpoint 管理
- `model_runner/`：模型主体实现（CNN 与各类 KF 时序模型）
- `nets/`：InceptionResNetV2 网络定义
- `train.py`：CNN 主干训练入口
- `train_h36m.py`：当前主时序训练入口（LSTM/KF/Transformer-KF）
- `train_transformer.py`：Transformer 专项训练脚本（保留版）
- `evaluate_mpjpe.py`：离线评估（总 MPJPE + 分动作统计）
- `compare_models.py`：多模型样本级可视化对比工具

---

## 3. 关键配置（helper/config.py）

`get_params()` 集中管理训练、模型、数据、路径、序列长度等参数。

最关键字段：
- 模型相关：`model`、`n_output=51`、`n_hidden`、`Q/R/K` 各自网络深度与宽度
- Transformer 相关：`d_model`、`nhead`、`num_layers`、`Q_d_model`、`R_d_model`
- 训练相关：`lr`、`batch_size`、`seq_length`、`reset_state`
- 数据相关：`h36m_root`、`cnn_model`、`cache_dir`
- 模式切换：`predict_next_frame`（预测当前帧/下一帧）

`update_params()` 负责刷新日志文件路径。

---

## 4. 数据管线（helper/dt_utils.py）

### 4.1 Human3.6M 扫描与组织
`prepare_training_set()` 是时序训练数据准备总入口：
1. 扫描 H36M 的 train/valid split
2. 用 CNN 提取或读取缓存的观测特征
3. 根据 `normalise_data` 做归一化
4. 按模式调用：
   - `prepare_sequences()`：当前帧监督
   - `prepare_sequences_next_frame()`：下一帧监督

### 4.2 观测特征缓存
`build_or_load_observations()` 会：
- 加载 InceptionResNetV2
- 对每帧图像提取 51 维观测向量
- 与 GT 对齐后缓存，减少重复抽特征成本

### 4.3 序列索引
`get_seq_indexes()` 当前实现简单返回 `np.arange(len(S_L))`，训练阶段按该索引切 batch。

---

## 5. 状态管理与 Batch 组装（helper/train_helper.py + helper/utils.py）

### 5.1 状态初始化
`utils.get_state_list()` / `get_zero_state()` 负责创建 LSTM/KF 所需状态：
- `F_pre/F_t`：状态转移分支
- `Q_pre/Q_t`、`R_pre/R_t`、`K_pre/K_t`：噪声或增益分支
- `PCov_pre`、`_x_pre` 在 batch 组装时初始化/继承

### 5.2 batch 构建核心
`prepare_kfl_QRFf_batch()` 是核心函数，负责：
- 计算当前 batch 的样本 id 与上一个 batch 对应 id
- 根据是否跨序列、是否达到 reset 周期，决定状态“重置 or 继承”
- 输出模型 forward 所需张量：`x/y/repeat_data/_x_inp/_P_inp` 等

这部分是把“卡尔曼状态机”与 mini-batch 训练桥接起来的关键。

---

## 6. 模型层设计

### 6.1 LSTM + KF（model_runner/klstm/kfl_QRFf.py）
单步时间循环中做四件事：
1. `F` 分支（LSTM）预测 `pred_x`
2. `Q` 分支预测过程噪声与协方差传递项
3. `R` 分支预测观测噪声
4. 卡尔曼更新（预测协方差、卡尔曼增益、状态更新）

并用 `repeat_data` mask 处理 padding/无效帧。

### 6.2 Transformer + KF（model_runner/klstm/kfl_QRFf_transformer.py）
将 LSTM 分支替换成 Transformer 编码器：
- `transformer_F`：预测状态
- `transformer_Q`：预测 Q 和 F 的对角项
- `transformer_R`：预测 R 对角项

为稳定性，多处采用 `softplus + clamp + 对角化`。

### 6.3 其他变体
- `kfl_QRf.py`：Q/R 分支版本
- `kfl_K.py`：直接学习 K 的版本
- `model_runner/lstm/pt_lstm.py`：不含 KF 的基线 LSTM

---

## 7. 训练主流程

### 7.1 时序主训练（train_h36m.py）
完整流程：
1. 读配置并选模型类型
2. 调 `prepare_training_set()` 准备 X/Y/序列 id/mask
3. 训练循环中：
   - `prepare_kfl_QRFf_batch()` 准备 batch + 状态
   - model forward 得到 loss + 新状态
   - mixed precision（`autocast` + `GradScaler`）反向更新
4. 每个 epoch 后在 train/test 上调用 `test_data()`
5. `helper/checkpoint.py` 统一保存 best + regular checkpoint，并写 resume 点

### 7.2 CNN 训练（train.py）
独立训练 InceptionResNetV2，用于产生观测特征模型权重（`model_final.pth`）。

---

## 8. 评估与可视化

### 8.1 MPJPE 评估（evaluate_mpjpe.py）
- 自动发现并识别 checkpoint 对应模型类型
- 可加载新旧两种 checkpoint 格式
- `compute_mpjpe()` 统一将预测/真值 reshape 为 `(N,17,3)` 后计算 mm 误差
- `evaluate_model_by_action()` 支持按动作类别统计结果

### 8.2 多模型对比（compare_models.py）
- 扫描多个模型权重
- 对同一样本画出 GT 与各模型预测骨架
- 支持 3D + 2D 可视化输出

---

## 9. Checkpoint 机制（helper/checkpoint.py）

`ModelCheckpoint` 封装了：
- 普通 checkpoint 保存
- 最优 checkpoint（`best_*.ckpt`）维护
- 保留最近 N 个普通 checkpoint
- 保存 `best_info.json`
- 支持 load best / load latest

训练脚本还额外写 `resume_checkpoint.ckpt` 方便断点续训。

---

## 10. 一条从数据到误差的完整链路

1. `train.py` 训练 Inception（或直接使用现成 `cnn_model`）
2. `train_h36m.py` 调 `dt_utils.prepare_training_set()` 生成时序训练样本
3. `train_helper.prepare_kfl_QRFf_batch()` 构造 batch 与 KF/LSTM 状态
4. `kfl_*.py` / `kfl_*_transformer.py` 做前向 + 卡尔曼更新
5. `checkpoint.py` 保存模型
6. `evaluate_mpjpe.py` 计算整体与分动作 MPJPE
7. `compare_models.py` 对误差做可视化解释

---

## 11. 当前代码库的工程特点（读代码建议）

- 混合了“旧版迁移痕迹（TF 风格命名）”与“新 PyTorch 实现”
- 中英文注释并存，强调训练稳定性修复（NaN/Inf、mask、状态重置）
- 训练主线以 `train_h36m.py + helper/*.py + model_runner/klstm/*.py` 为核心

建议阅读顺序：
1. `README.md`
2. `train_h36m.py`
3. `helper/dt_utils.py` / `helper/train_helper.py`
4. `model_runner/klstm/kfl_QRFf.py` 和 `kfl_QRFf_transformer.py`
5. `evaluate_mpjpe.py` 与 `compare_models.py`

