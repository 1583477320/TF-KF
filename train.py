import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
from helper import utils as ut
from helper import dt_utils as dt 
from model_runner import model_provider
from helper import config
from nets.inception_resnet_v2 import InceptionResNetV2
from helper import train_helper as th
from model_runner.cnn.inception_train import run_one_epoch
# 假设 evaller 是 model_provider 下的一个模块，或者你需要实现它
# from model_runner import evaller 

def run_training_process(params, args):
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 数据加载
    train_loader = th.get_dataloader(params, is_training=True)
    # 建议：也加载验证集
    # val_loader = th.get_dataloader(params, is_training=False) 

    # 3. 模型定义
    model = InceptionResNetV2(num_classes=params['n_output'])
    model.to(device)
    
    # 4. 优化器定义 (在加载模型权重前定义，以便加载其状态)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params.get('weight_decay', 0.00004)
    )

    # 5. 混合精度
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 6. 模型加载逻辑 (支持恢复训练)
    cp_dir = params.get('cp_file')
    start_epoch = args.epoch_counter_start
    
    # 确保保存目录存在
    if cp_dir:
        os.makedirs(cp_dir, exist_ok=True)

    # 尝试加载最新的检查点以恢复训练
    # 逻辑：如果指定了 start_epoch > 0，尝试加载对应的模型和优化器
    # 或者简单点：总是尝试加载 model_final.pth，如果存在则视为继续训练
    load_path = f"{cp_dir}/model_final.pth" if cp_dir else None
    
    if load_path and Path(load_path).is_file():
        print(f"加载检查点: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        
        # 尝试加载优化器状态
        opt_path = f"{cp_dir}/optimizer_final.pth"
        if Path(opt_path).is_file():
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                print("优化器状态已恢复。")
            except Exception as e:
                print(f"警告：优化器状态加载失败 ({e})，将重新初始化。")
    else:
        print("未找到预训练权重，使用随机初始化。")

    # 7. 损失函数 (建议改为 mean)
    criterion = nn.MSELoss(reduction='mean') 

    # 8. 学习率调度器 (修正原本过于激进的衰减)
    # 示例：每 10 个 epoch 衰减为原来的 50%
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, params.get('num_epochs', 3)):
        # --- 训练阶段 ---
        avg_loss = run_one_epoch(model, optimizer, scaler, criterion, train_loader, device, epoch, params)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

        # --- 验证阶段 (建议添加) ---
        # if val_loader:
        #     val_loss = run_validation(model, criterion, val_loader, device)
        #     print(f"Validation Loss: {val_loss:.4f}")

        # --- 更新学习率 ---
        scheduler.step()

        # --- 保存模型 ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == params.get('num_epochs', 100):
            if cp_dir:
                torch.save(model.state_dict(), f"{cp_dir}/model_final.pth")
                torch.save(optimizer.state_dict(), f"{cp_dir}/optimizer_final.pth")
                print(f"Model saved to {cp_dir}")

        torch.save(model.state_dict(), f"{cp_dir}/model_final.pth")
        torch.save(optimizer.state_dict(), f"{cp_dir}/optimizer_final.pth")
        print(f"Model saved to {cp_dir}")
    print("Training finished.")

if __name__ == "__main__":
    # 1. 获取基础参数
    params = config.get_params()

    # 2. 命令行参数解析
    parser = argparse.ArgumentParser(description='处理一些整数。')
    parser.add_argument('--mode', type=int, default=1)  # 1=训练+测试, 2=测试+训练, 3=仅测试指定模型
    parser.add_argument('--run_mode', type=int, default=2)
    parser.add_argument('--model_file', default="/home/coskunh/PycharmProjects/data/36m/cp_tr/model.ckpt-10199")
    parser.add_argument('--epoch_counter_start', type=int, default=0)

    args = parser.parse_args()

    # 3. 配置参数设置
    params['write_est'] = True
    params["ds_training"] = "crop350"
    params["ds_test"] = "crop350"
    params['batch_size'] = 128


    # 修复 mode 3 部分
    if args.mode == 3:
        # 1. 这里缺少 evaller 的定义，需要确认 eval 模块在哪里
        # 假设需要在 model_provider 中实现或导入
        print("Mode 3: Testing specific model...")
        
        # 确保路径存在
        if not os.path.exists(args.model_file):
            print(f"Error: Model file not found at {args.model_file}")
        else:
            params['model_file'] = args.model_file
            params['run_mode'] = 3
            ut.start_log(params)
            
            # 这里需要你实现或导入 evaller
            # 例如: from model_runner import evaluator
            # test_loss = evaluator.eval(params)
            print("WARNING: 'evaller' module is not defined in the script.")
            
    elif args.mode == 1:
        run_training_process(params, args)