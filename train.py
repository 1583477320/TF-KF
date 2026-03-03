import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from helper import utils as ut
from helper import dt_utils as dt 
from model_runner import model_provider
from helper import config
from nets.inception_resnet_v2 import InceptionResNetV2
from pathlib import Path
from helper import train_helper as th
from model_runner.cnn.inception_train import run_one_epoch

def run_training_process(params, args):
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 数据加载
    train_loader = th.get_dataloader(params, is_training=True)

    # 3. 模型定义 (在循环外定义一次)
    model = InceptionResNetV2(num_classes=params['n_output'])
    model.to(device)
    
    # 4. 修复模型加载逻辑
    cp_dir = params.get('cp_file')
    # 假设 cp_file 是目录，检查目录是否存在
    load_path = f"{cp_dir}/model_final.pth"
    if Path(load_path).is_file():
        model.load_state_dict(torch.load(load_path))
        print(f"已加载预训练模型权重: {load_path}")
    else:
        print(f"权重文件不存在，使用随机初始化: {load_path}")


    # 5. 优化器定义 (在循环外定义一次，保持状态)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params.get('weight_decay', 0.00004)
    )
    
    # 如果是恢复训练，且保存了优化器状态，这里也应该加载 optimizer 的 state_dict
    # if load_path: 
    #     optimizer.load_state_dict(torch.load(f"{cp_dir}/optimizer_final.pth"))

    # 6. 损失函数与混合精度
    criterion = nn.MSELoss(reduction='sum') # 建议改为 'mean'
    scaler = torch.amp.GradScaler(device=device.type == 'cuda',enabled=True)

    # 7. 训练主循环 (合并之前的双层循环)
    # 假设 params['num_epochs'] 是总训练轮数
    num_epochs = params.get('num_epochs', 100) 
    start_epoch = args.epoch_counter_start

    print(f"Starting training from epoch {start_epoch} to {num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        # --- 训练阶段 ---
        # 每一个 epoch 调用一次
        avg_loss = run_one_epoch(model, optimizer, scaler, criterion, train_loader, device, epoch, params)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # --- 学习率衰减 ---
        # 逻辑：lr = base_lr / (5^epoch_counter)
        # 注意：这里 epoch_counter 应该是基于整个训练过程的计数
        # 如果你只是想简单的衰减，可以这样：
        current_lr = params['lr'] / (5 ** (epoch + 1)) 
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # --- 保存模型 ---
        if (epoch + 1) % 5 == 0: # 每5个epoch保存一次，避免频繁IO
            save_path = f"{params['cp_file']}/model_final.pth"
            torch.save(model.state_dict(), save_path)
            # 同时也保存优化器状态以便完全恢复
            torch.save(optimizer.state_dict(), f"{params['cp_file']}/optimizer_final.pth")
            print(f"Model saved to {save_path}")

    # 最终保存
    save_path = f"{params['cp_file']}/model_final.pth"
    torch.save(model.state_dict(), save_path)
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


    if args.mode == 3:
        assert params['model_file'] != ""
        params['model_file'] = args.model_file
        params['run_mode'] = 3
        
        ut.start_log(params)
        ut.log_write("正在测试指定的模型...", params)
        
        params["batch_size"] = 100
        # 更新估计结果的保存路径
        model_name = args.model_file.split('/')[-1]
        params["est_file"] = os.path.join(params["est_file"], model_name) + '/'
        
        # 执行评估
        test_loss = evaller.eval(params)
        
        s = '验证结果 --> 模型 %s | 误差 %f' % (model_name, test_loss)
        ut.log_write(s, params)

    # ----------------------------------------------------------------------------
    # 模式 1: 训练并同步进行验证循环
    # ----------------------------------------------------------------------------
    elif args.mode == 1:
        run_training_process(params, args)