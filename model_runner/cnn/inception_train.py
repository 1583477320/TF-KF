import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm  # 用于进度条显示
# 假设引入了自定义的工具库
from helper import dt_utils as dut
from helper import train_helper as th

def run_one_epoch(model, optimizer, scaler, criterion, train_loader, device, epoch, params):
    """
    执行单个 Epoch 的训练
    """
    model.train()
    
    # 修复：len(train_loader) 已经是 batch 的数量，不需要除以 batch_size
    steps_per_epoch = len(train_loader)
    
    # 使用 tqdm 创建进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    total_loss = 0
    total_rmse = 0
    processed_batches = 0

    try:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- 计算指标 ---
            with torch.no_grad():
                diff = logits - labels
                sample_sq_err = torch.sum(torch.square(diff), dim=1)
                err_tr = torch.sqrt(sample_sq_err).mean()

            # 累加指标
            total_loss += loss.item()
            total_rmse += err_tr.item()
            processed_batches += 1

            # 更新进度条显示
            if processed_batches % 50 == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'RMSE': f"{err_tr.item():.4f}"
                })

    except KeyboardInterrupt:
        print("\nTraining Interrupted.")
    
    # 返回该 Epoch 的平均 Loss
    return total_loss / processed_batches if processed_batches > 0 else 0
