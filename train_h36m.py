import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
from helper import config
from helper import dt_utils as dut
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf


@dataclass
class TrainingConfig:
    """
    训练配置类
    
    使用 dataclass 定义训练超参数，便于管理和修改
    """
    model_name: str = "kfl_QRFf_transformer"  # 模型名称
    batch_size: int = 64  # 批次大小
    num_epochs: int = 100  # 训练轮数
    learning_rate: float = 1e-3  # 学习率
    weight_decay: float = 1e-4  # 权重衰减（L2正则化）
    max_grad_norm: float = 1.0  # 梯度裁剪的最大范数，防止梯度爆炸
    seq_length: int = 50  # 序列长度
    reset_state: int = 5  # 状态重置周期
    predict_next_frame: bool = True  # 是否预测下一帧
    device: str = "cuda"  # 设备类型
    checkpoint_dir: str = "model"  # 模型检查点保存目录
    log_dir: str = "runs"  # TensorBoard 日志目录
    save_every: int = 5  # 每隔多少轮保存一次检查点
    val_every: int = 1  # 每隔多少轮验证一次
    patience: int = 10  # 早停的耐心值
    param_smoothness_coef: float = 0.01  # 参数平滑正则化系数
    
    def __post_init__(self):
        """
        初始化后自动检测并设置设备
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class PoseSequenceDataset(Dataset):
    """
    姿态序列数据集类
    
    继承自 PyTorch 的 Dataset 类，用于加载姿态序列数据
    """
    def __init__(self, X, Y, seq_ids, masks):
        """
        初始化数据集
        
        Args:
            X: 输入特征序列 (N, seq_length, n_features)
            Y: 目标姿态序列 (N, seq_length, n_joints * 3)
            seq_ids: 序列ID (N,)，用于区分不同的动作序列
            masks: 掩码 (N, seq_length)，标记有效帧
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.seq_ids = torch.from_numpy(seq_ids).long()
        self.masks = torch.from_numpy(masks).float()
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            dict: 包含输入、目标、序列ID和掩码的字典
        """
        return {
            'x': self.X[idx],
            'y': self.Y[idx],
            'seq_id': self.seq_ids[idx],
            'mask': self.masks[idx]
        }


class MPJPELoss(nn.Module):
    """
    MPJPE (Mean Per Joint Position Error) 损失函数
    
    计算预测姿态和真实姿态之间的平均关节位置误差
    可选地添加参数平滑正则化项
    """
    def __init__(self, n_joints: int = 17, smooth_weight: float = 0.1, 
                 param_smoothness_coef: float = 0.0, model: nn.Module = None):
        """
        初始化 MPJPE 损失函数
        
        Args:
            n_joints: 关节数量，默认为17（Human3.6M数据集）
            smooth_weight: 平滑损失的权重系数，默认为0.1
            param_smoothness_coef: 参数平滑正则化系数，默认为0.0（不使用）
            model: 要监控的模型，用于参数平滑
        """
        super().__init__()
        self.n_joints = n_joints
        self.smooth_weight = smooth_weight
        self.param_smoothness_coef = param_smoothness_coef
        self.model = model
        
        if self.param_smoothness_coef > 0 and self.model is not None:
            self.prev_params = {
                name: param.data.clone().detach()
                for name, param in self.model.named_parameters()
            }
        else:
            self.prev_params = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 MPJPE 损失
        
        Args:
            pred: 预测姿态 (batch_size, n_joints * 3)
            target: 真实姿态 (batch_size, n_joints * 3)
        
        Returns:
            总损失 = MPJPE + smooth_weight * 平滑损失 + param_smoothness_coef * 参数平滑损失
        """
        input_dtype = pred.dtype
        eps = torch.tensor(1e-8, device=pred.device, dtype=input_dtype)
        
        pred_reshaped = pred.view(-1, self.n_joints, 3)
        target_reshaped = target.view(-1, self.n_joints, 3)
        
        joint_errors = torch.sqrt(
            torch.sum((pred_reshaped - target_reshaped) ** 2, dim=-1) + 1e-6
        )
        mpjpe = joint_errors.mean().to(input_dtype)
        
        total_loss = mpjpe 
        
        if self.param_smoothness_coef > 0 and self.prev_params is not None:
            param_diff_norm = torch.zeros(1, device=pred.device, dtype=input_dtype)
            eps_param = torch.tensor(1e-8, device=pred.device, dtype=input_dtype)
            
            for name, param in self.model.named_parameters():
                if name in self.prev_params:
                    diff = param - self.prev_params[name].to(input_dtype)
                    param_diff_norm = param_diff_norm + torch.sum(diff.float() ** 2)
            
            param_diff_norm = torch.sqrt(param_diff_norm + eps_param).to(input_dtype)
            param_smoothness_loss = self.param_smoothness_coef * param_diff_norm
            total_loss = total_loss + param_smoothness_loss
        
        return total_loss
    
    def update_prev_params(self):
        """
        更新保存的参数，在每次训练步骤后调用
        """
        if self.param_smoothness_coef > 0 and self.model is not None:
            self.prev_params = {
                name: param.data.clone().detach()
                for name, param in self.model.named_parameters()
            }


class ParameterSmoothnessLoss(nn.Module):
    """
    参数平滑正则化损失
    
    通过惩罚参数的剧烈变化，使训练过程更加稳定
    """
    def __init__(self, model: nn.Module, smoothness_coef: float = 0.01):
        """
        初始化参数平滑损失
        
        Args:
            model: 要监控的模型
            smoothness_coef: 平滑系数
        """
        super().__init__()
        self.model = model
        self.smoothness_coef = smoothness_coef
        self.prev_params = None
        
        self._save_current_params()
    
    def _save_current_params(self):
        """保存当前模型参数的副本"""
        self.prev_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
    
    def forward(self) -> torch.Tensor:
        """
        计算参数平滑损失
        
        Returns:
            参数变化量的加权范数
        """
        first_param = next(self.model.parameters())
        device = first_param.device
        dtype = first_param.dtype
        
        if self.prev_params is None:
            return torch.zeros(1, device=device, dtype=dtype)
        
        param_diff_norm = torch.zeros(1, device=device, dtype=dtype)
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        
        for name, param in self.model.named_parameters():
            if name in self.prev_params:
                diff = param - self.prev_params[name].to(dtype)
                param_diff_norm = param_diff_norm + torch.sum(diff.float() ** 2)
        
        param_diff_norm = torch.sqrt(param_diff_norm + eps).to(dtype)
        
        smoothness_coef_tensor = torch.tensor(self.smoothness_coef, device=device, dtype=dtype)
        return smoothness_coef_tensor * param_diff_norm
    
    def update(self):
        """更新保存的参数，在每次训练步骤后调用"""
        self._save_current_params()


class Trainer:
    """
    训练器类
    
    封装模型训练、验证、保存等完整流程
    """
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        self.criterion = MPJPELoss(
            n_joints=17,
            smooth_weight=0.1,
            param_smoothness_coef=config.param_smoothness_coef,
            model=self.model
        )
        
        log_dir = os.path.join(
            config.log_dir,
            config.model_name,
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.checkpoint_dir = os.path.join(config.checkpoint_dir, config.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.best_mpjpe = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 训练数据加载器
        
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        total_loss = 0.0
        total_pred_norm = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in pbar:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            batch_size = x.shape[0]
            input_dtype = x.dtype
            
            I = torch.eye(self.model.NOUT, device=self.device, dtype=input_dtype).unsqueeze(0).expand(batch_size, -1, -1)
            
            x_state = torch.zeros(batch_size, self.model.NOUT, device=self.device, dtype=input_dtype)
            P_state = I.clone()
            
            result = self.model(
                _z=x,
                target_data=y,
                repeat_data=mask,
                _x_inp=x_state,
                _P_inp=P_state,
                _I=I,
                state_dict={},
                is_training=True
            )
            
            if len(result) == 4:
                _, new_states, pred, gt = result
                loss = self.criterion(pred, gt)
                
                if 'P_pred_norm' in new_states:
                    pred_norm = new_states['P_pred_norm']
                    total_pred_norm += pred_norm
            else:
                loss, new_states = result
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.criterion.update_prev_params()
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            avg_pred_norm = total_pred_norm / num_batches if num_batches > 0 else 0.0
            pbar.set_postfix(loss=f"{loss.item() if not torch.isnan(loss) else 0.0:.4f}, P_norm={avg_pred_norm:.2f}")
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'pred_norm': total_pred_norm / num_batches if num_batches > 0 else 0.0
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        Args:
            dataloader: 验证数据加载器
        
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        total_mpjpe = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        
        for batch in pbar:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            batch_size = x.shape[0]
            input_dtype = x.dtype
            
            I = torch.eye(self.model.NOUT, device=self.device, dtype=input_dtype).unsqueeze(0).expand(batch_size, -1, -1)
            
            x_state = torch.zeros(batch_size, self.model.NOUT, device=self.device, dtype=input_dtype)
            P_state = I.clone()
            
            result = self.model(
                _z=x,
                target_data=y,
                repeat_data=mask,
                _x_inp=x_state,
                _P_inp=P_state,
                _I=I,
                state_dict={},
                is_training=False
            )
            
            if len(result) == 4:
                model_loss, _, pred, gt = result
                loss = model_loss
                if pred.shape[0] > 0:
                    pred_reshaped = pred.float().view(-1, 17, 3)
                    gt_reshaped = gt.float().view(-1, 17, 3)
                    eps = torch.tensor(1e-8, device=pred.device, dtype=torch.float32)
                    diff = pred_reshaped - gt_reshaped
                    joint_errors = torch.sqrt(
                        torch.sum(diff ** 2, dim=-1, keepdim=True) + eps
                    )
                    mpjpe = joint_errors.mean().to(pred.dtype)
                    total_mpjpe += mpjpe.item()
                    
                    correct = (joint_errors < 70.0).float().sum().item()
                    total_correct += correct
                    total_samples += joint_errors.numel()
            else:
                loss, _ = result
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item() if not torch.isnan(loss) else 0.0:.4f}',
                'mpjpe': f'{total_mpjpe / num_batches if num_batches > 0 and not torch.isnan(torch.tensor(total_mpjpe / num_batches)) else 0.0:.2f}',
                'acc': f'{total_correct / total_samples * 100 if total_samples > 0 else 0.0:.1f}%'
            })
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'mpjpe': total_mpjpe / num_batches if num_batches > 0 else 0.0,
            'acc': total_correct / total_samples * 100 if total_samples > 0 else 0.0
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前轮数
            metrics: 当前指标字典
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'prev_params': self.criterion.prev_params
        }
        
        loss_str = f"{metrics['loss']:.5f}"
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.config.model_name}_epoch{epoch}_loss{loss_str}.ckpt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        resume_path = os.path.join(self.checkpoint_dir, 'resume_checkpoint.ckpt')
        torch.save(checkpoint, resume_path)
        
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, 
                f"best_{self.config.model_name}_epoch{epoch}_loss{loss_str}.ckpt"
            )
            torch.save(checkpoint, best_path)
            
            import json
            best_info = {
                'epoch': epoch,
                'loss': metrics['loss'],
                'mpjpe': metrics['mpjpe'],
                'model_name': self.config.model_name
            }
            best_info_path = os.path.join(self.checkpoint_dir, 'best_info.json')
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=2)
            
            print(f"Saved best model with loss: {metrics['loss']:.4f}, MPJPE: {metrics['mpjpe']:.2f} mm")
    
    def load_checkpoint(self, path: str):
        """
        加载模型检查点
        
        Args:
            path: 检查点文件路径
        
        Returns:
            (epoch, metrics): 加载的轮数和指标
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'prev_params' in checkpoint:
            self.param_smoothness_loss.prev_params = checkpoint['prev_params']
        return checkpoint['epoch'], checkpoint['metrics']
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print(f"Training {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Param smoothness coef: {self.config.param_smoothness_coef}")
        print(f"Validate every {self.config.val_every} epoch(s)")
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Training epochs", unit="epoch"):
            
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['loss']
            train_pred_norm = train_metrics.get('pred_norm', 0.0)
            
            if (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics['loss']
                val_mpjpe = val_metrics['mpjpe']
                val_acc = val_metrics['acc']
                
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('MPJPE/val', val_mpjpe, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                self.writer.add_scalar('P_Norm/train', train_pred_norm, epoch)
                
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - "
                      f"Train Loss: {train_loss:.4f} - P_Norm: {train_pred_norm:.2f} - "
                      f"Val Loss: {val_loss:.4f} - Val MPJPE: {val_mpjpe:.2f} mm - Val Acc: {val_acc:.2f}% - LR: {current_lr:.6f}")
                
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    self.best_mpjpe = val_mpjpe
                    self.patience_counter = 0
                    print(f"  ★ New best model! Val Loss: {val_loss:.4f}, Val MPJPE: {val_mpjpe:.2f} mm")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f} - LR: {current_lr:.6f}")
            
            if (epoch + 1) % self.config.save_every == 0:
                save_path = os.path.join(self.checkpoint_dir, 'model_final.pth')
                torch.save(self.model.state_dict(), save_path)
                optimizer_path = os.path.join(self.checkpoint_dir, 'optimizer_final.pth')
                torch.save(self.optimizer.state_dict(), optimizer_path)
                print(f"Model saved to {save_path}")
        
        self.writer.close()
        print("Training completed!")
        print(f"Best Val Loss: {self.best_loss:.4f}")
        print(f"Best Val MPJPE: {self.best_mpjpe:.2f} mm")


def main():
    """
    主函数
    
    配置训练参数、加载数据、创建模型和训练器，开始训练
    """
    config_params = config.get_params()
    
    train_config = TrainingConfig(
        model_name="kfl_QRFf_transformer",
        batch_size=256,
        num_epochs=5,
        learning_rate=1e-6,
        predict_next_frame=True,
        param_smoothness_coef=0.0,
        patience=10,
        val_every=2,
        save_every=1,
    )
    
    print("Preparing dataset...")
    params = config.update_params(config_params)
    params['predict_next_frame'] = train_config.predict_next_frame
    params['batch_size'] = train_config.batch_size
    params['seq_length'] = train_config.seq_length
    params['device'] = train_config.device
    
    (params, X_train, Y_train, F_train, G_train, S_train, R_L_train,
     X_test, Y_test, F_test, G_test, S_test, R_L_test) = dut.prepare_training_set(params)
    
    train_dataset = PoseSequenceDataset(X_train, Y_train, S_train, R_L_train)
    val_dataset = PoseSequenceDataset(X_test, Y_test, S_test, R_L_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    if train_config.model_name == "kfl_QRFf_transformer":
        model = kfl_QRFf_transformer(params=params)
    elif train_config.model_name == "kfl_QRFf":
        model = kfl_QRFf(params=params)
    else:
        raise ValueError(f"Unknown model: {train_config.model_name}")
    
    trainer = Trainer(model, train_config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
