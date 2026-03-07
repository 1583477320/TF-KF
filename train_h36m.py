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
    model_name: str = "kfl_QRFf_transformer"
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    seq_length: int = 50
    reset_state: int = 5
    predict_next_frame: bool = True
    device: str = "cuda"
    checkpoint_dir: str = "model"
    log_dir: str = "runs"
    save_every: int = 10
    patience: int = 10
    param_smoothness_coef: float = 0.01
    
    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


class PoseSequenceDataset(Dataset):
    def __init__(self, X, Y, seq_ids, masks):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.seq_ids = torch.from_numpy(seq_ids).long()
        self.masks = torch.from_numpy(masks).float()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'y': self.Y[idx],
            'seq_id': self.seq_ids[idx],
            'mask': self.masks[idx]
        }


class MPJPELoss(nn.Module):
    def __init__(self, n_joints: int = 17):
        super().__init__()
        self.n_joints = n_joints
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_reshaped = pred.view(-1, self.n_joints, 3)
        target_reshaped = target.view(-1, self.n_joints, 3)
        
        joint_errors = torch.sqrt(
            torch.sum((pred_reshaped - target_reshaped) ** 2, dim=-1) + 1e-6
        )
        return joint_errors.mean()


class SmoothMPJPELoss(nn.Module):
    def __init__(self, n_joints: int = 17, smooth_weight: float = 0.1):
        super().__init__()
        self.n_joints = n_joints
        self.smooth_weight = smooth_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_dtype = pred.dtype
        eps = torch.tensor(1e-8, device=pred.device, dtype=input_dtype)
        
        pred_reshaped = pred.view(-1, self.n_joints, 3)
        target_reshaped = target.view(-1, self.n_joints, 3)
        
        joint_errors = torch.sqrt(
            torch.sum((pred_reshaped - target_reshaped) ** 2, dim=-1) + 1e-6
        )
        mpjpe = joint_errors.mean().to(input_dtype)
        
        diff_norm = torch.sqrt((pred - target) ** 2 + eps)
        smooth_loss = torch.mean(diff_norm).to(input_dtype)
        
        smooth_weight_tensor = torch.tensor(self.smooth_weight, device=pred.device, dtype=input_dtype)
        return mpjpe + smooth_weight_tensor * smooth_loss


class ParameterSmoothnessLoss(nn.Module):
    def __init__(self, model: nn.Module, smoothness_coef: float = 0.01):
        super().__init__()
        self.model = model
        self.smoothness_coef = smoothness_coef
        self.prev_params = None
        
        self._save_current_params()
    
    def _save_current_params(self):
        self.prev_params = {
            name: param.data.clone().detach()
            for name, param in self.model.named_parameters()
        }
    
    def forward(self) -> torch.Tensor:
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
        self._save_current_params()


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig):
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
        
        self.criterion = SmoothMPJPELoss(n_joints=17)
        self.param_smoothness_loss = ParameterSmoothnessLoss(
            model, 
            smoothness_coef=config.param_smoothness_coef
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
        self.model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_smooth_loss = 0.0
        total_mpjpe = 0.0
        total_correct = 0
        total_samples = 0
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
                _, _, pred, gt = result
                if pred.shape[0] > 0:
                    task_loss = self.criterion(pred, gt)
                    
                    pred_reshaped = pred.float().view(-1, 17, 3)
                    gt_reshaped = gt.float().view(-1, 17, 3)
                    eps = torch.tensor(1e-8, device=pred.device, dtype=torch.float32)
                    diff = pred_reshaped - gt_reshaped
                    joint_errors = torch.sqrt(
                        torch.sum(diff ** 2, dim=-1, keepdim=True) + eps
                    )
                    batch_mpjpe = joint_errors.mean().to(pred.dtype)
                    total_mpjpe += batch_mpjpe.item()
                    
                    correct = (joint_errors < 70.0).float().sum().item()
                    total_correct += correct
                    total_samples += joint_errors.numel()
                else:
                    task_loss = torch.zeros(1, device=self.device, dtype=x.dtype, requires_grad=True)
            else:
                task_loss, _ = result
            
            smooth_loss = self.param_smoothness_loss()
            loss = task_loss + smooth_loss
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if not torch.isnan(loss):
                total_loss += loss.item()
            if not torch.isnan(task_loss):
                total_task_loss += task_loss.item()
            if not torch.isnan(smooth_loss):
                total_smooth_loss += smooth_loss.item()
            num_batches += 1
            self.global_step += 1
            
            acc = total_correct / total_samples * 100 if total_samples > 0 else 0.0
            pbar.set_postfix(
                loss=f"{loss.item() if not torch.isnan(loss) else 0.0:.4f}", 
                mpjpe=f"{batch_mpjpe:.1f}" if total_mpjpe > 0 else "N/A",
                acc=f"{acc:.1f}%"
            )
        
        self.param_smoothness_loss.update()
        
        return {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'task_loss': total_task_loss / num_batches if num_batches > 0 else 0.0,
            'smooth_loss': total_smooth_loss / num_batches if num_batches > 0 else 0.0,
            'mpjpe': total_mpjpe / num_batches if num_batches > 0 else 0.0,
            'acc': total_correct / total_samples * 100 if total_samples > 0 else 0.0
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
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
                loss, _, pred, gt = result
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
                    
                    loss = self.criterion(pred, gt)
                else:
                    loss = torch.tensor(0.0, device=self.device, dtype=x.dtype)
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'prev_params': self.param_smoothness_loss.prev_params
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'prev_params' in checkpoint:
            self.param_smoothness_loss.prev_params = checkpoint['prev_params']
        return checkpoint['epoch'], checkpoint['metrics']
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Param smoothness coef: {self.config.param_smoothness_coef}")
        
        for epoch in tqdm(range(self.config.num_epochs), desc="Training epochs", unit="epoch"):
            
            train_metrics = self.train_epoch(train_loader)
            train_loss = train_metrics['loss']
            train_task_loss = train_metrics['task_loss']
            train_smooth_loss = train_metrics['smooth_loss']
            
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_mpjpe = val_metrics['mpjpe']
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/train_task', train_task_loss, epoch)
            self.writer.add_scalar('Loss/train_smooth', train_smooth_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('MPJPE/val', val_mpjpe, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                  f"Train Loss: {train_loss:.4f} (task: {train_task_loss:.4f}, smooth: {train_smooth_loss:.6f}) - "
                  f"Val Loss: {val_loss:.4f} - Val MPJPE: {val_mpjpe:.2f} mm - LR: {current_lr:.6f}")
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.best_mpjpe = val_mpjpe
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # if self.patience_counter >= self.config.patience:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     break
        
        self.writer.close()
        print("Training completed!")
        print(f"Best Val Loss: {self.best_loss:.4f}")
        print(f"Best Val MPJPE: {self.best_mpjpe:.2f} mm")


def main():
    config_params = config.get_params()
    
    train_config = TrainingConfig(
        model_name="kfl_QRFf_transformer",
        batch_size=256,
        num_epochs=50,
        learning_rate=2e-4,
        predict_next_frame=True,
        param_smoothness_coef=0.0,
        patience=10,
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
