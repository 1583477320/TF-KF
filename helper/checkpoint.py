import torch
import os
import json
import shutil
from datetime import datetime


class ModelCheckpoint:
    """
    模型检查点管理器
    
    功能：
    - 保存/加载完整 checkpoint（模型权重、优化器状态、epoch、loss 等）
    - 自动管理最佳模型
    - 支持断点续训
    - 保存训练配置
    - 清理旧检查点
    """
    
    def __init__(self, checkpoint_dir, model_name, max_checkpoints=5, save_best_only=False):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            model_name: 模型名称
            max_checkpoints: 最多保留的检查点数量
            save_best_only: 是否只保存最佳模型
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.checkpoint_history = []
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self._load_best_info()
    
    def _load_best_info(self):
        """加载最佳模型信息"""
        best_info_path = os.path.join(self.checkpoint_dir, "best_info.json")
        if os.path.exists(best_info_path):
            with open(best_info_path, 'r') as f:
                info = json.load(f)
                self.best_loss = info.get('best_loss', float('inf'))
                self.best_epoch = info.get('best_epoch', -1)
    
    def _save_best_info(self):
        """保存最佳模型信息"""
        best_info_path = os.path.join(self.checkpoint_dir, "best_info.json")
        info = {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'model_name': self.model_name,
            'updated_at': datetime.now().isoformat()
        }
        with open(best_info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _get_checkpoint_path(self, epoch, loss, is_best=False):
        """生成检查点文件路径"""
        if is_best:
            filename = f"best_{self.model_name}_epoch{epoch}_loss{loss:.5f}.ckpt"
        else:
            filename = f"{self.model_name}_epoch{epoch}_loss{loss:.5f}.ckpt"
        return os.path.join(self.checkpoint_dir, filename)
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最新的几个"""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.ckpt') and not f.startswith('best_'):
                path = os.path.join(self.checkpoint_dir, f)
                checkpoints.append((path, os.path.getmtime(path)))
        
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        for path, _ in checkpoints[self.max_checkpoints:]:
            try:
                os.remove(path)
                print(f"[Checkpoint] Removed old checkpoint: {os.path.basename(path)}")
            except Exception as e:
                print(f"[Checkpoint] Failed to remove {path}: {e}")
    
    def save(self, model, optimizer, epoch, loss, params=None, metrics=None, is_best=False):
        """
        保存检查点
        
        Args:
            model: PyTorch 模型
            optimizer: 优化器
            epoch: 当前 epoch
            loss: 当前 loss
            params: 训练参数
            metrics: 其他指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if params:
            checkpoint['params'] = {k: v for k, v in params.items() 
                                   if isinstance(v, (int, float, str, bool, list, dict))}
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        is_best_model = loss < self.best_loss
        if is_best_model:
            self.best_loss = loss
            self.best_epoch = epoch
        
        if self.save_best_only and not is_best_model:
            return None
        
        saved_paths = []
        
        if is_best_model:
            best_path = self._get_checkpoint_path(epoch, loss, is_best=True)
            
            for f in os.listdir(self.checkpoint_dir):
                if f.startswith('best_') and f.endswith('.ckpt'):
                    old_path = os.path.join(self.checkpoint_dir, f)
                    try:
                        os.remove(old_path)
                    except Exception:
                        pass
            
            torch.save(checkpoint, best_path)
            saved_paths.append(best_path)
            print(f"[Checkpoint] Saved best model: {os.path.basename(best_path)} (loss: {loss:.5f})")
            
            self._save_best_info()
        
        if not self.save_best_only:
            regular_path = self._get_checkpoint_path(epoch, loss, is_best=False)
            torch.save(checkpoint, regular_path)
            saved_paths.append(regular_path)
            print(f"[Checkpoint] Saved checkpoint: {os.path.basename(regular_path)}")
            
            self._cleanup_old_checkpoints()
        
        return saved_paths
    
    def load_best(self, model, optimizer=None, device='cpu'):
        """
        加载最佳模型
        
        Args:
            model: PyTorch 模型
            optimizer: 优化器（可选）
            device: 设备
        
        Returns:
            checkpoint: 检查点信息
        """
        best_files = [f for f in os.listdir(self.checkpoint_dir) 
                     if f.startswith('best_') and f.endswith('.ckpt')]
        
        if not best_files:
            print(f"[Checkpoint] No best model found in {self.checkpoint_dir}")
            return None
        
        best_path = os.path.join(self.checkpoint_dir, best_files[0])
        return self.load(model, best_path, optimizer, device)
    
    def load_latest(self, model, optimizer=None, device='cpu'):
        """
        加载最新的检查点
        
        Args:
            model: PyTorch 模型
            optimizer: 优化器（可选）
            device: 设备
        
        Returns:
            checkpoint: 检查点信息
        """
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.ckpt'):
                path = os.path.join(self.checkpoint_dir, f)
                checkpoints.append((path, os.path.getmtime(path)))
        
        if not checkpoints:
            print(f"[Checkpoint] No checkpoint found in {self.checkpoint_dir}")
            return None
        
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        latest_path = checkpoints[0][0]
        return self.load(model, latest_path, optimizer, device)
    
    def load(self, model, checkpoint_path, optimizer=None, device='cpu'):
        """
        加载指定的检查点
        
        Args:
            model: PyTorch 模型
            checkpoint_path: 检查点路径
            optimizer: 优化器（可选）
            device: 设备
        
        Returns:
            checkpoint: 检查点信息
        """
        if not os.path.exists(checkpoint_path):
            print(f"[Checkpoint] Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"[Checkpoint] Loaded checkpoint from {os.path.basename(checkpoint_path)}")
        print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.5f}")
        
        return checkpoint
    
    def get_best_loss(self):
        """获取最佳 loss"""
        return self.best_loss
    
    def get_best_epoch(self):
        """获取最佳 epoch"""
        return self.best_epoch


def save_model_simple(model, save_path, epoch=None, loss=None, params=None):
    """
    简单的模型保存函数
    
    Args:
        model: PyTorch 模型
        save_path: 保存路径
        epoch: 当前 epoch
        loss: 当前 loss
        params: 训练参数
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if params:
        checkpoint['params'] = {k: v for k, v in params.items() 
                               if isinstance(v, (int, float, str, bool, list, dict))}
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved model to: {save_path}")


def load_model_simple(model, load_path, device='cpu'):
    """
    简单的模型加载函数
    
    Args:
        model: PyTorch 模型
        load_path: 加载路径
        device: 设备
    
    Returns:
        checkpoint: 检查点信息
    """
    if not os.path.exists(load_path):
        print(f"[Checkpoint] Model not found: {load_path}")
        return None
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"[Checkpoint] Loaded model from: {load_path}")
    return checkpoint
