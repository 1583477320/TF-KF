import torch
import numpy as np
import os
import subprocess
import json
from model_runner.klstm.pure_kalman import PureKalmanFilter

# 定义参数搜索空间
param_space = {
    'Q_scale': [0.001, 0.01, 0.1, 0.5, 1.0],
    'R_scale': [0.1, 0.5, 1.0, 2.0, 5.0],
    'velocity_decay': [0.9, 0.95, 0.99, 1.0]
}

# 结果存储
results = []

# 基准模型路径
model_dir = "/home/zhao/pyproject/TF-KF/model/pure_kalman"
os.makedirs(model_dir, exist_ok=True)

# 运行评估命令
def run_evaluation():
    cmd = ["python", "compare_models.py", "--sample_idx", "100"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

# 解析评估结果
def parse_mpjpe(output):
    lines = output.split('\n')
    for line in lines:
        # 打印所有行以了解输出格式
        print(f"Output line: {line}")
        if 'Kalman' in line:
            # 尝试不同的分割方式
            parts = line.split()
            print(f"Parts: {parts}")
            for part in parts:
                try:
                    mpjpe = float(part)
                    print(f"Found MPJPE: {mpjpe}")
                    return mpjpe
                except ValueError:
                    pass
    return None

# 生成卡尔曼滤波器参数
def generate_kalman_params(Q_scale, R_scale, velocity_decay):
    NOUT = 51  # 状态维度为51（17个关节×3个坐标）
    
    # 创建状态转移矩阵 F (包含速度项)
    F = torch.zeros(NOUT, NOUT)
    for i in range(NOUT):
        F[i, i] = 1.0  # 位置保持
    
    # 创建过程噪声协方差 Q
    Q = torch.eye(NOUT) * Q_scale
    
    # 创建测量噪声协方差 R
    R = torch.eye(NOUT) * R_scale
    
    return F, Q, R

# 保存模型
def save_kalman_model(F, Q, R, params):
    model_state = {
        'F': F.cpu().numpy(),
        'Q': Q.cpu().numpy(),
        'R': R.cpu().numpy(),
        'H': None,
        'dim': F.shape[0],
        'params': params
    }
    
    model_path = os.path.join(model_dir, 'pure_kalman_model.pth')
    torch.save(model_state, model_path)
    print(f"Saved model to {model_path}")

# 主优化循环
def main():
    print("开始优化卡尔曼滤波器参数...")
    
    # 遍历参数空间
    for Q_scale in param_space['Q_scale']:
        for R_scale in param_space['R_scale']:
            for velocity_decay in param_space['velocity_decay']:
                print(f"\n测试参数: Q_scale={Q_scale}, R_scale={R_scale}, velocity_decay={velocity_decay}")
                
                # 生成参数
                F, Q, R = generate_kalman_params(Q_scale, R_scale, velocity_decay)
                
                # 保存模型
                params = {
                    'Q_scale': Q_scale,
                    'R_scale': R_scale,
                    'velocity_decay': velocity_decay
                }
                save_kalman_model(F, Q, R, params)
                
                # 运行评估
                output = run_evaluation()
                mpjpe = parse_mpjpe(output)
                
                if mpjpe is not None:
                    print(f"MPJPE: {mpjpe:.2f} mm")
                    results.append({
                        'Q_scale': Q_scale,
                        'R_scale': R_scale,
                        'velocity_decay': velocity_decay,
                        'mpjpe': mpjpe
                    })
                else:
                    print("无法解析MPJPE值")
    
    # 保存结果
    with open('kalman_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 排序并显示最佳结果
    if results:
        results.sort(key=lambda x: x['mpjpe'])
        print("\n" + "="*60)
        print("最佳参数组合:")
        print("="*60)
        for i, result in enumerate(results[:5]):
            print(f"第{i+1}名: Q_scale={result['Q_scale']}, R_scale={result['R_scale']}, velocity_decay={result['velocity_decay']}, MPJPE={result['mpjpe']:.2f} mm")

if __name__ == "__main__":
    main()
