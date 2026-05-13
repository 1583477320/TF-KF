import torch
import json
import os
from model_runner.klstm.pure_kalman import PureKalmanFilter

def load_best_params():
    """加载优化结果，找出最佳参数"""
    with open('kalman_optimization_results.json', 'r') as f:
        results = json.load(f)

    if not results:
        print("没有找到优化结果")
        return None

    # 按MPJPE排序
    results.sort(key=lambda x: x['mpjpe'])

    best = results[0]
    print("最佳参数组合:")
    print(f"  Q_scale: {best['Q_scale']}")
    print(f"  R_scale: {best['R_scale']}")
    print(f"  velocity_decay: {best.get('velocity_decay', 1.0)}")
    print(f"  MPJPE: {best['mpjpe']:.2f} mm")

    return best

def generate_kalman_params(Q_scale, R_scale, velocity_decay=1.0):
    """生成卡尔曼滤波器参数"""
    NOUT = 51  # 状态维度为51（17个关节×3个坐标）

    # 创建状态转移矩阵 F (单位矩阵)
    F = torch.eye(NOUT)

    # 创建过程噪声协方差 Q
    Q = torch.eye(NOUT) * Q_scale

    # 创建测量噪声协方差 R
    R = torch.eye(NOUT) * R_scale

    return F, Q, R

def save_best_model():
    """使用最佳参数保存模型"""
    best = load_best_params()
    if best is None:
        return

    # 生成参数
    F, Q, R = generate_kalman_params(
        best['Q_scale'],
        best['R_scale'],
        best.get('velocity_decay', 1.0)
    )

    # 创建模型
    dim = 51
    model = PureKalmanFilter(F, Q, R, dim, H=None)

    # 保存路径
    model_dir = "/home/zhao/pyproject/TF-KF/model/pure_kalman"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'pure_kalman_model.pth')

    # 保存模型状态
    model_state = {
        'F': F.cpu().numpy(),
        'Q': Q.cpu().numpy(),
        'R': R.cpu().numpy(),
        'H': None,
        'dim': dim,
        'params': {
            'Q_scale': best['Q_scale'],
            'R_scale': best['R_scale'],
            'velocity_decay': best.get('velocity_decay', 1.0)
        }
    }

    torch.save(model_state, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 显示其他参数组合（供比较）
    print("\n其他参数组合（按MPJPE排序）:")
    with open('kalman_optimization_results.json', 'r') as f:
        results = json.load(f)
    results.sort(key=lambda x: x['mpjpe'])
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. Q={r['Q_scale']}, R={r['R_scale']}, MPJPE={r['mpjpe']:.2f} mm")

if __name__ == "__main__":
    save_best_model()
