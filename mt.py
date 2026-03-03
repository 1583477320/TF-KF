import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# H36M 骨架连接关系 (17个关节点)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16)
]

def visualize_inference(img_path, gt_3d, pred_3d):
    """
    img_path: 原始图片路径
    gt_3d: 真实 3D 坐标 (17, 3)
    pred_3d: 模型预测 3D 坐标 (17, 3)
    """
    fig = plt.figure(figsize=(15, 7))

    # -------- 左侧：原始图像 --------
    ax_img = fig.add_subplot(121)
    raw_img = Image.open(img_path)
    ax_img.imshow(raw_img)
    ax_img.set_title(f"Input Image\n{img_path.name}")
    ax_img.axis('off')

    # -------- 右侧：3D 骨架对比 --------
    ax_3d = fig.add_subplot(122, projection='3d')
    ax_3d.set_title("3D Pose: GT (Blue) vs Pred (Red)")

    # 坐标转换函数：为了在 matplotlib 中看起来更自然
    # H36M 原始坐标通常 Y 是向上，这里通过变换调整视角
    def transform_coords(p):
        return p[:, 0], p[:, 2], -p[:, 1]

    gt_x, gt_y, gt_z = transform_coords(gt_3d)
    pr_x, pr_y, pr_z = transform_coords(pred_3d)

    # 绘制骨架连线
    for s, e in SKELETON_EDGES:
        # 绘制真实值 (蓝色虚线)
        ax_3d.plot([gt_x[s], gt_x[e]], [gt_y[s], gt_y[e]], [gt_z[s], gt_z[e]], 
                   color='blue', linestyle='--', alpha=0.5)
        # 绘制预测值 (红色实线)
        ax_3d.plot([pr_x[s], pr_x[e]], [pr_y[s], pr_y[e]], [pr_z[s], pr_z[e]], 
                   color='red', lw=2)

    # 绘制关节点
    ax_3d.scatter(gt_x, gt_y, gt_z, color='blue', s=20, label='Ground Truth')
    ax_3d.scatter(pr_x, pr_y, pr_z, color='red', s=20, label='Prediction')

    ax_3d.legend()
    # 调整视角
    ax_3d.view_init(elev=15, azim=-70)
    
    plt.tight_layout()
    plt.savefig("pose1.png")
    plt.show()


@torch.no_grad()
def verify_model_prediction(model, dataset, idx, device="cuda"):
    """
    针对当前模型结构的验证过程
    """
    model.to(device)
    model.eval()

    # 1. 获取数据 (从 dataset 中直接提取)
    # 注意：dataset[idx] 返回的是 (tensor_image, pose3d_flat)
    input_tensor, gt_pose_flat = dataset[idx]
    img_path = dataset.img_paths[idx]
    
    # 2. 模型推理
    # input_tensor 形状为 (C, H, W)，需要增加 batch 维度 -> (1, C, H, W)
    input_batch = input_tensor.unsqueeze(0).to(device)
    output_flat,_ = model(input_batch)

    # 3. 数据还原 (后处理)
    # 假设模型输出是 (51,)，我们需要将其 reshape 回 (17, 3)
    # 同时如果训练时用了 /1000.0，这里预测出来的也是米，保持一致即可
    pred_3d = output_flat.squeeze().cpu().numpy().reshape(17, 3)
    gt_3d = gt_pose_flat.numpy().reshape(17, 3)

    # 4. 可视化
    print(f"正在展示第 {idx} 张样本的对比图...")
    visualize_inference(img_path, gt_3d, pred_3d)

# ============================
# 使用示例
# ============================
if __name__ == "__main__":
    from helper.data_lodaer import H36MDataset,build_inception_test_transform
    from nets.inception_resnet_v2 import InceptionResNetV2 
    import os

    train_transform = build_inception_test_transform()
    DATA_ROOT = "data/h36m"

    H5_TRAIN  = os.path.join(DATA_ROOT, "annot", "valid.h5")
    IMG_ROOT  = os.path.join(DATA_ROOT, "images", "test")
    IMG_TRAIN = os.path.join(DATA_ROOT, "annot", "valid_images.txt")

    dataset = H36MDataset(
        img_txt=IMG_TRAIN,
        h5_path=H5_TRAIN,
        img_root=IMG_ROOT,
        transform=train_transform
    )
    model = InceptionResNetV2(num_classes=51) 
    model.load_state_dict(torch.load("/home/zhao/pyproject/TF-KF/model/model_final.pth"))

    # 随机选一张图片进行测试
    test_idx = 500 
    verify_model_prediction(model, dataset, test_idx)