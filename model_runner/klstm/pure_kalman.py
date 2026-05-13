import torch

class PureKalmanFilter:
    def __init__(self, F, Q, R, dim, H=None):
        """
        纯卡尔曼滤波器
        
        参数:
        F: 状态转移矩阵 (dim, dim) 或 (batch, dim, dim)
        Q: 过程噪声协方差矩阵 (dim, dim) 或 (batch, dim, dim)
        R: 测量噪声协方差矩阵 (dim, dim) 或 (batch, dim, dim)
        dim: 状态维度
        H: 观测矩阵 (dim, dim) 或 (batch, dim, dim)。如果为None，则默认为单位矩阵
        """
        self.F = F
        self.Q = Q
        self.R = R
        self.dim = dim
        self.H = H
    
    def forward(self, z, mask, x_init, P_init):
        """
        执行卡尔曼滤波
        
        参数:
        z: 测量序列 (batch, seq, dim)
        mask: 测量有效性掩码 (batch, seq)  (1为有效，0为无效)
        x_init: 初始状态
        P_init: 初始协方差矩阵 (batch, dim, dim)
        
        返回:
        xres: 滤波后的状态序列 (batch, seq, dim)
        pres: 协方差矩阵序列 (batch, seq, dim, dim)
        """
        device = z.device
        batch_size = z.shape[0]
        seq_length = z.shape[1]
        
        # 辅助函数：确保矩阵具有 batch 维度
        def expand_mat(mat):
            if mat.dim() == 2:
                return mat.unsqueeze(0).expand(batch_size, -1, -1).to(device)
            return mat.to(device)
            
        F_mat = expand_mat(self.F)
        Q_mat = expand_mat(self.Q)
        R_mat = expand_mat(self.R)
        H_mat = expand_mat(self.H) if self.H is not None else torch.eye(self.dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 单位矩阵，用于正则化和更新
        I = torch.eye(self.dim, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        x = x_init
        P = P_init
        
        xres_lst = []
        pres_lst = []
        
        for time_step in range(seq_length):
            z_t = z[:, time_step, :]
            mask_t = mask[:, time_step]
            
            # --- 预测步 ---
            # x_{k|k-1} = F * x_{k-1|k-1}
            pred_x = torch.matmul(F_mat, x.unsqueeze(-1)).squeeze(-1)
            
            # P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
            P_pred = torch.matmul(F_mat, torch.matmul(P, F_mat.transpose(-2, -1))) + Q_mat
            
            # 数值稳定性：保证 P_pred 对称正定
            P_pred = 0.5 * (P_pred + P_pred.transpose(-2, -1)) # 强制对称
            P_pred = P_pred + 1e-6 * I # 强制正定
            
            # --- 更新步 ---
            # y_k = z_k - H * x_{k|k-1}
            y = z_t - torch.matmul(H_mat, pred_x.unsqueeze(-1)).squeeze(-1)
            
            # S_k = H * P_{k|k-1} * H^T + R
            S = torch.matmul(H_mat, torch.matmul(P_pred, H_mat.transpose(-2, -1))) + R_mat
            S = 0.5 * (S + S.transpose(-2, -1)) # 强制对称
            S = S + 1e-6 * I # 强制正定
            
            # 计算卡尔曼增益 K = P_{k|k-1} * H^T * S^{-1}
            # 为了避免直接求逆，使用 torch.linalg.solve
            # solve(S, (P * H^T)^T) -> 求解 S * X = (P * H^T)^T
            P_H_T = torch.matmul(P_pred, H_mat.transpose(-2, -1))
            K = torch.linalg.solve(S, P_H_T.transpose(-2, -1)).transpose(-2, -1)
            
            # x_{k|k} = x_{k|k-1} + K * y_k
            x_k_update = pred_x + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
            
            # 处理极少数情况下的 NaN (通常加上正则化后不会出现)
            if torch.isnan(x_k_update).any():
                x_k_update = pred_x
            
            # P_{k|k} = (I - K * H) * P_{k|k-1} * (I - K * H)^T + K * R * K^T (Joseph形式)
            IKH = I - torch.matmul(K, H_mat)
            P_k_update = torch.matmul(IKH, torch.matmul(P_pred, IKH.transpose(-2, -1))) + \
                        torch.matmul(K, torch.matmul(R_mat, K.transpose(-2, -1)))
            
            # 保证更新后的 P 也是对称正定的
            P_k_update = 0.5 * (P_k_update + P_k_update.transpose(-2, -1))
            P_k_update = P_k_update + 1e-6 * I
            
            # --- 应用掩码 ---
            # mask=1: 使用更新值; mask=0: 仅使用预测值 (等同于跳过更新步)
            mask_b1 = mask_t.unsqueeze(1).to(x_k_update.dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(P_k_update.dtype)
            
            x = x_k_update * mask_b1 + pred_x * (1.0 - mask_b1)
            P = P_k_update * mask_b11 + P_pred * (1.0 - mask_b11)
            
            # 存储结果
            xres_lst.append(x)
            pres_lst.append(P)
        
        # 结果堆叠与转置
        xres = torch.stack(xres_lst, dim=1) # (batch, seq, dim)
        pres = torch.stack(pres_lst, dim=1) # (batch, seq, dim, dim)
        
        return xres, pres