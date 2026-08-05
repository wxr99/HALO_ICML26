import torch
import torch.nn as nn
import torch.nn.functional as F

class ACILDynamicClasses(nn.Module):
    def __init__(self, feature_dim, expansion_dim, num_classes, gamma=0.1, device="cpu"):
        """
        Incremental learning head based on ACIL.
        Args:
            feature_dim (int): Input feature dimension.
            expansion_dim (int): Expanded feature dimension (hidden size).
            num_classes (int): Number of classes for classification.
            gamma (float): Regularization strength.
            device (str): Device on which the computation will be performed.
        """
        super(ACILDynamicClasses, self).__init__()
        self.feature_dim = feature_dim
        self.expansion_dim = expansion_dim
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device

        # Initialize feature expansion weights
        self.W_fe = nn.Parameter(torch.randn(self.feature_dim, self.expansion_dim, device=self.device) * 0.1)
        self.R = None  # Regularization matrix
        self.weight = nn.Parameter(torch.zeros((self.expansion_dim, self.num_classes), device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ACIL head.
        """
        # Feature expansion
        features = torch.relu(torch.matmul(x, self.W_fe))
        # Classification
        outputs = torch.matmul(features, self.weight)
        return outputs

    def base_training(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Base training for the head.
        """
        # 使用 torch.no_grad() 防止梯度累积
        with torch.no_grad():
            # Filter valid samples
            valid_mask = Y_train != -1
            if valid_mask.sum() == 0:
                return
            
            X_train_valid = X_train[valid_mask]
            Y_train_valid = Y_train[valid_mask]

            # Convert to one-hot encoding
            Y_train_one_hot = F.one_hot(Y_train_valid, num_classes=self.num_classes).float()

            # 分离梯度，确保不会累积计算图
            X_fe = torch.relu(torch.matmul(X_train_valid.detach(), self.W_fe.detach()))
            X_fe_T = X_fe.T
            
            # 计算 R 矩阵
            self.R = torch.linalg.inv(
                torch.matmul(X_fe_T, X_fe) + self.gamma * torch.eye(X_fe.shape[1], device=self.device)
            )
            
            # 更新权重，使用 .data 避免梯度追踪
            self.weight.data = torch.matmul(self.R, torch.matmul(X_fe_T, Y_train_one_hot))

    def incremental_learning(self, X_train: torch.Tensor, Y_train: torch.Tensor):
        """
        Incremental learning for the head.
        """
        # 使用 torch.no_grad() 防止梯度累积
        with torch.no_grad():
            # Filter valid samples
            valid_mask = Y_train != -1
            if valid_mask.sum() == 0:
                return

            X_train_valid = X_train[valid_mask]
            Y_train_valid = Y_train[valid_mask]

            # Convert to one-hot encoding
            Y_train_one_hot = F.one_hot(Y_train_valid, num_classes=self.num_classes).float()

            # Feature expansion - 分离梯度
            # print(X_train_valid.shape, self.W_fe.shape)
            X_fe = torch.relu(torch.matmul(X_train_valid.detach(), self.W_fe.detach()))
            X_fe_T = X_fe.T

            # Update R and weights using Woodbury matrix identity
            if self.R is None:
                self.R = torch.linalg.inv(
                    torch.matmul(X_fe_T, X_fe) + self.gamma * torch.eye(X_fe.shape[1], device=self.device)
                )
            else:
                # 使用 Woodbury 恒等式更新 R
                # 创建中间变量避免重复计算
                R_X_fe_T = torch.matmul(self.R, X_fe_T)
                X_fe_R = torch.matmul(X_fe, self.R)
                
                K = torch.inverse(
                    torch.eye(X_train_valid.shape[0], device=self.device) + torch.matmul(X_fe, R_X_fe_T)
                )
                
                # 更新 R，确保不保留旧的 R 的引用
                self.R = self.R - torch.matmul(R_X_fe_T, torch.matmul(K, X_fe_R))

            # Update weights - 计算残差并更新
            predictions = torch.matmul(X_fe, self.weight.data)
            residual = Y_train_one_hot - predictions
            
            # 使用 .data 更新权重，避免梯度追踪
            self.weight.data = self.weight.data + torch.matmul(self.R, torch.matmul(X_fe_T, residual))

    def reset_memory(self):
        """
        手动重置 R 矩阵以释放内存
        """
        if self.R is not None:
            del self.R
            self.R = None
        torch.cuda.empty_cache()