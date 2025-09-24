import torch
import torch.nn as nn

# PyTorch使用训练时缩放
def demonstrate_pytorch_dropout():
    print("PyTorch Dropout实现:")
    
    dropout = nn.Dropout(p=0.5)
    x = torch.ones(5)  # [1, 1, 1, 1, 1]
    
    # 训练模式（启用缩放）
    dropout.train()
    output_train = dropout(x)
    print(f"训练模式输出: {output_train}")
    print(f"训练输出均值: {output_train.mean():.3f}")
    
    # 推理模式（无缩放）
    dropout.eval()
    output_eval = dropout(x)
    print(f"推理模式输出: {output_eval}")
    print(f"推理输出均值: {output_eval.mean():.3f}")

demonstrate_pytorch_dropout()