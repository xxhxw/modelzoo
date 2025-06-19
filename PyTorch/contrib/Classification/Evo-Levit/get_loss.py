import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def extract_loss_values(log_file):
    """从日志文件中提取loss值"""
    loss_values = []
    # 支持多种常见的loss格式（如 "loss: 0.123", "loss = 0.123", "Loss 0.123"）
    patterns = [
        r'loss[:\s=]+([\d\.e-]+)',  # 匹配 "loss: 0.123", "loss=0.123", "loss 0.123"
        r'Loss:\s+([\d\.e-]+)',     # 匹配 "Loss: 0.123"
        r'train\.loss\s+([\d\.e-]+)' # 匹配 "train.loss 0.123"
    ]
    
    with open(log_file, 'r') as file:
        for line in file:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        loss_values.append(float(match.group(1)))
                        break  # 匹配到一个模式后跳出循环
                    except ValueError:
                        continue
    return loss_values

def calculate_errors(sdaa_loss, cuda_loss):
    """计算MRE和MAE"""
    sdaa_loss = np.array(sdaa_loss)
    cuda_loss = np.array(cuda_loss)
    
    # 确保两个数组长度相同（取较短的长度）
    min_length = min(len(sdaa_loss), len(cuda_loss))
    sdaa_loss = sdaa_loss[:min_length]
    cuda_loss = cuda_loss[:min_length]
    
    # 计算误差
    mre = np.mean((sdaa_loss - cuda_loss) / cuda_loss)
    mae = np.mean(np.abs(sdaa_loss - cuda_loss))
    
    return mre, mae

def plot_comparison(sdaa_loss, cuda_loss):
    """绘制比较曲线图"""
    plt.figure(figsize=(12, 6))
    
    # 平滑曲线（可选）
    if len(sdaa_loss) > 5:
        sdaa_smoothed = savgol_filter(sdaa_loss, window_length=5, polyorder=1)
    else:
        sdaa_smoothed = sdaa_loss
    
    if len(cuda_loss) > 5:
        cuda_smoothed = savgol_filter(cuda_loss, window_length=5, polyorder=1)
    else:
        cuda_smoothed = cuda_loss
    
    # 绘制曲线
    x = range(len(sdaa_loss))
    plt.plot(x, sdaa_smoothed, label='SDAA Loss', color='blue')
    plt.plot(x, cuda_smoothed, '--', label='CUDA Loss', color='red')
    
    plt.xlabel('Iteration/Step')
    plt.ylabel('Loss Value')
    plt.title('Loss Comparison: SDAA vs CUDA')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig('loss_comparison.png')
    plt.close()
    print("Comparison plot saved as 'loss_comparison.png'")

def main():
    # 从日志文件提取loss值
    sdaa_loss = extract_loss_values('sdaa.log')
    cuda_loss = extract_loss_values('cuda.log')
    
    if not sdaa_loss or not cuda_loss:
        print("Error: Could not extract loss values from one or both log files.")
        return
    
    print(f"SDAA loss values count: {len(sdaa_loss)}")
    print(f"CUDA loss values count: {len(cuda_loss)}")
    
    # 计算误差
    mre, mae = calculate_errors(sdaa_loss, cuda_loss)
    print(f"Mean Relative Error (MRE): {mre:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    
    # 判断结果
    threshold_mre = 0.05
    threshold_mae = 0.0002
    if mre <= threshold_mre or mae <= threshold_mae:
        result = "PASS"
    else:
        result = "FAIL"
    
    print("\nTest Result:")
    print(f"{result} - MRE ({mre:.6f}) <= {threshold_mre} or MAE ({mae:.6f}) <= {threshold_mae}")
    
    # 绘制比较图
    plot_comparison(sdaa_loss, cuda_loss)

if __name__ == "__main__":
    main()

# Mean Relative Error (MRE): 0.001798
# Mean Absolute Error (MAE): 0.018771

# Test Result:
# PASS - MRE (0.001798) <= 0.05 or MAE (0.018771) <= 0.0002

