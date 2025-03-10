import pandas as pd
import numpy as np
import re

def read_dat_file(file_path):
    """手动解析 .dat 文件，去除括号并转换数据格式"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = re.split(r'\s+', line.strip(), maxsplit=1)  # 按第一个空格拆分
            epoch = int(parts[0])  # 解析 epoch
            values = eval(parts[1])  # 解析 [x, y, z] 这样的列表
            data.append([epoch] + values)
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    df.rename(columns={0: 'epoch'}, inplace=True)
    return df

def read_timelog_file(file_path):
    """读取 timelog.csv 文件"""
    timelog = pd.read_csv(file_path, names=['epoch', 'time'], skiprows=1)  # 假设有表头，跳过第一行
    return timelog

def calculate_error_thresholds(data, col_index, base_value, thresholds=[0.1, 0.05], window_size=500):
    """计算训练过程中首次达到指定误差以下的 epoch，并确保 5% 误差收敛稳定"""
    if col_index not in data.columns:
        raise ValueError(f"列索引 {col_index} 超出范围！")
    
    col_data = data[['epoch', col_index]].copy()
    
    # 计算误差
    col_data['error'] = abs(col_data[col_index] - base_value) / base_value
    
    results = {}
    
    for threshold in thresholds:
        stable_epoch = None
        i = 0
        while i <= len(col_data) - window_size:
            window = col_data.iloc[i:i + window_size]  # 取滑动窗口数据
            if (window['error'] > threshold).any():  # 只要窗口内有超出阈值的点，就跳过这段不稳定区域
                i += window_size  # 跳过整段不稳定区域
                continue
            
            # 发现一个窗口内所有点均 <= 阈值，检查是否稳定
            stable_candidate = window.iloc[0]['epoch']
            
            # 确保 `window_size` 之后的点仍然稳定
            stability_check = col_data.iloc[i + window_size: i + 2 * window_size]
            if (stability_check['error'] <= threshold).all():
                stable_epoch = stable_candidate
                break
            
            i += window_size  # 如果不稳定，跳过这个窗口
        
        results[threshold] = stable_epoch  # 返回第一个稳定收敛的点
    
    return results

def find_closest_time(epoch, timelog):
    """在 timelog.csv 里找到最接近的时间"""
    if epoch is None:
        return None  # 没有达到阈值，返回 None
    diff = abs(timelog['epoch'] - epoch)
    return timelog.loc[diff.idxmin(), 'time']

def main(dat_file, timelog_file, col_index, base_value):
    # 读取数据
    data = read_dat_file(dat_file)
    timelog = read_timelog_file(timelog_file)
    
    # 将 col_index 调整为 0-based（因为输入是 1-based）
    col_index = col_index
    if col_index < 1 or col_index >= len(data.columns):
        raise ValueError("输入的列索引超出范围！")
    
    # 计算误差首次小于阈值的 epoch，确保 5% 误差收敛
    error_epochs = calculate_error_thresholds(data, col_index, base_value, window_size=10)
    
    # 查找对应时间
    time_results = {thr: find_closest_time(epoch, timelog) if epoch is not None else None
                    for thr, epoch in error_epochs.items()}
    
    # 输出结果
    for thr in error_epochs:
        if error_epochs[thr] is None:
            print(f"误差未能稳定收敛到 {thr*100:.0f}%，或从未达到此阈值")
        else:
            print(f"误差首次稳定小于 {thr*100:.0f}% 的 epoch: {error_epochs[thr]}, 时间: {time_results[thr]}")
    
    return error_epochs, time_results

# 示例调用（手动输入）
if __name__ == "__main__":
    dat_file = "PI_variables_ModelSave_0226.dat"  # 替换为实际路径
    timelog_file = "timelog.csv"  # 替换为实际路径
    col_index = int(input("输入要计算的列索引（从1开始）: "))
    base_value = float(input("输入用于计算误差的基准值: "))
    
    main(dat_file, timelog_file, col_index, base_value)
