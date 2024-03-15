# -*- coding: utf-8 -*-
"""
''''''
python版本大于3.6, 谌立坤2024-01-17更新chenlk@whu.edu.cn
用本地机器作为承载接口127.0.0:1117的网络地址与RTDS通信
对应的文件为ListenOnPort2.scr
不需要在runtime里点击运行，先运行scr文件再运行py文件即可
多写点注释防止我忘记以前写这坨屎是干什么用的

scr源代码 RTDS运行文件名称为ListenOnPort.scr
float temp_float;
float temp_float2;
string temp_string;
fprintf(stdmsg, "Initialization of RTDS simulation");
ListenOnPort(1117, true); 
fprintf(stdmsg, "Execution of script is done\n");
''''''
"""


import socket
import time
import re

# Initialization parameters
TCP_IP = '127.0.0.1'
TCP_PORT = 1117
BUFFER_SIZE = 8192

# 就是“键 = 值”的形式完成正则化
data_pattern = re.compile(r'(\w+)\s*=\s*(-?\d+\.\d+)')

# Create a TCP/IP socket 建立通信
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

# RTDS下载仿真模型，并等待微电网启动
s.send(f'Start;'.encode()) 
s.send(f'SUSPEND 10;'.encode())  

# 用于计时Start timing
start = time.perf_counter()

# 以字典形式匹配数据，以后都按照XXX = %f的格式来数据输入输出Function to read multiple data points from the returned byte data
def read_data(s, data_buffer=b''):
    while True:
        data = s.recv(BUFFER_SIZE)
        if not data:
            break  # 防止没有数据的时候死循环导致RTDS内存溢出If no data, break the loop
        data_buffer += data

        # Decode data buffer and find all matches
        decoded_data = data_buffer.decode('utf-8')
        matches = data_pattern.findall(decoded_data)

        if matches:
            data_dict = {name: float(value) for name, value in matches}
            end_match = re.search(r'END\r\n', decoded_data) # 最后以end结尾 end with 'END'
            if end_match:
                return data_dict, data_buffer[end_match.end():]  # Return the data dictionary and remaining buffer

# 用来改变RTDS滑块，输送计算好的值，Function to send calculated PREF value back to RTDS
def send_pref(s, pref):
    s.send(f'SetSlider "PREF" = {pref};'.encode())
    s.send(f'SUSPEND 0.2;'.encode())

# 功能函数区，可以做能量平衡、状态检测等等
def calculate_pref(data_dict):
    ppv = data_dict.get('PPV', 0)
    pload = data_dict.get('PLOAD', 0)
    pgen = data_dict.get('PGEN', 0)
    return pload - ppv - pgen


# 发送RTDS命令截取需要的数据
def DataCapture():
    s.send(f'grid_status = SwitchCapture("GRID");'.encode()) 
    s.send(f'temp_float = MeterCapture("PPV");'.encode())
    s.send(f'temp_float2 = MeterCapture("PLOAD");'.encode())
    s.send(f'temp_float3 = MeterCapture("PGEN");'.encode())


# 让RTDS把截取的数据返回，保持“键 = 值”的模板不变
def DataReturn():
    s.send(f'sprintf(temp_string, "GRID = %d PPV = %f PLOAD  = %f PGEN = %f END ", grid_status, temp_float, temp_float2, temp_float3);'.encode())
    s.send(f'ListenOnPortHandshake(temp_string);'.encode())


# Perform 100 iterations or more
remaining_data = b''
for i in range(1, 101):
    print(f'Iteration number: {i}')
    
    # 等待稳定
    s.send(f'SUSPEND 0.2;'.encode())
    
    DataCapture()
    DataReturn()
    
    
    # Read and parse data
    data_dict, remaining_data = read_data(s, remaining_data)
    if 'PPV' in data_dict and 'PLOAD' in data_dict:
        print(f"EMS Monitoring ppv = {data_dict['PPV']}, pload = {data_dict['PLOAD']}, pgen = {data_dict['PGEN']}")
        print(f"EMS Monitoring Current Running Status = {data_dict['GRID']}")
        # Calculate PREF and send it back to RTDS
        pref = calculate_pref(data_dict)
        send_pref(s, pref)

# Stop simulation and close the connection properly
try:
    s.send(f'Stop;'.encode())
    time.sleep(1)  # Wait for RTDS to process the stop command
    s.send(f'ClosePort({TCP_PORT})'.encode())
except IOError as e:
    print(f"IOError occurred: {e}")

# Check if the socket is open and then close it
if s:
    s.close()

# Print execution time
finish = time.perf_counter() - start
print(f'All iterations complete. The execution time is {finish}s.')
