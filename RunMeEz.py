import subprocess
import threading

def run_script(script_path):
    """
    运行指定的脚本，并在需要时自动发送回车键。
    Auto Script Running
    """
    # 使用Popen启动脚本，允许向其发送输入，同时捕获输出 Popen start the script
    with subprocess.Popen(['python', script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        # 检测是否需要输入（例如，等待回车键），如果是，则自动提供 detect the output
        while True:
            output_line = proc.stdout.readline()
            if not output_line:
                break  # 如果没有输出，跳出循环 if there is no output, then break
            print(f"{script_path}: {output_line}", end='')
            if "press Enter to continue" in output_line or "请按回车继续" in output_line:  # 根据实际提示信息调整条件
                proc.stdin.write('\n')
                proc.stdin.flush()

# 定义要运行的脚本路径 PATH SETTING
script_paths = [
    "./PINN-for-inverter-control-loop/PID_MAIN.py",
    "./PINN-for-Synchronous/SYNC_MAIN.py"
]

# 创建并启动线程
# Creating and starting a thread
threads = []
for script_path in script_paths:
    thread = threading.Thread(target=run_script, args=(script_path,))
    thread.start()
    threads.append(thread)

# 等待所有线程完成 WAITING FOR RUNNING
for thread in threads:
    thread.join()

print("ALL DONE")
