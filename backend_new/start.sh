#!/bin/bash
# 医疗设备检测服务启动脚本
# 使用 nohup 在后台运行，断开连接后仍保持运行

cd "$(dirname \"$0\")\"

# 创建日志文件
mkdir -p logs

# 使用 nohup 启动，输出日志到文件
nohup python3 app.py > logs/app.log 2>&1 &

# 保存进程ID
echo $! > logs/app.pid

echo \"服务已启动！\"
echo \"PID: $(cat logs/app.pid)\"
echo \"日志位置: logs/app.log\"
echo \"查看日志: tail -f logs/app.log\"
echo \"停止服务: kill $(cat logs/app.pid)\"
