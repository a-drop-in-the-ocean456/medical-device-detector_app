#!/bin/bash
# 医疗设备检测服务停止脚本

cd "$(dirname \"$0\")\"

if [ -f logs/app.pid ]; then
    PID=$(cat logs/app.pid)
    if kill $PID 2>/dev/null; then
        echo \"服务已停止 (PID: $PID)\"
        rm logs/app.pid
    else
        echo \"服务进程不存在或已停止\"
        rm logs/app.pid
    fi
else
    echo \"未找到服务PID文件，服务可能未运行\"
fi
