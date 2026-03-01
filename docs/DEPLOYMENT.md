# 医疗设备检测系统 - 部署指南

## 目录
- [快速启动](#快速启动)
- [后台运行](#后台运行)
- [系统服务部署](#系统服务部署)
- [常见问题](#常见问题)

## 快速启动

```bash
cd backend
python3 app.py
```

服务将在 `http://0.0.0.0:5000` 启动

## 后台运行

### 方法一：使用启动脚本（推荐）

```bash
# 启动服务
cd backend
chmod +x start.sh stop.sh
./start.sh

# 查看日志
tail -f logs/app.log

# 停止服务
./stop.sh
```

### 方法二：使用 nohup

```bash
cd backend
nohup python3 app.py > app.log 2>&1 &
echo $! > app.pid  # 保存进程ID
```

### 方法三：使用 screen

```bash
# 创建新会话
screen -S detector

# 在会话中启动
python3 app.py

# 分离会话（按 Ctrl+A 然后按 D）
```

### 方法四：使用 tmux

```bash
# 创建新窗口
tmux new -s detector

# 在窗口中启动
python3 app.py

# 分离（按 Ctrl+B 然后按 D）
```

## 系统服务部署（生产环境）

创建服务文件 `/etc/systemd/system/detector.service`：

```ini
[Unit]
Description=Medical Device Detection API
After=network.target

[Service]
User=root
WorkingDirectory=/path/to/medical-device-detector/backend
ExecStart=/usr/bin/python3 /path/to/medical-device-detector/backend/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

然后执行：

```bash
systemctl daemon-reload
systemctl enable detector
systemctl start detector
systemctl status detector
```

## 常见问题

### Q: 连接断开后服务停止？
A: 使用 `nohup`、`screen` 或 `systemd` 可以在后台持续运行。

### Q: 如何检查服务是否运行？
A: 执行 `curl http://localhost:5000/api/health`

### Q: 如何查看日志？
A: `tail -f backend/logs/app.log`

### Q: 如何停止服务？
A: `kill $(cat backend/logs/app.pid)` 或 `./backend/stop.sh`
