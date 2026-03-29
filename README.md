# 医疗设备识别系统

## 项目结构

```
medical-device-detector/
├── weapp/                          # 微信小程序前端
│   ├── pages/
│   │   └── index/
│   │       ├── index.wxml
│   │       ├── index.wxss
│   │       ├── index.js
│   │       └── index.json
│   ├── app.js
│   ├── app.json
│   ├── app.wxss
│   └── project.config.json
├── webapp/                         # Web网页前端
│   └── index.html
├── backend/                        # Python后端服务
│   ├── app.py                      # Flask API服务
│   ├── detector.py                 # YOLOv8检测器
│   ├── requirements.txt
│   ├── yolov8n.pt                  # YOLOv8n模型文件
│   └── uploads/                    # 上传文件目录
└── docs/                           # 文档
    └── DEPLOYMENT.md

```

## 技术栈

- **前端**: 微信小程序 + HTML5/CSS3/JavaScript
- **后端**: Python + Flask
- **AI模型**: YOLOv8n (Ultralytics)
- **部署**: 本地开发

## 支持的设备类别

1. 注射器
2. 持针器
3. 医疗废物
4. 大纱布
5. 碘伏

## 快速开始

### 1. 后端启动

```bash
cd backend
pip install -r requirements.txt
python app.py
```

后端服务启动后会显示：
```
Server starting on http://0.0.0.0:5000
```

### 2. Web网页前端启动

**方法一：使用 VS Code Live Server（推荐）**

1. 在 VS Code 中安装「Live Server」插件
2. 右键点击 `webapp/index.html`
3. 选择「Open with Live Server」

**方法二：使用 Python HTTP 服务器**

```bash
cd webapp
python -m http.server 5500
```

然后在浏览器中访问：`http://localhost:5500/index.html`

### 3. 微信小程序前端

使用[微信开发者工具](https://developers.weixin.qq.com/miniprogram/dev/devtools/download.html)打开 `weapp` 目录

## 手机真机调试

### 环境要求

- 手机和电脑连接**同一个 WiFi**
- 后端服务已启动（`python backend/app.py`）
- 电脑防火墙已开放 5000 和 5500 端口

### 真机调试步骤

1. **打开微信开发者工具**

   导入项目：`weapp` 目录

2. **编译项目**

   - 点击右上角的「编译」按钮（或按 `Ctrl+B`）
   - 等待编译完成，确认模拟器中显示小程序界面

3. **配置本地设置**

   - 点击右上角「详情」→「本地设置」
   - 勾选☑️「不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书」

4. **开启真机调试**

   - 点击右上角的「真机调试」按钮（或按 `F10`）
   - 点击「开始调试」

5. **手机连接**

   - 微信开发者工具会生成一个二维码
   - 用手机微信扫描二维码
   - 手机会打开小程序，开始调试

6. **测试识别**

   - 在小程序中点击「服务器设置」
   - 确认服务器地址是 `http://电脑IP地址:5000`
   - 点击「测试连接」
   - 连接成功后上传图片进行识别

### 查看电脑IP地址

```bash
ipconfig
```

查找「IPv4 地址」，例如：`192.168.16.102`

## 上传与发布

### 1. 上传小程序

1. 在微信开发者工具中点击右上角「上传」
2. 填写版本号（如 `1.0.0`）和备注
3. 点击「上传」
4. 等待上传完成

### 2. 添加体验成员（可选）

让身边朋友扫码体验：

1. 打开 https://mp.weixin.qq.com
2. 扫码登录微信公众平台
3. 点击「成员管理」→「体验成员」
4. 点击「添加体验成员」
5. 输入朋友的微信号
6. 朋友微信会收到邀请，点击接受后即可扫码体验

### 3. 提交审核

1. 在微信公众平台点击「版本管理」→「上传的版本」
2. 点击「提交审核」
3. 填写审核信息：
   - 功能描述：医疗设备识别，上传图片自动识别设备类型
   - 测试账号：可留空
4. 点击「提交审核」

### 4. 发布

审核通过后：

1. 在微信公众平台点击「发布」
2. 小程序即可在微信中搜索到

---

## API接口

- `POST /api/detect` - 上传图片进行检测
  - 参数: `image` (文件)
  - 返回: 检测结果和描述
- `GET /api/health` - 健康检查
- `GET /api/detect_test` - 测试接口
