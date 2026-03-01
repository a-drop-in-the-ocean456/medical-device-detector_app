"""
================================================================================
医疗设备识别后端API服务
================================================================================
功能: 使用Flask框架提供RESTful API接口，用于接收图片并进行医疗设备识别
技术: Flask + YOLOv8 + CORS跨域支持
================================================================================
"""

# =========================== 导入必要的模块 ===========================

from flask import Flask, request, jsonify  # Flask框架核心组件：Flask应用对象、请求对象、JSON响应
from flask_cors import CORS  # 跨域资源共享(CORS)支持，允许前端跨域请求
import os  # 操作系统相关功能，如文件路径操作、目录创建
import io  # 输入输出流，用于处理内存中的二进制数据
import base64  # Base64编码解码，用于处理Base64格式的图片数据
from PIL import Image  # Python图像库，用于打开、验证和处理图片
import numpy as np  # 数值计算库，用于图像数据的数组操作
from detector import MedicalDeviceDetector  # 自定义检测器模块，包含YOLOv8模型和检测逻辑

# =========================== 初始化Flask应用 ===========================

# 创建Flask应用实例
# __name__参数用于确定应用根目录，便于加载模板和静态文件
app = Flask(__name__)

# 启用CORS跨域支持
# resources参数指定只对/api/*路径启用CORS
# origins="*"允许所有域名访问API（开发环境使用，生产环境建议限制）
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 配置上传文件大小限制为16MB
# 防止客户端上传过大的文件导致服务器资源耗尽
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB = 16 * 1024 * 1024 字节

# =========================== 初始化检测器 ===========================

# 全局检测器实例（在整个应用生命周期内只初始化一次）
# 这样做的好处是避免每次请求都重新加载模型，提高响应速度
print("Initializing Medical Device Detector...")  # 打印初始化提示信息
detector = MedicalDeviceDetector()  # 创建检测器实例，加载YOLOv8模型
print("Detector ready!")  # 打印初始化完成提示

# =========================== API接口定义 ===========================

# ---------- 根路径：服务状态检查 ----------
@app.route('/')  # 路由装饰器，绑定URL路径'/'到index函数
def index():
    """
    根路径 - 服务状态检查接口
    
    功能: 返回服务的基本信息，用于确认服务是否正常运行
    
    返回JSON:
        {
            "status": "success",  # 状态码
            "message": "Medical Device Detection API is running",  # 状态消息
            "version": "1.0.0",  # API版本号
            "endpoints": {  # 可用接口列表
                "detect": "/api/detect (POST)",
                "health": "/api/health (GET)"
            }
        }
    """
    return jsonify({
        'status': 'success',
        'message': 'Medical Device Detection API is running',
        'version': '1.0.0',
        'endpoints': {
            'detect': '/api/detect (POST)',
            'health': '/api/health (GET)'
        }
    })


# ---------- 健康检查接口 ----------
@app.route('/api/health')  # 路由装饰器，绑定URL路径'/api/health'到health_check函数
def health_check():
    """
    健康检查接口
    
    功能: 检查服务器和模型是否正常运行
    
    返回JSON:
        {
            "status": "healthy",  # 服务器状态
            "model_loaded": True/False  # 模型是否加载成功
        }
    
    使用场景:
        - 前端在发起请求前先检查服务器状态
        - 监控系统定期检查服务健康状况
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None  # 检查检测器是否已初始化
    })


# ---------- 图片检测接口（核心功能） ----------
@app.route('/api/detect', methods=['POST'])  # 只接受POST请求
@app.route('/detect', methods=['POST'])  # 兼容旧路径
def detect():
    """
    图像检测接口 - 医疗设备识别核心功能
    
    功能: 接收图片，调用YOLOv8模型进行目标检测，返回检测结果
    
    接收方式（二选一）:
        1. multipart/form-data: 上传图片文件（字段名: image）
           - 用于微信小程序 wx.uploadFile 上传
           - 用于网页表单上传
        
        2. application/json: base64编码的图片（字段: image_base64）
           - 用于JavaScript fetch API发送JSON数据
           - 数据格式: "data:image/jpeg;base64,/9j/4AAQ..."
    
    返回JSON:
        {
            "success": True/False,  # 是否成功处理
            "detections": [  # 检测结果列表
                {
                    "class": "bottle",  # 检测到的类别名称
                    "confidence": 0.918,  # 置信度（0-1之间）
                    "bbox": [x1, y1, x2, y2]  # 边界框坐标
                },
                ...
            ],
            "detection_count": 2,  # 检测到的物体数量
            "description": "检测到药瓶/试剂瓶...",  # 详细描述文本
            "image_base64": "data:image/jpeg;base64,..."  # 带检测框的可视化图像
        }
    
    错误返回:
        400 Bad Request - 未提供图片或图片格式无效
        500 Internal Server Error - 服务器内部错误
    """
    try:
        # 初始化图片数据变量
        image_data = None
        
        # ------------------- 方式1: 从form-data获取文件 -------------------
        # 检查请求中是否包含'image'字段的文件
        if 'image' in request.files:
            file = request.files['image']  # 获取上传的文件对象
            
            # 检查文件名是否为空（防止空文件上传）
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                }), 400  # 返回400状态码
            
            # 读取文件的二进制数据
            image_data = file.read()
            # 打印日志：文件名和文件大小（字节）
            print(f"Received file: {file.filename}, size: {len(image_data)} bytes")
        
        # ------------------- 方式2: 从JSON获取base64图片 -------------------
        # 检查请求是否为JSON格式
        elif request.is_json:
            data = request.get_json()  # 解析JSON请求体
            
            # 检查JSON中是否包含'image_base64'字段
            if 'image_base64' in data:
                base64_str = data['image_base64']  # 获取Base64编码的图片字符串
                
                # 处理Data URL格式（移除前缀）
                # 格式: "data:image/jpeg;base64,/9j/4AAQ..."
                if ',' in base64_str:
                    base64_str = base64_str.split(',')[1]  # 只取逗号后面的编码部分
                
                # 解码Base64字符串为二进制数据
                image_data = base64.b64decode(base64_str)
                print(f"Received base64 image, size: {len(image_data)} bytes")
        
        # ------------------- 验证是否收到图片 -------------------
        # 如果两种方式都没有收到图片数据，返回400错误
        if image_data is None:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please upload via "image" field or provide "image_base64" in JSON'
            }), 400
        
        # ------------------- 验证图片格式 -------------------
        # 尝试用PIL打开图片，验证是否为有效图片
        try:
            img = Image.open(io.BytesIO(image_data))  # 从二进制数据创建图片对象
            print(f"Image format: {img.format}, size: {img.size}")  # 打印图片格式和尺寸
        except Exception as e:
            # 如果打开失败，说明不是有效的图片格式
            return jsonify({
                'success': False,
                'error': f'Invalid image format: {str(e)}'
            }), 400
        
        # ------------------- 执行检测 -------------------
        print("Running detection...")  # 打印开始检测提示
        result = detector.detect(image_data)  # 调用检测器的detect方法
        print(f"Detection complete: {result['detection_count']} objects found")  # 打印检测结果数量
        print(f"Detections: {result['detections']}")  # 打印检测结果详情
        
        # ------------------- 返回结果 -------------------
        return jsonify(result)  # 将检测结果转换为JSON并返回
        
    # ------------------- 异常处理 -------------------
    except Exception as e:
        # 捕获并打印任何异常
        print(f"Error during detection: {str(e)}")
        import traceback  # 导入跟踪模块，用于打印详细错误信息
        traceback.print_exc()  # 打印完整的错误堆栈信息
        
        # 返回500服务器错误
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ---------- 测试接口 ----------
@app.route('/api/detect_test', methods=['GET'])  # 只接受GET请求
def detect_test():
    """
    测试接口 - 返回示例检测结果
    
    功能: 在不实际检测图片的情况下，返回模拟的检测结果
         用于前端开发和调试，验证API接口是否正常工作
    
    返回JSON:
        {
            "success": True,
            "message": "Test endpoint working",
            "example_detection": {
                "detections": [
                    {
                        "class": "bottle",
                        "confidence": 0.89,
                        "bbox": [100, 150, 300, 450]
                    }
                ],
                "detection_count": 1,
                "description": "检测到药瓶/试剂瓶..."
            }
        }
    
    使用方法:
        在浏览器中访问: http://localhost:5000/api/detect_test
    """
    return jsonify({
        'success': True,
        'message': 'Test endpoint working',
        'example_detection': {
            'detections': [
                {
                    'class': 'large gauze',
                    'confidence': 0.89,
                    'bbox': [100, 150, 300, 450]
                }
            ],
            'detection_count': 1,
            'description': '检测到大纱布 (置信度: 0.89)\n无菌物品——大纱布\n\n一、包装完整性检查\n1、外包装状态：检查包装是否完整，无破损、撕裂、穿孔或潮湿痕迹。确认包装材料符合标准（如双层灭菌包装：内层为医用皱纹纸或无纺布，外层为布袋或纸塑袋）。\n2、标识与信息：核对包装上是否标注以下信息——物品名称（如“大纱布”）、规格（如尺寸、层数）；灭菌日期、有效期；灭菌批次号、灭菌器编号；“无菌”标识及化学指示卡。'
        }
    })


# =========================== 错误处理 ===========================

# ---------- 文件过大错误处理 ----------
@app.errorhandler(413)  # 413状态码表示请求体过大
def too_large(e):
    """
    处理文件过大错误的专门处理函数
    
    当上传文件超过MAX_CONTENT_LENGTH限制时触发
    """
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB'
    }), 413


# ---------- 服务器内部错误处理 ----------
@app.errorhandler(500)  # 500状态码表示服务器内部错误
def server_error(e):
    """
    处理服务器内部错误的通用处理函数
    
    捕获未处理的异常，防止服务器崩溃
    """
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# =========================== 启动服务 ===========================

if __name__ == '__main__':
    """
    主程序入口
    
    当直接运行此脚本时（而非作为模块导入），启动Flask开发服务器
    """
    
    # 创建上传文件存储目录
    # exist_ok=True 表示如果目录已存在不会报错
    os.makedirs('uploads', exist_ok=True)
    
    # 打印启动信息
    print("\n" + "="*50)
    print("Medical Device Detection API Server")
    print("="*50)
    print("Server starting on http://0.0.0.0:5000")
    print("API Documentation:")
    print("  POST /api/detect - Upload image for detection")
    print("  GET  /api/health - Health check")
    print("  GET  /api/detect_test - Test endpoint")
    print("="*50 + "\n")
    
    # 启动Flask开发服务器
    # host='0.0.0.0' 表示监听所有网络接口，允许外部访问
    # port=5000 指定端口号
    # debug=True 开启调试模式（自动重载代码，显示详细错误信息）
    # 注意：生产环境应关闭debug模式
    app.run(host='0.0.0.0', port=5000, debug=True)
