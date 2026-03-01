"""
医疗设备检测器 - 基于YOLOv8
使用best.pt预训练模型进行医疗设备检测，针对医疗质量控制中的5类物品进行识别
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import io
import base64

# 医疗设备描述信息
DEVICE_DESCRIPTIONS = {
    'syringe': {
        'name': '注射器',
        'description': '一次性医疗用品——2ml注射器\n\n一、采购与验收管理\n1、供应商资质审核：确认供应商是否具备合法资质（如医疗器械生产许可证、经营许可证），且在有效期内。产品是否在供应商注册范围内，并核对注册证号与产品一致性。\n2、产品注册与备案：核对2ml注射器的医疗器械注册证或备案凭证，确保产品已通过国家药监局审批。确认产品技术要求（如规格、材质、灭菌方式）与注册信息一致。\n\n二、包装与标识检查\n1、外包装完整性：检查注射器外包装是否完好，无破损、变形、潮湿或污染迹象。确认包装材料符合标准（如医用透析纸或塑料复合膜），且封口严密。\n2、标识信息：核对包装上是否标注生产日期、有效期、灭菌批号、灭菌方式。确认内包装（如独立塑封）是否完整，注射器针头保护套是否牢固，无脱落风险。\n\n三、储存条件核查\n1、环境要求：检查储存区域是否符合条件——清洁、干燥、通风良好，温度控制在10℃-30℃，湿度≤70%。远离热源、腐蚀性气体及化学污染物。定期进行空气消毒并记录。\n2、存放规范：确认注射器按灭菌日期顺序摆放（先到期先使用），避免过期。检查是否与未灭菌物品、清洁物品或医疗废物分开放置，防止交叉污染。确认货架或柜子是否稳固，避免注射器因挤压或坠落导致包装破损。\n3、防护措施：检查储存区是否配备防尘罩、防潮箱或密闭柜，防止灰尘和潮湿影响产品质量。\n\n四、使用过程监督\n1、操作前准备：监督医护人员执行手卫生。确认操作台面清洁，无污染风险。\n2、无菌原则遵守：确认注射器开启后立即使用，未使用部分不得放回原包装或重复使用。监督针头使用后是否回套针帽，避免刺伤风险。\n3、使用后处理：确认使用后的注射器按损伤性废物分类处理。'
    },
    'Needle holder': {
        'name': '持针器',
        'description': '医疗器械——持针器\n\n一、采购与验收管理\n1、供应商资质审核：确认供应商是否具备合法资质（如医疗器械生产许可证、经营许可证），且在有效期内。\n2、产品注册与备案：核对持针器的医疗器械注册证或备案凭证，确保产品已通过国家药监局审批。\n\n二、使用管理\n1、一人一用一更换：确保每次使用后均进行彻底清洁、消毒和灭菌，避免交叉感染。\n2、操作前检查：使用前检查持针器外观是否完整，有无破损、变形或锈蚀；功能是否正常，如钳口开合是否顺畅、夹持力是否足够。\n\n三、清洁与消毒\n1、清洁流程：持针器使用后应立即进行初步冲洗，去除表面可见污染物。随后在专用清洗池或清洗机中，使用含酶清洗剂进行彻底清洗，确保所有缝隙和关节处无残留物。\n2、消毒方法：根据持针器材质和污染程度选择合适的消毒方法。如为耐高温、耐湿器械，可采用高压蒸汽灭菌；如为不耐高温器械，可采用低温等离子灭菌或化学浸泡消毒。\n3、消毒效果监测：定期对消毒后的持针器进行微生物学监测，确保消毒效果符合标准要求。\n\n四、灭菌监测\n1、物理监测：每次灭菌时记录灭菌温度、压力、时间等参数，确保灭菌过程符合规定。\n2、化学监测：在灭菌包内放置化学指示卡，通过颜色变化判断灭菌效果。同时，在灭菌包外粘贴化学指示胶带，作为灭菌过程的外部标识。\n3、生物监测：每周至少进行一次生物监测，采用嗜热脂肪杆菌芽孢等标准菌株，验证灭菌器的灭菌效果。\n\n五、储存与维护\n1、储存条件：灭菌后的持针器应存放在干燥、清洁、通风良好的环境中，避免潮湿和污染。储存架或柜应离地面20~25厘米，离天花板50厘米，离墙5厘米。\n2、有效期管理：根据灭菌方式和包装材料确定持针器的有效期，并严格按照有效期顺序使用。过期器械应重新进行清洗、消毒和灭菌处理。\n3、维护保养：定期对持针器进行维护保养，如润滑关节、更换磨损部件等，确保其处于良好状态。\n\n六、追溯：建立医疗器械的追溯系统，记录其采购、验收、使用、清洁、消毒、灭菌、储存等全过程信息。'
    },
    'waste': {
        'name': '医疗废物',
        'description': '医疗废物——感染性废物\n\n一、分类收集\n1、是否与其它废物混放：检查感染性废物是否规范置于黄色医疗废物袋内，是否与生活垃圾、损伤性废物、药物性废物、化学性废物混放。\n2、隔离的传染病病人或者疑似传染病病人产生的医疗废物（包括生活垃圾）应使用双层包装物，并及时密封。\n3、实验室产生的培养基、标本和菌种、毒种保存液等应在产生地点进行压力蒸汽灭菌或者化学消毒处理，再按感染性废物收集处理。\n4、收集容器：检查是否使用专用的黄色医疗废物袋或容器收集感染性废物，容器外表面是否有警示标识。\n\n二、包装标识\n1、警示标识：盛装医疗废物的包装物、容器外表面应有警示标识，包括中文标签，内容应包括医疗废物产生单位、产生日期、类别及需要的特别说明。\n2、封口方式：盛装的医疗废物达到包装物或者容器的3/4时，应使用鹅颈式包扎，使包装物或者容器的封口紧实、严密。\n\n三、暂存管理：医疗废物在暂存点的存放时间不得超过2天。\n\n四、运送交接\n1、运送工具：检查运送工具是否符合防渗漏、防遗撒、无锐利边角、易于装卸和清洁等要求。\n2、交接记录：医疗废物交接时应登记来源、种类、重量或者数量、交接时间、最终去向以及经办人签名等项目。交接记录应至少保存3年。\n\n五、人员防护\n1、个人防护：在从事医疗废物分类收集、运送、暂时贮存、处置等工作时，人员应采取适合的职业卫生防护措施，如佩戴手套、口罩、防护服等。\n2、健康监测：医疗废物管理相关人员应定期进行健康检查，必要时进行免疫接种。'
    },
    'large gauze': {
        'name': '大纱布',
        'description': '无菌物品——大纱布\n\n一、包装完整性检查\n1、外包装状态：检查包装是否完整，无破损、撕裂、穿孔或潮湿痕迹。确认包装材料符合标准（如双层灭菌包装：内层为医用皱纹纸或无纺布，外层为布袋或纸塑袋）。\n2、标识与信息：核对包装上是否标注以下信息——物品名称（如“大纱布”）、规格（如尺寸、层数）；灭菌日期、有效期；灭菌批次号、灭菌器编号；“无菌”标识及化学指示卡。\n\n二、灭菌质量验证\n1、化学监测：检查包装内是否放置化学指示卡，且指示卡变色均匀、符合标准。确认化学指示卡位置正确。\n生物监测（如适用）检查生物监测报告是否存档且可追溯。对于高危物品（如植入物相关纱布），需核对是否定期进行生物监测（如嗜热脂肪杆菌芽孢培养），并确认结果阴性。\n2、物理参数监测：确认灭菌过程参数（如温度、压力、时间）是否符合标准（如压力蒸汽灭菌：121℃维持30分钟或132℃维持4分钟）。检查灭菌器运行记录是否完整，无异常报警或故障记录。\n\n三、储存条件核查\n1、环境要求：确认无菌物品存放区符合以下条件——清洁、干燥、通风良好，温度≤24℃，湿度≤70%；远离水源、化学污染物及腐蚀性气体；定期进行空气消毒（如紫外线照射）并记录。\n2、存放规范：检查大纱布是否按灭菌日期顺序摆放（先灭菌先使用，避免过期）。确认物品与地面距离≥20cm，与墙壁距离≥5cm，与天花板距离≥50cm。避免与未灭菌物品、清洁物品或医疗废物混放。\n3、防护措施：观察存放区是否配备防尘罩、防潮箱（如需）或密闭柜。发现破损或潮湿立即重新灭菌。\n\n四、使用过程监督\n1、操作前准备：确认操作人员执行手卫生（如七步洗手法或使用速干手消毒剂）。检查是否佩戴无菌手套，并避免触碰纱布内面或非无菌区域。\n2、无菌原则遵守：监督是否使用无菌持物钳或镊子取用纱布，禁止直接用手抓取。确认纱布打开后未使用部分是否立即丢弃（不可放回原包装）。检查操作台面是否清洁，避免纱布接触污染物品（如未消毒的器械、患者体液）。\n3、使用后处理：确认使用后的纱布按感染性废物处理。'
    },
    'Iodophor': {
        'name': '碘伏',
        'description': '消毒用品——碘伏\n\n一、质量验证\n1、有效碘含量：需检查碘伏的有效碘含量是否符合国家标准（如皮肤消毒用碘伏有效碘浓度需在0.45%~0.55%之间，黏膜消毒浓度不得超过0.1%）。\n2、微生物污染：定期抽检碘伏的微生物污染情况，如菌落总数、霉菌和大肠菌群等指标。\n\n二、包装与标识\n1、包装完整性：检查碘伏的包装是否完整，无破损、泄漏或污染迹象。包装材料应符合相关标准，能够防止碘伏挥发和外界污染。\n2、标识信息：核对碘伏包装上的标识信息是否完整，包括产品名称、规格、生产日期、有效期、生产厂家、使用说明等。\n\n三、储存条件\n1、环境要求：碘伏应储存在干燥、阴凉、通风良好的地方，避免阳光直射和高温环境。储存温度建议控制在25℃以下。\n2、容器选择：碘伏应使用密闭性好的容器储存，避免挥发和污染。容器材质应与碘伏相容，不发生化学反应。\n\n四、使用管理\n1、使用规范：监督医护人员在使用碘伏时是否遵循操作规范，如使用无菌棉球或纱布蘸取碘伏、避免交叉污染、确保消毒范围和时间等。\n2、开封后管理：碘伏开封后应注明开启日期和有效期（一般建议开封后30天内用完），并定期检查其杀菌效果。如发现杀菌效果下降或溶液变质，应立即停止使用。'
    }
}

# 通用描述（当检测到非医疗设备时）
GENERIC_DESCRIPTION = "检测到物体，建议上传更清晰的医疗设备照片以获得准确识别。"


class MedicalDeviceDetector:
    def __init__(self, model_path=None):
        """初始化检测器"""
        # 使用best.pt预训练模型
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # 默认使用best.pt模型
            model_file = 'best.pt'
            if os.path.exists(model_file):
                print(f"Loading {model_file} model...")
                self.model = YOLO(model_file)
            else:
                print("best.pt not found, using yolov8n.pt...")
                self.model = YOLO('yolov8n.pt')
            print("Model loaded successfully!")
        
        # 设置置信度阈值
        self.conf_threshold = 0.25
        
    def detect(self, image_data):
        """
        检测图像中的医疗设备
        
        Args:
            image_data: 可以是文件路径、字节数据或numpy数组
            
        Returns:
            dict: 包含检测结果和可视化图像
        """
        # 处理输入图像
        if isinstance(image_data, str):
            # 文件路径
            image = Image.open(image_data)
            img_array = cv2.imread(image_data)
        elif isinstance(image_data, bytes):
            # 字节数据
            image = Image.open(io.BytesIO(image_data))
            img_array = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            # numpy数组
            img_array = image_data
            image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        
        # 运行YOLO检测
        results = self.model(img_array, conf=self.conf_threshold)
        
        # 解析检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]
                
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 3),
                    'bbox': [x1, y1, x2, y2]
                })
        
        # 生成描述
        description = self._generate_description(detections)
        
        # 绘制检测结果
        vis_image = self._draw_detections(image.copy(), detections)
        
        # 转换为RGB模式（JPEG不支持alpha通道）
        if vis_image.mode == 'RGBA':
            vis_image = vis_image.convert('RGB')
        
        # 转换为base64
        buffered = io.BytesIO()
        vis_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'description': description,
            'image_base64': f'data:image/jpeg;base64,{img_base64}'
        }
    
    def _generate_description(self, detections):
        """生成检测结果的描述文本"""
        if not detections:
            return "未检测到明显的医疗设备或物体，请尝试：\n1. 调整拍摄角度\n2. 确保光线充足\n3. 将设备置于画面中央"
        
        # 按置信度排序
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        descriptions = []
        unique_classes = set()
        
        for det in detections_sorted:
            class_name = det['class']
            if class_name not in unique_classes:
                unique_classes.add(class_name)
                
                if class_name in DEVICE_DESCRIPTIONS:
                    device_info = DEVICE_DESCRIPTIONS[class_name]
                    descriptions.append(f"• {device_info['name']} (置信度: {det['confidence']})\n  {device_info['description']}")
                else:
                    descriptions.append(f"• {class_name} (置信度: {det['confidence']})\n  {GENERIC_DESCRIPTION}")
        
        # 添加总结
        if len(descriptions) > 0:
            summary = f"检测到 {len(detections)} 个物体，识别出 {len(unique_classes)} 种类型：\n\n"
            return summary + '\n\n'.join(descriptions[:5])  # 最多显示5个
        
        return "检测到物体但无法准确识别，建议重新拍摄。"
    
    def _draw_detections(self, image, detections):
        """在图像上绘制检测框和标签"""
        draw = ImageDraw.Draw(image)
        
        # 尝试加载中文字体
        try:
            # Windows系统字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 20)
            except:
                font = ImageFont.load_default()
        
        # 为不同类别分配颜色
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            color = colors[i % len(colors)]
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 准备标签文本
            label = f"{class_name} {confidence:.2f}"
            
            # 获取文本尺寸
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制标签背景
            draw.rectangle(
                [x1, y1 - text_height - 8, x1 + text_width + 8, y1],
                fill=color
            )
            
            # 绘制标签文字
            draw.text((x1 + 4, y1 - text_height - 4), label, fill='white', font=font)
        
        return image


# 测试代码
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detector = MedicalDeviceDetector()
        result = detector.detect(image_path)
        
        print(f"检测完成!")
        print(f"检测到 {result['detection_count']} 个物体")
        print(f"\n描述:\n{result['description']}")
        
        # 保存结果图像
        if result['image_base64']:
            img_data = base64.b64decode(result['image_base64'].split(',')[1])
            with open('result.jpg', 'wb') as f:
                f.write(img_data)
            print("\n结果图像已保存为 result.jpg")
    else:
        print("用法: python detector.py <image_path>")
