from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size
from ultralytics import YOLO
import torch
import os

class YOLOv8LabelStudioML(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8LabelStudioML, self).__init__(**kwargs)
        
        # 优化 CPU 性能
        torch.set_num_threads(4)
        
        # 加载模型
        model_path = '/root/model/best.pt'
        self.model = YOLO(model_path)
        self.model.to('cpu')
        
        print(f"[INFO] 模型已加载: {model_path}")
        
    def predict(self, tasks, **kwargs):
        predictions = []
        
        with torch.no_grad():
            for task in tasks:
                try:
                    # 获取图片路径
                    image_path = self.get_local_path(
                        task['data']['image'], 
                        task_id=task['id']
                    )
                    print(f"[INFO] 处理图片: {image_path}")
                    
                    # 获取图片尺寸
                    image_width, image_height = get_image_size(image_path)
                    
                    # 执行预测
                    results = self.model(image_path)[0]
                    
                    pred = {
                        'model_version': self.model.__class__.__name__,
                        'result': []
                    }
                    
                    # 转换预测结果为 Label Studio 格式
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        pred['result'].append({
                            'type': 'rectanglelabels',
                            'score': confidence,
                            'value': {
                                'rectanglelabels': [class_name],
                                'x': x1 / image_width * 100,
                                'y': y1 / image_height * 100,
                                'width': (x2 - x1) / image_width * 100,
                                'height': (y2 - y1) / image_height * 100
                            }
                        })
                    
                    predictions.append(pred)
                    print(f"[INFO] 成功处理图片")
                    
                except Exception as e:
                    print(f"[ERROR] 处理失败: {str(e)}")
                    continue
        
        return predictions
