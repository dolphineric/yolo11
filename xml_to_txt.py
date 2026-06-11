import xml.etree.ElementTree as ET
import os

# 1. 定義類別名稱（順序必須與標註時一致）
classes = ["1","2"] 

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(output_path, 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

# 2. 設定路徑
input_dir = './data/labels/train'   # XML 來源資料夾
output_dir = './data/labels/train' # TXT 輸出資料夾

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 執行轉換
for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        xml_path = os.path.join(input_dir, filename)
        txt_path = os.path.join(output_dir, filename.replace('.xml', '.txt'))
        convert_annotation(xml_path, txt_path)
        print(f"Processed: {filename}")

print("轉換完成！")