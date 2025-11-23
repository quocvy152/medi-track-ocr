from fastapi import FastAPI, UploadFile, File
import easyocr
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
reader = easyocr.Reader(['en', 'vi'])

def group_texts_by_lines(texts, y_threshold=10):
    """
    Nhóm các text OCR theo từng dòng dựa trên tọa độ y của bbox.
    
    Args:
        texts: List các dict chứa text, confidence, bbox
        y_threshold: Khoảng cách y tối đa để coi là cùng dòng (pixels)
    
    Returns:
        List các dict chứa line_number, line_text, confidence (trung bình), bbox (toàn dòng)
    """
    if not texts:
        return []
    
    # Tính y-center cho mỗi text
    text_with_y = []
    for text_item in texts:
        bbox = text_item['bbox']
        # Tính y-center từ bbox (bbox là 4 điểm: top-left, top-right, bottom-right, bottom-left)
        y_coords = [point[1] for point in bbox]
        y_center = sum(y_coords) / len(y_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        text_with_y.append({
            **text_item,
            'y_center': y_center,
            'y_min': y_min,
            'y_max': y_max,
            'x_min': min([point[0] for point in bbox]),
            'x_max': max([point[0] for point in bbox])
        })
    
    # Sắp xếp theo y-center (từ trên xuống)
    text_with_y.sort(key=lambda x: x['y_center'])
    
    # Nhóm các text cùng dòng
    lines = []
    current_line = []
    current_y_center = None
    
    for item in text_with_y:
        if current_y_center is None:
            # Dòng đầu tiên
            current_line = [item]
            current_y_center = item['y_center']
        elif abs(item['y_center'] - current_y_center) <= y_threshold:
            # Cùng dòng
            current_line.append(item)
            # Cập nhật y_center trung bình của dòng
            current_y_center = sum([t['y_center'] for t in current_line]) / len(current_line)
        else:
            # Dòng mới
            if current_line:
                lines.append(current_line)
            current_line = [item]
            current_y_center = item['y_center']
    
    # Thêm dòng cuối cùng
    if current_line:
        lines.append(current_line)
    
    # Xử lý từng dòng: sắp xếp theo x và nối text
    result_lines = []
    for line_num, line_items in enumerate(lines, 1):
        # Sắp xếp các text trong dòng theo x từ trái sang phải
        line_items.sort(key=lambda x: x['x_min'])
        
        # Nối các text lại với nhau
        line_text = ' '.join([item['text'] for item in line_items])
        
        # Tính confidence trung bình
        avg_confidence = sum([item['confidence'] for item in line_items]) / len(line_items)
        
        # Tính bbox của toàn dòng (bounding box bao quanh tất cả text trong dòng)
        x_min = min([item['x_min'] for item in line_items])
        x_max = max([item['x_max'] for item in line_items])
        y_min = min([item['y_min'] for item in line_items])
        y_max = max([item['y_max'] for item in line_items])
        
        line_bbox = [
            [x_min, y_min],  # top-left
            [x_max, y_min],  # top-right
            [x_max, y_max],  # bottom-right
            [x_min, y_max]   # bottom-left
        ]
        
        result_lines.append({
            "line_number": line_num,
            "text": line_text,
            "confidence": float(avg_confidence),
            "bbox": line_bbox,
            "word_count": len(line_items)
        })
    
    return result_lines

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    # Read file content as bytes
    contents = await file.read()
    
    # Validate that we have content
    if not contents:
        return {"success": False, "error": "Empty file uploaded"}
    
    # Create BytesIO object and ensure it's at the beginning
    image_bytes = BytesIO(contents)
    image_bytes.seek(0)
    
    # Open image from bytes
    try:
        image = Image.open(image_bytes)
        # Verify it's a valid image by loading it
        image.verify()
    except Exception as e:
        return {"success": False, "error": f"Cannot identify image file: {str(e)}"}
    
    # Reopen image after verify (verify closes the image)
    image_bytes.seek(0)
    image = Image.open(image_bytes)
    
    # Convert to RGB if necessary (handles RGBA, P, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_np = np.array(image)

    result = reader.readtext(image_np, detail=1)

    # Chuyển đổi kết quả OCR sang format chuẩn và lọc theo confidence >= 0.7
    texts = []
    min_confidence = 0.7
    for bbox, text, confidence in result:
        # Chỉ lấy kết quả có confidence >= 0.7 (70%)
        if confidence >= min_confidence:
            # Convert numpy arrays to lists and numpy scalars to Python types
            bbox_list = [[float(coord) for coord in point] for point in bbox]
            texts.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox_list
            })

    # Nhóm các text theo từng dòng
    lines = group_texts_by_lines(texts, y_threshold=10)
    
    # Lọc lại các dòng có confidence trung bình >= 0.7 (để đảm bảo chính xác)
    lines = [line for line in lines if line['confidence'] >= min_confidence]

    return {"success": True, "results": lines}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
