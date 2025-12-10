from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR
from PIL import Image, ImageEnhance, ImageOps
from contextlib import asynccontextmanager
import io
import numpy as np
import logging

# Tắt log rác
logging.getLogger("ppocr").setLevel(logging.ERROR)

ocr_model = None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Tiền xử lý ảnh chuyên sâu hơn cho tiếng Việt
    """
    # 1. Chuyển hệ màu
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 2. Thêm viền trắng (Padding)- Giúp model không bị mất chữ nằm sát mép ảnh
    image = ImageOps.expand(image, border=20, fill='white')

    # 3. Tăng tương phản (Contrast)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)  # Tăng mạnh hơn chút để tách chữ khỏi nền
    
    # 4. Tăng độ sắc nét (Sharpness)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)

    # 5. Resize thông minh (nếu ảnh quá nhỏ)
    width, height = image.size
    if min(width, height) < 500: # Nếu ảnh nhỏ hơn 500px chiều ngắn nhất
        scale = 2
        image = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)
    
    return np.array(image)

def group_text_into_lines(boxes, texts, scores, threshold_y=10):
    """
    Thuật toán gộp các từ riêng lẻ thành dòng dựa trên tọa độ Y.
    boxes: danh sách các numpy array tọa độ
    texts: danh sách nội dung chữ
    scores: độ tin cậy
    threshold_y: độ lệch Y cho phép để coi là cùng 1 dòng (pixel)
    """
    # Tạo danh sách các item gồm (box, text, score, center_y, min_x)
    items = []
    for box, text, score in zip(boxes, texts, scores):
        if score < 0.5: continue # Lọc tin cậy thấp
        
        # box là array 4 điểm [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Tính Y trung bình của box để xác định vị trí dòng
        center_y = np.mean(box[:, 1]) 
        min_x = np.min(box[:, 0])
        items.append({'text': text, 'center_y': center_y, 'min_x': min_x})

    # Sắp xếp tất cả theo thứ tự từ trên xuống dưới (Center Y)
    items.sort(key=lambda x: x['center_y'])

    lines = []
    if not items:
        return lines

    current_line = [items[0]]
    
    # Duyệt qua các item còn lại để gom nhóm
    for item in items[1:]:
        last_item = current_line[-1]
        
        # Nếu item này có Y gần bằng item trước đó -> Cùng 1 dòng
        if abs(item['center_y'] - last_item['center_y']) < threshold_y:
            current_line.append(item)
        else:
            # Kết thúc dòng cũ, xử lý sắp xếp trái -> phải
            current_line.sort(key=lambda x: x['min_x'])
            lines.append(" ".join([i['text'] for i in current_line]))
            
            # Bắt đầu dòng mới
            current_line = [item]
    
    # Đừng quên dòng cuối cùng
    if current_line:
        current_line.sort(key=lambda x: x['min_x'])
        lines.append(" ".join([i['text'] for i in current_line]))
        
    return lines

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_model
    import os
    import logging
    
    # Đảm bảo sử dụng CPU-only để tránh OOM
    os.environ["PADDLE_USE_GPU"] = "0"
    os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
    
    try:
        import paddle
        # Luôn sử dụng CPU để tránh OOM trên Render
        paddle.device.set_device("cpu")
        print("✓ PaddlePaddle đã được cấu hình để sử dụng CPU")
    except Exception as e:
        print(f"⚠ Warning khi cấu hình PaddlePaddle: {e}")
    
    try:
        print("Đang khởi tạo PaddleOCR model...")
        # Tinh chỉnh tham số detection để bắt chữ tiếng Việt tốt hơn
        # use_gpu=False để đảm bảo sử dụng CPU và tiết kiệm memory
        ocr_model = PaddleOCR(
            use_angle_cls=True, 
            lang="vi",
            use_gpu=False,  # CPU-only để tránh OOM
            det_db_thresh=0.3,    # Ngưỡng phát hiện (thấp hơn để bắt nét mờ)
            det_db_box_thresh=0.5,# Ngưỡng box
            det_db_unclip_ratio=1.6, # Tỉ lệ mở rộng box (giúp bắt đủ dấu tiếng Việt)
            enable_mkldnn=False,  # Tắt MKLDNN để tránh memory issues
        )
        print("✓ PaddleOCR model đã được khởi tạo thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo PaddleOCR model: {e}")
        import traceback
        traceback.print_exc()
        ocr_model = None
    
    yield
    
    # Cleanup khi shutdown
    if ocr_model is not None:
        try:
            del ocr_model
            print("✓ Model đã được cleanup")
        except:
            pass

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    API kiểm tra trạng thái hoạt động của server
    """
    import sys
    model_status = "ready" if ocr_model is not None else "not_initialized"
    
    return {
        "status": "healthy" if model_status == "ready" else "degraded",
        "model_status": model_status,
        "python_version": sys.version.split()[0]
    }

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    if ocr_model is None:
        return {"error": "Model chưa được khởi tạo."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = preprocess_image(image)
    except Exception as e:
        return {"error": f"Lỗi đọc file ảnh: {e}"}

    try:
        # Chạy OCR
        result = ocr_model.ocr(image_np)
        
        final_lines = []
        
        if result and isinstance(result, list) and len(result) > 0:
            page_data = result[0]
            
            # Xử lý Dictionary format (như log bạn cung cấp)
            if isinstance(page_data, dict):
                texts = page_data.get('rec_texts', [])
                scores = page_data.get('rec_scores', [])
                boxes = page_data.get('dt_polys', []) # Lấy tọa độ để sắp xếp
                
                if len(boxes) > 0 and len(texts) == len(boxes):
                    # Gọi hàm gộp dòng thông minh
                    final_lines = group_text_into_lines(boxes, texts, scores)
                else:
                    # Fallback nếu không có box (ít xảy ra)
                    final_lines = texts

            # Xử lý List format (Dự phòng cho phiên bản cũ/khác)
            elif isinstance(page_data, list):
                boxes = [line[0] for line in page_data]
                texts = [line[1][0] for line in page_data]
                scores = [line[1][1] for line in page_data]
                final_lines = group_text_into_lines(boxes, texts, scores)

        return {
            "status": "success",
            "count_lines": len(final_lines),
            "text": final_lines
        }

    except Exception as e:
        return {"error": f"Lỗi xử lý OCR: {str(e)}"}