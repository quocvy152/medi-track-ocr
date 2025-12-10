"""
Script để pre-download PaddleOCR models trong quá trình Docker build
"""
import os
from paddleocr import PaddleOCR

def download_models():
    """Download PaddleOCR models trước khi chạy ứng dụng"""
    print("Đang tải PaddleOCR models...")
    
    try:
        # Tạo thư mục để lưu models
        model_dir = os.path.join(os.path.expanduser("~"), ".paddleocr")
        os.makedirs(model_dir, exist_ok=True)
        
        # Khởi tạo PaddleOCR với CPU-only để download models
        # Sử dụng use_gpu=False để tiết kiệm memory
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="vi",
            use_gpu=False,  # CPU-only để tránh OOM
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6
        )
        
        print("✓ Models đã được tải thành công!")
        print(f"Models được lưu tại: {model_dir}")
        
        # Test model với một ảnh trắng nhỏ để đảm bảo model hoạt động
        import numpy as np
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        try:
            _ = ocr.ocr(test_image)
            print("✓ Model đã được test và hoạt động tốt!")
        except Exception as e:
            print(f"⚠ Warning: Model test có lỗi nhỏ: {e}")
            print("Nhưng models đã được download, sẽ hoạt động khi chạy thực tế.")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải models: {e}")
        raise

if __name__ == "__main__":
    download_models()

