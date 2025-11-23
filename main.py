from fastapi import FastAPI, UploadFile, File
import easyocr
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
reader = easyocr.Reader(['en', 'vi'])

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

    texts = []
    for bbox, text, confidence in result:
        # Convert numpy arrays to lists and numpy scalars to Python types
        bbox_list = [[float(coord) for coord in point] for point in bbox]
        texts.append({
            "text": text,
            "confidence": float(confidence),
            "bbox": bbox_list
        })

    return {"success": True, "results": texts}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
