import pytesseract
from PIL import Image

def ocr_image_tool(image_path: str):
    """
    Extract text from an image using Tesseract OCR.
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}
