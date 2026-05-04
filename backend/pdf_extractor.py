import fitz  # PyMuPDF
import os
import io
from PIL import Image

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extracts images from a PDF and saves them to output_dir.
    Returns a list of absolute image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Simple heuristic: ignore very small images (like icons/logos)
            try:
                img_obj = Image.open(io.BytesIO(image_bytes))
                if img_obj.width < 150 or img_obj.height < 150:
                    continue
            except Exception as e:
                print(f"Error checking image size: {e}")
                continue

            image_name = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
                
            image_paths.append(image_path)
            
    return image_paths
