import fitz
import os

#extract text from pdf using PyMuPDF
def extract_txt_from_pdf(pdf_path):
    txt = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt += page.get_text("text")
    return txt

#extract images from pdf using PyMuPDF
def extract_images_from_pdf(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_path = f"{output_dir}/page{page_number}_img{img_index}.{image_ext}"
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)
    return saved_images

