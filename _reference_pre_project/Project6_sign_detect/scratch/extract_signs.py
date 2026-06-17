import os
import fitz  # PyMuPDF
from PIL import Image

def main():
    pdf_path = "docs/路標.pdf"
    output_dir = "docs"
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    print(f"Number of pages: {len(doc)}")
    
    page = doc[0]
    rect = page.rect
    print(f"Page dimensions: {rect.width}x{rect.height}")
    
    # Render page to a high-res image
    zoom = 4.0  # 4x zoom for high quality
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img_path = os.path.join(output_dir, "temp_page.png")
    pix.save(img_path)
    print(f"Saved temporary full page image to {img_path}")
    
    # Open with PIL to crop
    img = Image.open(img_path)
    w, h = img.size
    print(f"Rendered image size: {w}x{h}")
    
    # Let's crop the 4 quadrants
    # Based on the screenshot:
    # Top-left: Stop sign (stop)
    # Top-right: Railway crossing (rail)
    # Bottom-left: Pedestrian (pedestrian)
    # Bottom-right: Blocked (blocked)
    #
    # The layout seems to be a grid of cells. Let's calculate the bounding boxes
    # from the screenshot structure:
    # Table has borders and spacing. Let's define the 4 crop areas.
    # The vertical middle division is at w / 2.
    # The horizontal divisions:
    # There are empty cells or spacing in between.
    # From screenshot, it looks like a table with:
    # Row 1 (Stop, Rail)
    # Row 2 (Empty spacers?)
    # Row 3 (Pedestrian, Blocked)
    # Row 4 (Empty spacers?)
    # Let's inspect the height divisions:
    # Row 1 height: from 0 to approx 0.35 * h
    # Row 2 (empty): from 0.35 * h to 0.5 * h
    # Row 3 height: from 0.5 * h to 0.85 * h
    # Row 4 (empty): from 0.85 * h to h
    
    mid_x = w // 2
    
    # Crop regions: (left, top, right, bottom)
    # Let's define box coordinates proportionally:
    # Stop: top-left quadrant, but we want to crop just the sign itself (the octagonal area).
    # Since the signs are centered in their respective cells, let's find the content.
    # Stop: cell is left: 0 to mid_x, top: 0 to approx h * 0.35.
    # Let's crop them based on the actual pixels.
    # We can write a quick auto-crop helper to remove white margins!
    
    def crop_and_autocrop(box, name):
        cropped = img.crop(box)
        # Convert to bounding box of non-white pixels
        # A simple way: find bounding box where pixel color != white (255, 255, 255)
        # Let's convert to grayscale and invert to find bounding box
        bg = Image.new(cropped.mode, cropped.size, (255, 255, 255))
        from PIL import ImageChops
        diff = ImageChops.difference(cropped, bg)
        bbox = diff.getbbox()
        if bbox:
            # Add a small padding of 15px
            pad = 20
            left = max(0, bbox[0] - pad)
            top = max(0, bbox[1] - pad)
            right = min(cropped.width, bbox[2] + pad)
            bottom = min(cropped.height, bbox[3] + pad)
            cropped = cropped.crop((left, top, right, bottom))
        
        save_path = os.path.join(output_dir, f"{name}.png")
        cropped.save(save_path)
        print(f"Saved cropped sign {name} to {save_path} (size: {cropped.size})")

    # Stop: left = 0 to mid_x, top = 0 to h * 0.35
    crop_and_autocrop((0, 0, mid_x, int(h * 0.38)), "stop")
    
    # Rail: left = mid_x to w, top = 0 to h * 0.38
    crop_and_autocrop((mid_x, 0, w, int(h * 0.38)), "rail")
    
    # Pedestrian: left = 0 to mid_x, top = int(h * 0.50) to int(h * 0.85)
    crop_and_autocrop((0, int(h * 0.50), mid_x, int(h * 0.85)), "pedestrian")
    
    # Blocked: left = mid_x to w, top = int(h * 0.50) to int(h * 0.85)
    crop_and_autocrop((mid_x, int(h * 0.50), w, int(h * 0.85)), "blocked")
    
    # Clean up temp page
    try:
        os.remove(img_path)
        print("Removed temp full page image.")
    except OSError:
        pass

if __name__ == "__main__":
    main()
