"""
Image and table extraction utilities for multi-modal document processing.
Handles PDF image extraction, OCR, and table detection.
"""

import io
import base64
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
import tabula
import camelot

logger = logging.getLogger(__name__)  # Use standard logger

class ImageExtractor:
    """Extracts images, tables, and visual content from PDF documents."""
    
    def __init__(self):
        """Initialize the image extractor."""
        self.min_image_size = (100, 100)  # Minimum image size to consider
        self.image_formats = ['png', 'jpg', 'jpeg']
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get images on this page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip if image is too small or not RGB/GRAY
                        if pix.width < self.min_image_size[0] or pix.height < self.min_image_size[1]:
                            pix = None
                            continue
                            
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Convert to base64 for API calls
                            buffered = io.BytesIO()
                            pil_image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            
                            images.append({
                                'page_number': page_num + 1,
                                'image_index': img_index,
                                'width': pix.width,
                                'height': pix.height,
                                'format': 'png',
                                'base64_data': img_base64,
                                'size_bytes': len(img_data),
                                'bbox': self._get_image_bbox(page, xref)
                            })
                            
                        pix = None  # Release memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from {pdf_path}")
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from PDF {pdf_path}: {e}")
            return []
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using both tabula and camelot.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing table data and metadata
        """
        tables = []
        
        try:
            # Try tabula first (works well with text-based tables)
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(tabula_tables):
                if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
                    tables.append({
                        'table_index': len(tables),
                        'extraction_method': 'tabula',
                        'rows': df.shape[0],
                        'columns': df.shape[1],
                        'data': df.to_dict('records'),
                        'csv_string': df.to_csv(index=False),
                        'html_string': df.to_html(index=False)
                    })
        except Exception as e:
            logger.warning(f"Tabula extraction failed for {pdf_path}: {e}")
        
        try:
            # Try camelot (works well with image-based tables)
            camelot_tables = camelot.read_pdf(pdf_path, pages='all')
            
            for table in camelot_tables:
                df = table.df
                if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
                    tables.append({
                        'table_index': len(tables),
                        'extraction_method': 'camelot',
                        'rows': df.shape[0],
                        'columns': df.shape[1],
                        'data': df.to_dict('records'),
                        'csv_string': df.to_csv(index=False),
                        'html_string': df.to_html(index=False),
                        'accuracy': table.accuracy,
                        'whitespace': table.whitespace
                    })
        except Exception as e:
            logger.warning(f"Camelot extraction failed for {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(tables)} tables from {pdf_path}")
        return tables
    
    def extract_text_with_ocr(self, image_base64: str) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Extracted text
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _get_image_bbox(self, page, xref: int) -> Optional[Dict[str, float]]:
        """Get bounding box of image on page."""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd need to find the image placement
            rect = page.rect
            return {
                'x0': 0,
                'y0': 0,
                'x1': rect.width,
                'y1': rect.height
            }
        except:
            return None
    
    def detect_chart_type(self, image_base64: str) -> str:
        """
        Detect the type of chart/graph in an image.
        This is a simplified implementation - you could use ML models for better accuracy.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Detected chart type
        """
        try:
            # Decode and analyze image
            image_data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_data))
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Simple heuristics for chart detection
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count lines and contours
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Simple classification
            if lines is not None and len(lines) > 10:
                return "line_chart"
            elif len(contours) > 5:
                # Check for circular shapes (pie charts)
                circular_contours = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.7:
                            circular_contours += 1
                
                if circular_contours > 0:
                    return "pie_chart"
                else:
                    return "bar_chart"
            
            return "unknown_chart"
            
        except Exception as e:
            logger.error(f"Chart type detection failed: {e}")
            return "unknown_chart"

# Convenience functions
def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract images from PDF file."""
    extractor = ImageExtractor()
    return extractor.extract_images_from_pdf(pdf_path)

def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables from PDF file."""
    extractor = ImageExtractor()
    return extractor.extract_tables_from_pdf(pdf_path)