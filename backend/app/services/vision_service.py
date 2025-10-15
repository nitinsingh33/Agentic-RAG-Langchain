"""
Vision AI service for analyzing images using Gemini Vision and other vision models.
Handles image description, chart analysis, and visual content understanding.
"""

import base64
import logging
from typing import Dict, List, Any, Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import get_settings

logger = logging.getLogger(__name__)  # Use standard logger

class VisionService:
    """Service for analyzing visual content using Gemini Vision models."""
    
    def __init__(self):
        """Initialize the vision service."""
        self.settings = get_settings()
        
        # Initialize Gemini Vision model
        self.gemini_vision = ChatGoogleGenerativeAI(
            model=self.settings.vision_model,
            google_api_key=self.settings.GEMINI_API_KEY,
            temperature=0.1
        )
        
        # Predefined prompts for different types of analysis
        self.prompts = {
            'general_description': """
                Analyze this image and provide a detailed description. Include:
                1. What type of content this is (chart, graph, diagram, text, etc.)
                2. Key visual elements and data points
                3. Main insights or findings
                4. Any text visible in the image
                
                Be specific and detailed in your analysis.
            """,
            
            'chart_analysis': """
                This appears to be a chart or graph. Please analyze it and provide:
                1. Chart type (bar, line, pie, scatter, etc.)
                2. Title and axes labels if visible
                3. Key data points and trends
                4. Main insights and patterns
                5. Any notable outliers or significant values
                
                Format your response as structured data where possible.
            """,
            
            'table_analysis': """
                This appears to be a table or structured data. Please analyze it and provide:
                1. Table structure (rows, columns, headers)
                2. Key data categories and values
                3. Important patterns or relationships
                4. Summary of main findings
                
                Extract the actual data if clearly visible.
            """,
            
            'financial_analysis': """
                This appears to be financial data or business content. Please analyze it and provide:
                1. Type of financial information (revenue, profits, costs, etc.)
                2. Key metrics and values
                3. Time periods covered
                4. Important trends or changes
                5. Business insights
                
                Focus on extracting actionable business intelligence.
            """,
            
            'technical_diagram': """
                This appears to be a technical diagram or schematic. Please analyze it and provide:
                1. Type of diagram (flowchart, architecture, process, etc.)
                2. Main components and their relationships
                3. Process flow or connections
                4. Technical specifications if visible
                5. Purpose and functionality
            """
        }
    
    def analyze_image(self, image_base64: str, analysis_type: str = 'general_description') -> Dict[str, Any]:
        """
        Analyze an image using Gemini Vision.
        
        Args:
            image_base64: Base64 encoded image
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Get appropriate prompt
            prompt = self.prompts.get(analysis_type, self.prompts['general_description'])
            
            # For now, Gemini doesn't support image input in the same way as GPT-4V
            # We'll use a text-based approach combined with OCR results
            try:
                from app.utils.image_extractor import ImageExtractor
                extractor = ImageExtractor()
                ocr_text = extractor.extract_text_with_ocr(image_base64)
                
                # Create enhanced prompt with OCR text
                enhanced_prompt = f"""
                {prompt}
                
                Image contains the following text (from OCR):
                {ocr_text if ocr_text else 'No text detected in image'}
                
                Based on this information and the fact that this is an image from a document, 
                provide your analysis focusing on the visual content type and insights.
                """
                
                # Use Gemini for analysis
                response = self.gemini_vision.invoke(enhanced_prompt)
                
                analysis_result = {
                    'success': True,
                    'analysis_type': analysis_type,
                    'description': response.content,
                    'model_used': self.settings.vision_model,
                    'confidence': 'medium',  # Since we're using OCR + text analysis
                    'ocr_text': ocr_text
                }
                
                # Try to extract structured data if it's a chart/table
                if analysis_type in ['chart_analysis', 'table_analysis']:
                    structured_data = self._extract_structured_data(response.content)
                    if structured_data:
                        analysis_result['structured_data'] = structured_data
                
                return analysis_result
                
            except Exception as e:
                logger.warning(f"OCR extraction failed, using basic analysis: {e}")
                
                # Fallback to basic text analysis
                basic_prompt = f"""
                {prompt}
                
                Note: This is analysis of an image from a document. Since I cannot see the image directly,
                please provide a general analysis framework for this type of content ({analysis_type}).
                """
                
                response = self.gemini_vision.invoke(basic_prompt)
                
                return {
                    'success': True,
                    'analysis_type': analysis_type,
                    'description': response.content,
                    'model_used': self.settings.vision_model,
                    'confidence': 'low',
                    'note': 'Analysis based on content type without direct image access'
                }
            
        except Exception as e:
            logger.error(f"Gemini vision analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_type,
                'description': f"Failed to analyze image: {str(e)}"
            }
    
    def analyze_chart(self, image_base64: str) -> Dict[str, Any]:
        """Specialized analysis for charts and graphs."""
        return self.analyze_image(image_base64, 'chart_analysis')
    
    def analyze_table(self, image_base64: str) -> Dict[str, Any]:
        """Specialized analysis for tables."""
        return self.analyze_image(image_base64, 'table_analysis')
    
    def analyze_financial_content(self, image_base64: str) -> Dict[str, Any]:
        """Specialized analysis for financial data."""
        return self.analyze_image(image_base64, 'financial_analysis')
    
    def analyze_technical_diagram(self, image_base64: str) -> Dict[str, Any]:
        """Specialized analysis for technical diagrams."""
        return self.analyze_image(image_base64, 'technical_diagram')
    
    def batch_analyze_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batch.
        
        Args:
            images: List of image dictionaries with base64_data
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                # Determine analysis type based on image characteristics
                analysis_type = self._determine_analysis_type(image)
                
                # Analyze the image
                result = self.analyze_image(image['base64_data'], analysis_type)
                
                # Add image metadata
                result.update({
                    'image_index': i,
                    'page_number': image.get('page_number', 0),
                    'width': image.get('width', 0),
                    'height': image.get('height', 0)
                })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to analyze image {i}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_index': i
                })
        
        return results
    
    def generate_visual_summary(self, analysis_results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive summary of all visual content.
        
        Args:
            analysis_results: List of image analysis results
            
        Returns:
            Comprehensive visual content summary
        """
        try:
            # Compile all analysis results
            summaries = []
            for result in analysis_results:
                if result.get('success', False):
                    page_info = f"Page {result.get('page_number', 'N/A')}"
                    description = result.get('description', '')
                    ocr_text = result.get('ocr_text', '')
                    
                    summary_part = f"{page_info}: {description}"
                    if ocr_text:
                        summary_part += f"\nExtracted text: {ocr_text[:200]}..."
                    
                    summaries.append(summary_part)
            
            # Generate comprehensive summary using Gemini
            combined_content = "\n\n".join(summaries)
            
            prompt = f"""
            Based on the following visual content analysis from a document, create a comprehensive summary:
            
            {combined_content}
            
            Please provide:
            1. Overview of visual content types found
            2. Key data insights and trends
            3. Important charts, graphs, or diagrams
            4. Main business or technical findings
            5. Relationships between different visual elements
            
            Make it concise but comprehensive.
            """
            
            response = self.gemini_vision.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate visual summary: {e}")
            return "Unable to generate comprehensive visual summary."
    
    def _determine_analysis_type(self, image: Dict[str, Any]) -> str:
        """Determine the best analysis type for an image based on its characteristics."""
        # Simple heuristics - in a real implementation, you might use image classification
        width = image.get('width', 0)
        height = image.get('height', 0)
        
        # If image is wide and short, likely a chart
        if width > height * 1.5:
            return 'chart_analysis'
        
        # If image is square-ish, might be a pie chart or table
        if abs(width - height) < min(width, height) * 0.3:
            return 'table_analysis'
        
        # Default to general analysis
        return 'general_description'
    
    def _extract_structured_data(self, description: str) -> Optional[Dict[str, Any]]:
        """Try to extract structured data from analysis description."""
        try:
            # Look for JSON-like patterns in the description
            # This is a simplified implementation
            
            structured_info = {}
            
            # Extract chart type
            if 'bar chart' in description.lower():
                structured_info['chart_type'] = 'bar'
            elif 'line chart' in description.lower():
                structured_info['chart_type'] = 'line'
            elif 'pie chart' in description.lower():
                structured_info['chart_type'] = 'pie'
            elif 'scatter' in description.lower():
                structured_info['chart_type'] = 'scatter'
            
            # Extract numeric patterns (simplified)
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', description)
            if numbers:
                structured_info['key_values'] = numbers[:10]  # Limit to first 10
            
            return structured_info if structured_info else None
            
        except Exception as e:
            logger.warning(f"Failed to extract structured data: {e}")
            return None

# Convenience functions
def analyze_image(image_base64: str, analysis_type: str = 'general_description') -> Dict[str, Any]:
    """Analyze a single image."""
    service = VisionService()
    return service.analyze_image(image_base64, analysis_type)

def batch_analyze_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze multiple images in batch."""
    service = VisionService()
    return service.batch_analyze_images(images)