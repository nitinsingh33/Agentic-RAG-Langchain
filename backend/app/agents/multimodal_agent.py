"""Production multi-modal agent for visual content processing."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MultiModalAgent:
    """Production multi-modal agent for images, charts, and tables."""
    
    def __init__(self):
        try:
            from app.services.multimodal_processor import MultiModalProcessor
            from app.services.vision_service import VisionService
            self.multimodal_processor = MultiModalProcessor()
            self.vision_service = VisionService()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing MultiModalAgent: {e}")
            self.multimodal_processor = None
            self.vision_service = None
            self.initialized = False
    
    def process_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process multi-modal query with visual content analysis."""
        try:
            if not self.initialized:
                return {
                    "answer": "Multi-modal processing is currently unavailable. Please try text-only queries.",
                    "intent": "multimodal",
                    "source_documents": [],
                    "agent_used": "MultiModal Agent (Fallback)"
                }
            
            # Analyze query type
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ['chart', 'graph', 'plot']):
                return self._analyze_charts(query)
            elif any(keyword in query_lower for keyword in ['table', 'data']):
                return self._analyze_tables(query)
            elif any(keyword in query_lower for keyword in ['image', 'picture', 'photo']):
                return self._analyze_images(query)
            else:
                return self._general_visual_analysis(query)
                
        except Exception as e:
            logger.error(f"Error in multimodal processing: {e}")
            return {
                "answer": f"Error processing visual query: {str(e)}",
                "intent": "multimodal",
                "source_documents": [],
                "agent_used": "MultiModal Agent (Error)"
            }
    
    def _analyze_charts(self, query: str) -> Dict[str, Any]:
        """Analyze chart-related queries."""
        try:
            if self.multimodal_processor:
                results = self.multimodal_processor.query_multimodal_content(query, content_type="charts")
                return {
                    "answer": results.get("response", "Chart analysis completed."),
                    "intent": "multimodal",
                    "source_documents": results.get("sources", []),
                    "agent_used": "Chart Analysis Agent"
                }
        except Exception as e:
            logger.error(f"Chart analysis error: {e}")
        
        return {
            "answer": "I can help analyze charts and graphs from your documents. Please ensure your PDFs contain visual charts for analysis.",
            "intent": "multimodal",
            "source_documents": [],
            "agent_used": "Chart Analysis Agent"
        }
    
    def _analyze_tables(self, query: str) -> Dict[str, Any]:
        """Analyze table-related queries."""
        try:
            if self.multimodal_processor:
                results = self.multimodal_processor.query_multimodal_content(query, content_type="tables")
                return {
                    "answer": results.get("response", "Table analysis completed."),
                    "intent": "multimodal",
                    "source_documents": results.get("sources", []),
                    "agent_used": "Table Analysis Agent"
                }
        except Exception as e:
            logger.error(f"Table analysis error: {e}")
        
        return {
            "answer": "I can help analyze tables and structured data from your documents. Please upload PDFs containing tables for analysis.",
            "intent": "multimodal",
            "source_documents": [],
            "agent_used": "Table Analysis Agent"
        }
    
    def _analyze_images(self, query: str) -> Dict[str, Any]:
        """Analyze image-related queries."""
        try:
            if self.vision_service:
                # Use vision service for image analysis
                analysis = self.vision_service.analyze_image(query, analysis_type="general")
                return {
                    "answer": analysis.get("description", "Image analysis completed."),
                    "intent": "multimodal",
                    "source_documents": analysis.get("sources", []),
                    "agent_used": "Image Analysis Agent"
                }
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
        
        return {
            "answer": "I can analyze images, diagrams, and visual content from your documents. Please upload PDFs with images for analysis.",
            "intent": "multimodal",
            "source_documents": [],
            "agent_used": "Image Analysis Agent"
        }
    
    def _general_visual_analysis(self, query: str) -> Dict[str, Any]:
        """General visual content analysis."""
        return {
            "answer": "I can help analyze visual content including charts, tables, images, and diagrams from your documents. Please specify what type of visual analysis you need or upload relevant documents.",
            "intent": "multimodal",
            "source_documents": [],
            "agent_used": "General Visual Agent"
        }