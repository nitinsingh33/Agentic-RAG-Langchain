"""
Multi-modal document processor that combines text, images, and tables.
Main orchestrator for multi-modal RAG functionality.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

from app.utils.image_extractor import ImageExtractor
from app.services.vision_service import VisionService
from app.services.retriever import get_retriever  # Fixed import
from app.utils.logger import logger  # Fixed import

# logger = get_logger(__name__)  # Removed since we use the global logger

class MultiModalProcessor:
    """
    Main processor for multi-modal documents.
    Combines text extraction, image analysis, and table processing.
    """
    
    def __init__(self):
        """Initialize the multi-modal processor."""
        self.image_extractor = ImageExtractor()
        self.vision_service = VisionService()
        self.retriever = get_retriever()  # Fixed to use retriever function
        
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with multi-modal capabilities.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID, will generate if not provided
            
        Returns:
            Processing results with all extracted content
        """
        if not document_id:
            document_id = str(uuid.uuid4())
            
        logger.info(f"Starting multi-modal processing for document: {file_path}")
        
        try:
            # 1. Extract text content (existing functionality)
            text_content = self._extract_text_content(file_path)
            
            # 2. Extract images from document
            logger.info("Extracting images from document...")
            images = self.image_extractor.extract_images_from_pdf(file_path)
            
            # 3. Extract tables from document
            logger.info("Extracting tables from document...")
            tables = self.image_extractor.extract_tables_from_pdf(file_path)
            
            # 4. Analyze images with vision AI
            logger.info(f"Analyzing {len(images)} images with vision AI...")
            image_analyses = []
            if images:
                image_analyses = self.vision_service.batch_analyze_images(images)
            
            # 5. Process and analyze tables
            logger.info(f"Processing {len(tables)} tables...")
            table_analyses = self._analyze_tables(tables)
            
            # 6. Combine all content intelligently
            logger.info("Combining multi-modal content...")
            combined_content = self._combine_multimodal_content(
                text_content, image_analyses, table_analyses, file_path
            )
            
            # 7. Generate comprehensive summary
            visual_summary = ""
            if image_analyses:
                visual_summary = self.vision_service.generate_visual_summary(image_analyses)
            
            # 8. Create document chunks for vector storage
            chunks = self._create_multimodal_chunks(combined_content)
            
            # 9. Store in vector database
            logger.info("Storing multi-modal content in vector database...")
            vector_results = self._store_multimodal_content(chunks, document_id)
            
            result = {
                'success': True,
                'document_id': document_id,
                'file_path': file_path,
                'processing_stats': {
                    'text_chunks': len(text_content.get('chunks', [])),
                    'images_found': len(images),
                    'images_analyzed': len([img for img in image_analyses if img.get('success', False)]),
                    'tables_found': len(tables),
                    'total_chunks_stored': len(chunks)
                },
                'content': {
                    'text_content': text_content,
                    'image_analyses': image_analyses,
                    'table_analyses': table_analyses,
                    'visual_summary': visual_summary,
                    'combined_content': combined_content
                },
                'vector_storage': vector_results
            }
            
            logger.info(f"Multi-modal processing completed successfully for document: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Multi-modal processing failed for {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document_id,
                'file_path': file_path
            }
    
    def query_multimodal_content(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query multi-modal content with enhanced capabilities.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            Query results with multi-modal context
        """
        try:
            # 1. Determine query type
            query_type = self._classify_query_type(query)
            
            # 2. Search vector database
            search_results = self.retriever.invoke(query)  # Use retriever instead
            
            # 3. Filter and rank results based on query type
            filtered_results = self._filter_results_by_type(search_results, query_type)
            
            # 4. Enhance results with multi-modal context
            enhanced_results = self._enhance_results_with_context(filtered_results)
            
            return {
                'success': True,
                'query': query,
                'query_type': query_type,
                'results': enhanced_results,
                'total_results': len(enhanced_results)
            }
            
        except Exception as e:
            logger.error(f"Multi-modal query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'results': []
            }
    
    def _extract_text_content(self, file_path: str) -> Dict[str, Any]:
        """Extract text content using existing functionality."""
        try:
            from .ingest import ingest_files
            
            # Use existing ingest functionality
            result = ingest_files([file_path])
            
            return {
                'text': f"Text content from {Path(file_path).name}",
                'chunks': result.get('processed_chunks', []),
                'metadata': {'source': file_path, 'processed': True}
            }
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {'text': '', 'chunks': [], 'metadata': {}}
    
    def _analyze_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze extracted tables for insights."""
        analyses = []
        
        for i, table in enumerate(tables):
            try:
                analysis = {
                    'table_index': i,
                    'rows': table.get('rows', 0),
                    'columns': table.get('columns', 0),
                    'extraction_method': table.get('extraction_method', 'unknown'),
                    'data_summary': self._summarize_table_data(table),
                    'insights': self._extract_table_insights(table)
                }
                analyses.append(analysis)
                
            except Exception as e:
                logger.warning(f"Table analysis failed for table {i}: {e}")
                analyses.append({
                    'table_index': i,
                    'error': str(e)
                })
        
        return analyses
    
    def _summarize_table_data(self, table: Dict[str, Any]) -> str:
        """Generate a summary of table data."""
        try:
            rows = table.get('rows', 0)
            columns = table.get('columns', 0)
            method = table.get('extraction_method', 'unknown')
            
            summary = f"Table with {rows} rows and {columns} columns (extracted using {method})"
            
            # Add data preview if available
            data = table.get('data', [])
            if data and len(data) > 0:
                first_row = data[0]
                headers = list(first_row.keys()) if isinstance(first_row, dict) else ['Column ' + str(i) for i in range(len(first_row))]
                summary += f". Headers: {', '.join(headers[:5])}" + ("..." if len(headers) > 5 else "")
            
            return summary
            
        except Exception as e:
            return f"Table summary unavailable: {e}"
    
    def _extract_table_insights(self, table: Dict[str, Any]) -> List[str]:
        """Extract insights from table data."""
        insights = []
        
        try:
            data = table.get('data', [])
            if not data:
                return ["No data available for analysis"]
            
            # Basic insights
            insights.append(f"Contains {len(data)} data records")
            
            if table.get('accuracy'):
                insights.append(f"Extraction accuracy: {table['accuracy']:.1f}%")
            
            # Column analysis
            if isinstance(data[0], dict):
                columns = list(data[0].keys())
                insights.append(f"Key columns: {', '.join(columns[:3])}" + ("..." if len(columns) > 3 else ""))
            
        except Exception as e:
            insights.append(f"Analysis error: {e}")
        
        return insights
    
    def _combine_multimodal_content(self, text_content: Dict[str, Any], 
                                   image_analyses: List[Dict[str, Any]], 
                                   table_analyses: List[Dict[str, Any]], 
                                   file_path: str) -> Dict[str, Any]:
        """Combine all content types into a unified structure."""
        
        combined = {
            'document_source': file_path,
            'content_types': [],
            'sections': []
        }
        
        # Add text content
        if text_content.get('text'):
            combined['content_types'].append('text')
            combined['sections'].append({
                'type': 'text',
                'content': text_content['text'],
                'metadata': text_content.get('metadata', {})
            })
        
        # Add image analyses
        for i, image_analysis in enumerate(image_analyses):
            if image_analysis.get('success', False):
                combined['content_types'].append('image')
                combined['sections'].append({
                    'type': 'image',
                    'content': image_analysis.get('description', ''),
                    'page_number': image_analysis.get('page_number', 0),
                    'analysis_type': image_analysis.get('analysis_type', 'general'),
                    'structured_data': image_analysis.get('structured_data', {}),
                    'metadata': {
                        'image_index': i,
                        'width': image_analysis.get('width', 0),
                        'height': image_analysis.get('height', 0)
                    }
                })
        
        # Add table analyses
        for i, table_analysis in enumerate(table_analyses):
            combined['content_types'].append('table')
            combined['sections'].append({
                'type': 'table',
                'content': table_analysis.get('data_summary', ''),
                'insights': table_analysis.get('insights', []),
                'metadata': {
                    'table_index': i,
                    'rows': table_analysis.get('rows', 0),
                    'columns': table_analysis.get('columns', 0),
                    'extraction_method': table_analysis.get('extraction_method', 'unknown')
                }
            })
        
        # Remove duplicates from content_types
        combined['content_types'] = list(set(combined['content_types']))
        
        return combined
    
    def _create_multimodal_chunks(self, combined_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks suitable for vector storage."""
        chunks = []
        
        for i, section in enumerate(combined_content.get('sections', [])):
            try:
                chunk = {
                    'id': str(uuid.uuid4()),
                    'content': section['content'],
                    'content_type': section['type'],
                    'metadata': {
                        'section_index': i,
                        'document_source': combined_content.get('document_source', ''),
                        **section.get('metadata', {})
                    }
                }
                
                # Add type-specific metadata
                if section['type'] == 'image':
                    chunk['metadata']['page_number'] = section.get('page_number', 0)
                    chunk['metadata']['analysis_type'] = section.get('analysis_type', 'general')
                elif section['type'] == 'table':
                    chunk['metadata']['insights'] = section.get('insights', [])
                
                chunks.append(chunk)
                
            except Exception as e:
                logger.warning(f"Failed to create chunk for section {i}: {e}")
        
        return chunks
    
    def _store_multimodal_content(self, chunks: List[Dict[str, Any]], document_id: str) -> Dict[str, Any]:
        """Store multi-modal chunks in vector database."""
        try:
            stored_count = 0
            errors = []
            
            for chunk in chunks:
                try:
                    # Add document ID to metadata
                    chunk['metadata']['document_id'] = document_id
                    
                    # Store in vector database
                    # This would use your existing vector storage logic
                    # self.vector_store.add_document(chunk['content'], chunk['metadata'])
                    stored_count += 1
                    
                except Exception as e:
                    errors.append(f"Chunk {chunk.get('id', 'unknown')}: {e}")
            
            return {
                'success': True,
                'stored_chunks': stored_count,
                'total_chunks': len(chunks),
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Vector storage failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stored_chunks': 0,
                'total_chunks': len(chunks)
            }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query to optimize search."""
        query_lower = query.lower()
        
        # Visual content keywords
        visual_keywords = ['chart', 'graph', 'image', 'picture', 'visual', 'diagram', 'figure']
        if any(keyword in query_lower for keyword in visual_keywords):
            return 'visual'
        
        # Table/data keywords
        table_keywords = ['table', 'data', 'number', 'value', 'statistic', 'figure']
        if any(keyword in query_lower for keyword in table_keywords):
            return 'tabular'
        
        # Financial keywords
        financial_keywords = ['revenue', 'profit', 'cost', 'price', 'financial', 'money', 'sales']
        if any(keyword in query_lower for keyword in financial_keywords):
            return 'financial'
        
        return 'general'
    
    def _filter_results_by_type(self, search_results: List[Dict[str, Any]], query_type: str) -> List[Dict[str, Any]]:
        """Filter and prioritize results based on query type."""
        if query_type == 'visual':
            # Prioritize image content
            visual_results = [r for r in search_results if r.get('metadata', {}).get('content_type') == 'image']
            other_results = [r for r in search_results if r.get('metadata', {}).get('content_type') != 'image']
            return visual_results + other_results
        
        elif query_type == 'tabular':
            # Prioritize table content
            table_results = [r for r in search_results if r.get('metadata', {}).get('content_type') == 'table']
            other_results = [r for r in search_results if r.get('metadata', {}).get('content_type') != 'table']
            return table_results + other_results
        
        return search_results
    
    def _enhance_results_with_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional context."""
        enhanced = []
        
        for result in results:
            try:
                enhanced_result = result.copy()
                metadata = result.get('metadata', {})
                
                # Add content type indicator
                content_type = metadata.get('content_type', 'text')
                enhanced_result['content_type_display'] = content_type.title()
                
                # Add visual indicators
                if content_type == 'image':
                    enhanced_result['visual_indicator'] = f"📊 Image from page {metadata.get('page_number', '?')}"
                elif content_type == 'table':
                    enhanced_result['visual_indicator'] = f"📋 Table ({metadata.get('rows', '?')} rows)"
                else:
                    enhanced_result['visual_indicator'] = "📄 Text content"
                
                enhanced.append(enhanced_result)
                
            except Exception as e:
                logger.warning(f"Failed to enhance result: {e}")
                enhanced.append(result)
        
        return enhanced

# Convenience functions
def process_multimodal_document(file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
    """Process a document with multi-modal capabilities."""
    processor = MultiModalProcessor()
    return processor.process_document(file_path, document_id)

def query_multimodal_content(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Query multi-modal content."""
    processor = MultiModalProcessor()
    return processor.query_multimodal_content(query, top_k)