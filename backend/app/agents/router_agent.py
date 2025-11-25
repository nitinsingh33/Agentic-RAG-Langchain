"""Production router agent with keyword-based intent classification."""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def classify_intent(query: str) -> str:
    """
    Classify query intent using production-ready keyword matching.
    No external ML dependencies, fast and reliable.
    """
    query_lower = query.lower()
    
    # Multi-modal visual content detection
    visual_keywords = [
        'image', 'chart', 'graph', 'visual', 'picture', 'diagram', 
        'table', 'figure', 'plot', 'visualization', 'screenshot',
        'photo', 'drawing', 'map', 'infographic', 'flowchart'
    ]
    
    # Company and business queries
    company_keywords = [
        'hero', 'vida', 'bajaj', 'tvs', 'company', 'business', 
        'revenue', 'earnings', 'financial', 'profit', 'loss', 
        'annual report', 'quarterly', 'balance sheet', 'income'
    ]
    
    # Pricing and cost queries
    pricing_keywords = [
        'price', 'cost', 'pricing', 'expensive', 'cheap', 'affordable',
        'budget', 'money', 'payment', 'finance', 'loan', 'emi'
    ]
    
    # Sales and purchase queries
    sales_keywords = [
        'sales', 'sell', 'buy', 'purchase', 'dealer', 'showroom',
        'booking', 'delivery', 'availability', 'store', 'outlet'
    ]
    
    # Check intent priority: visual > company > pricing > sales > general
    if any(keyword in query_lower for keyword in visual_keywords):
        return "multimodal"
    elif any(keyword in query_lower for keyword in company_keywords):
        return "company"
    elif any(keyword in query_lower for keyword in pricing_keywords):
        return "pricing"
    elif any(keyword in query_lower for keyword in sales_keywords):
        return "sales"
    else:
        return "general"


def get_agent_map() -> Dict[str, callable]:
    """Get production agent mapping."""
    from app.agents.company_agent import company_agent
    from app.agents.pricing_agent import pricing_agent
    from app.agents.sales_agent import sales_agent
    from app.agents.multimodal_agent import MultiModalAgent
    
    # Create multimodal agent instance
    multimodal = MultiModalAgent()
    
    return {
        "company": company_agent,
        "pricing": pricing_agent, 
        "sales": sales_agent,
        "multimodal": lambda query, chat_history=None: multimodal.process_query(query, chat_history or [])
    }


def route_query(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Route query to appropriate agent based on intent classification.
    Production-ready with proper error handling.
    """
    try:
        # Classify intent
        intent = classify_intent(query)
        logger.info(f"Query classified as: {intent}")
        
        # Get agent mapping
        agent_map = get_agent_map()
        agent_fn = agent_map.get(intent)
        
        if agent_fn:
            # Execute agent function
            result = agent_fn(query, chat_history=chat_history or [])
            
            # Ensure result has required structure
            if isinstance(result, dict):
                result["intent"] = intent
                result["agent_used"] = f"{intent.title()} Agent"
                return result
            else:
                # Handle string responses
                return {
                    "answer": str(result),
                    "intent": intent,
                    "source_documents": [],
                    "agent_used": f"{intent.title()} Agent"
                }
        else:
            # Fallback to general QA
            from app.services.stateful_bot import get_qa
            result = get_qa().invoke({"question": query, "chat_history": chat_history or []})
            
            return {
                "answer": result.get("answer", str(result)),
                "intent": "general",
                "source_documents": result.get("source_documents", []),
                "agent_used": "General QA Agent"
            }
                
    except Exception as e:
        logger.error(f"Error in query routing: {e}")
        return {
            "answer": "I encountered an error while processing your query. Please try again.",
            "intent": "error",
            "source_documents": [],
            "agent_used": "Error Handler"
        }