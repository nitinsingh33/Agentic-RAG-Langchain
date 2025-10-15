"""Production router agent with keyword-based intent classification.""""""Production router agent with keyword-based intent classification.""""""Production router agent with keyword-based intent classification."""# router_agent.py

import logging

from typing import Dict, List, Anyimport logging



logger = logging.getLogger(__name__)from typing import Dict, List, Any, Optionalimport loggingimport os



def classify_intent(query: str) -> str:

    """Classify query intent using keyword matching."""

    query_lower = query.lower()logger = logging.getLogger(__name__)from typing import Dict, List, Any, Optionalfrom dotenv import load_dotenv

    

    # Visual content detection

    visual_keywords = ['image', 'chart', 'graph', 'visual', 'picture', 'diagram', 'table', 'figure', 'plot']

    def classify_intent(query: str) -> str:from sentence_transformers import SentenceTransformer, util

    # Company queries

    company_keywords = ['hero', 'vida', 'bajaj', 'tvs', 'company', 'business', 'revenue', 'earnings', 'financial']    """

    

    # Pricing queries      Classify query intent using production-ready keyword matching.logger = logging.getLogger(__name__)

    pricing_keywords = ['price', 'cost', 'pricing', 'expensive', 'cheap', 'affordable', 'budget', 'money']

        No external ML dependencies, fast and reliable.

    # Sales queries

    sales_keywords = ['sales', 'sell', 'buy', 'purchase', 'dealer', 'showroom', 'booking', 'delivery']    """# --- Load agents ---

    

    if any(keyword in query_lower for keyword in visual_keywords):    query_lower = query.lower()

        return "multimodal"

    elif any(keyword in query_lower for keyword in company_keywords):    def classify_intent(query: str) -> str:from .sales_agent import run_sales_agent

        return "company"

    elif any(keyword in query_lower for keyword in pricing_keywords):    # Multi-modal visual content detection

        return "pricing"

    elif any(keyword in query_lower for keyword in sales_keywords):    visual_keywords = [    """from .pricing_agent import run_pricing_agent

        return "sales"

    else:        'image', 'chart', 'graph', 'visual', 'picture', 'diagram', 

        return "general"

        'table', 'figure', 'plot', 'visualization', 'screenshot',    Classify query intent using production-ready keyword matching.from .company_agent import run_company_agent

def route_query(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:

    """Route query to appropriate agent."""        'photo', 'drawing', 'map', 'infographic', 'flowchart'

    try:

        intent = classify_intent(query)    ]    No external ML dependencies, fast and reliable.from .multimodal_agent import create_multimodal_agent

        logger.info(f"Query classified as: {intent}")

            

        if intent == "multimodal":

            from app.agents.multimodal_agent import MultiModalAgent    # Company and business queries    """

            agent = MultiModalAgent()

            return agent.process_query(query, chat_history or [])    company_keywords = [

        

        elif intent == "company":        'hero', 'vida', 'bajaj', 'tvs', 'company', 'business',     query_lower = query.lower()load_dotenv()

            from app.agents.company_agent import company_agent

            return company_agent(query, chat_history or [])        'revenue', 'earnings', 'financial', 'profit', 'loss', 

        

        elif intent == "pricing":        'annual report', 'quarterly', 'balance sheet', 'income'    

            from app.agents.pricing_agent import pricing_agent

            return pricing_agent(query, chat_history or [])    ]

        

        elif intent == "sales":        # Multi-modal visual content detection# ---------------- CONFIG ---------------- #

            from app.agents.sales_agent import sales_agent

            return sales_agent(query, chat_history or [])    # Pricing and cost queries

        

        else:    pricing_keywords = [    visual_keywords = [INTENT_THRESHOLD = float(os.getenv("INTENT_THRESHOLD", 0.4))

            from app.services.stateful_bot import get_qa

            result = get_qa().invoke({"question": query, "chat_history": chat_history or []})        'price', 'cost', 'pricing', 'expensive', 'cheap', 'affordable',

            return {

                "answer": result.get("answer", str(result)),        'budget', 'money', 'payment', 'finance', 'loan', 'emi'        'image', 'chart', 'graph', 'visual', 'picture', 'diagram', 

                "intent": "general",

                "source_documents": result.get("source_documents", []),    ]

                "agent_used": "General QA Agent"

            }            'table', 'figure', 'plot', 'visualization', 'screenshot',# Semantic intent keywords (can expand easily)

            

    except Exception as e:    # Sales and purchase queries

        logger.error(f"Error in query routing: {e}")

        return {    sales_keywords = [        'photo', 'drawing', 'map', 'infographic', 'flowchart'INTENTS = {

            "answer": "I encountered an error while processing your query. Please try again.",

            "intent": "error",        'sales', 'sell', 'buy', 'purchase', 'dealer', 'showroom',

            "source_documents": [],

            "agent_used": "Error Handler"        'booking', 'delivery', 'availability', 'store', 'outlet'    ]    "sales": [

        }
    ]

                "sales", "volume", "revenue", "market share", "region", "state", "OEM", "model performance"

    # Check intent priority: visual > company > pricing > sales > general

    if any(keyword in query_lower for keyword in visual_keywords):    # Company and business queries    ],

        return "multimodal"

    elif any(keyword in query_lower for keyword in company_keywords):    company_keywords = [    "pricing": [

        return "company"

    elif any(keyword in query_lower for keyword in pricing_keywords):        'hero', 'vida', 'bajaj', 'tvs', 'company', 'business',         "price", "pricing", "cost", "discount", "ex-showroom", "on-road price", "price strategy"

        return "pricing"

    elif any(keyword in query_lower for keyword in sales_keywords):        'revenue', 'earnings', 'financial', 'profit', 'loss',     ],

        return "sales"

    else:        'annual report', 'quarterly', 'balance sheet', 'income'    "company": [

        return "general"

    ]        "hero", "tvs", "ola", "bajaj", "company profile", "market share", "partnerships", "product lines", "ev", "ice"

def get_agent_map() -> Dict[str, callable]:

    """Get production agent mapping."""        ],

    from app.agents.company_agent import company_agent

    from app.agents.pricing_agent import pricing_agent    # Pricing and cost queries    "multimodal": [

    from app.agents.sales_agent import sales_agent

    from app.agents.multimodal_agent import MultiModalAgent    pricing_keywords = [        "chart", "graph", "image", "table", "visual", "diagram", "figure", "data visualization", 

    

    # Create multimodal agent instance        'price', 'cost', 'pricing', 'expensive', 'cheap', 'affordable',        "show me", "what does this chart", "analyze the graph", "table data", "financial chart"

    multimodal = MultiModalAgent()

            'budget', 'money', 'payment', 'finance', 'loan', 'emi'    ]

    return {

        "company": company_agent,    ]}

        "pricing": pricing_agent, 

        "sales": sales_agent,    

        "multimodal": lambda query, chat_history=None: multimodal.process_query(query, chat_history or [])

    }    # Sales and purchase queries# Optional system messages per agents (future-proof)



def route_query(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:    sales_keywords = [SYSTEM_MESSAGES = {

    """

    Route query to appropriate agent based on intent classification.        'sales', 'sell', 'buy', 'purchase', 'dealer', 'showroom',    "sales": "You are a Sales Expert AI. Answer questions about sales, regions, volume, market share...",

    Production-ready with proper error handling.

    """        'booking', 'delivery', 'availability', 'store', 'outlet'    "pricing": "You are a Pricing Expert AI. Answer questions about pricing, discounts, and costs...",

    try:

        # Classify intent    ]    "company": "You are a Company Intelligence AI. Answer questions about companies, products, market positions...",

        intent = classify_intent(query)

        logger.info(f"Query classified as: {intent}")        "multimodal": "You are a Multi-Modal AI. Analyze visual content including charts, graphs, images, tables, and diagrams...",

        

        # Get agent mapping    # Check intent priority: visual > company > pricing > sales > general    "general": "You are a General AI. Answer any questions using the relevant documents..."

        agent_map = get_agent_map()

        agent_fn = agent_map.get(intent)    if any(keyword in query_lower for keyword in visual_keywords):}

        

        if agent_fn:        return "multimodal"

            # Execute agent function

            result = agent_fn(query, chat_history=chat_history or [])    elif any(keyword in query_lower for keyword in company_keywords):# Map intent → agent function (defined after the functions)

            

            # Ensure result has required structure        return "company"def get_agent_map():

            if isinstance(result, dict):

                result["intent"] = intent    elif any(keyword in query_lower for keyword in pricing_keywords):    return {

                result["agent_used"] = f"{intent.title()} Agent"

                return result        return "pricing"        "sales": run_sales_agent,

            else:

                # Handle string responses    elif any(keyword in query_lower for keyword in sales_keywords):        "pricing": run_pricing_agent,

                return {

                    "answer": str(result),        return "sales"        "company": run_company_agent,

                    "intent": intent,

                    "source_documents": [],    else:        "multimodal": run_multimodal_agent

                    "agent_used": f"{intent.title()} Agent"

                }        return "general"    }

        else:

            # Fallback to general QA

            from app.services.stateful_bot import get_qa

            result = get_qa().invoke({"question": query, "chat_history": chat_history or []})def get_agent_map() -> Dict[str, callable]:# Sentence Transformer for semantic intent detection (Lazy Loading)

            

            return {    """Get production agent mapping."""intent_model = None

                "answer": result.get("answer", str(result)),

                "intent": "general",    from app.agents.company_agent import company_agentintent_embeddings = None

                "source_documents": result.get("source_documents", []),

                "agent_used": "General QA Agent"    from app.agents.pricing_agent import pricing_agentmultimodal_agent_instance = None

            }

                from app.agents.sales_agent import sales_agent

    except Exception as e:

        logger.error(f"Error in query routing: {e}")    from app.agents.multimodal_agent import MultiModalAgentdef get_multimodal_agent():

        return {

            "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",        """Lazy loading for multimodal agent"""

            "intent": "error",

            "source_documents": [],    # Create multimodal agent instance    global multimodal_agent_instance

            "agent_used": "Error Handler"

        }    multimodal = MultiModalAgent()    if multimodal_agent_instance is None:

            multimodal_agent_instance = create_multimodal_agent()

    return {    return multimodal_agent_instance

        "company": company_agent,

        "pricing": pricing_agent, def run_multimodal_agent(query: str, chat_history=None):

        "sales": sales_agent,    """Run the multi-modal agent for visual content queries."""

        "multimodal": lambda query, chat_history=None: multimodal.process_query(query, chat_history or [])    try:

    }        agent = get_multimodal_agent()

        result = agent.process_query(query, context={'chat_history': chat_history})

def route_query(query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:        

    """        # Format response to match other agents

    Route query to appropriate agent based on intent classification.        return {

    Production-ready with proper error handling.            "answer": result.get('response', 'No response from multi-modal agent'),

    """            "source_documents": [],  # Multi-modal agent handles its own sources

    try:            "agent_type": "multimodal",

        # Classify intent            "intermediate_steps": result.get('intermediate_steps', [])

        intent = classify_intent(query)        }

        logger.info(f"Query classified as: {intent}")    except Exception as e:

                return {

        # Get agent mapping            "answer": f"⚠️ Multi-modal analysis error: {e}",

        agent_map = get_agent_map()            "source_documents": [],

        agent_fn = agent_map.get(intent)            "agent_type": "multimodal"

                }

        if agent_fn:

            # Execute agent functiondef get_intent_model():

            result = agent_fn(query, chat_history=chat_history or [])    global intent_model, intent_embeddings

                if intent_model is None:

            # Ensure result has required structure        intent_model = SentenceTransformer("all-MiniLM-L6-v2")

            if isinstance(result, dict):        intent_embeddings = {k: intent_model.encode(" ".join(v), convert_to_tensor=True) for k, v in INTENTS.items()}

                result["intent"] = intent    return intent_model, intent_embeddings

                result["agent_used"] = f"{intent.title()} Agent"

                return result# ---------------- ROUTER LOGIC ---------------- #

            else:def detect_intent(query: str) -> str:

                # Handle string responses    """Detect intent using embeddings + semantic similarity"""

                return {    model, embeddings = get_intent_model()

                    "answer": str(result),    query_emb = model.encode(query, convert_to_tensor=True)

                    "intent": intent,    scores = {intent: util.cos_sim(query_emb, emb).item() for intent, emb in embeddings.items()}

                    "source_documents": [],    top_intent = max(scores, key=scores.get)

                    "agent_used": f"{intent.title()} Agent"

                }    # Debug log for monitoring

        else:    print(f"[Intent Detection] Query: {query}")

            # Fallback to general QA    print(f"[Intent Scores] {scores}")

            from app.services.stateful_bot import get_qa    

            result = get_qa().invoke({"question": query, "chat_history": chat_history or []})    if scores[top_intent] < INTENT_THRESHOLD:

                    return "general"

            return {    return top_intent

                "answer": result.get("answer", str(result)),

                "intent": "general",def route_query(query: str, chat_history=None):

                "source_documents": result.get("source_documents", []),    """

                "agent_used": "General QA Agent"    Route query intelligently to specific agent or fallback RAG.

            }    Handles multi-turn context and semantic intent detection.

                """

    except Exception as e:    chat_history = chat_history or []

        logger.error(f"Error in query routing: {e}")    try:

        return {        intent = detect_intent(query)

            "answer": f"I encountered an error while processing your query. Please try again or rephrase your question.",        agent_map = get_agent_map()

            "intent": "error",        agent_fn = agent_map.get(intent)

            "source_documents": [],

            "agent_used": "Error Handler"        if agent_fn:

        }            return agent_fn(query, chat_history=chat_history)
        else:
            from app.services.stateful_bot import get_qa
            return get_qa().invoke({"question": query, "chat_history": chat_history})

    except Exception as e:
        return {
            "answer": f"⚠️ Error while processing query: {e}",
            "source_documents": []
        }

# ---------------- CLI TEST ---------------- #
if __name__ == "__main__":
    import readline
    print("🚀 Hero-Vida Multi-Agent Central Router (CLI Test)")
    history = []
    while True:
        q = input("\nQuery: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break
        res = route_query(q, history)
        history.append((q, res.get("answer", "")))
        print("\n--- Response ---\n")
        print(res.get("answer", "No answer."))
        if res.get("source_documents"):
            print("\n📄 Sources:")
            for doc in res["source_documents"][:5]:
                fname = doc.get("filename", "Unknown")
                page = doc.get("page_or_row", "N/A")
                print(f"- {fname} (page/row: {page})")
        print("\n" + "-"*60 + "\n")
