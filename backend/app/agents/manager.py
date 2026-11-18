"""Manager agent: plan, retrieve evidence, delegate to experts, and synthesize."""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from langchain_core.prompts import ChatPromptTemplate

from app.agents.experts import EXPERTS, EXPERT_PROMPT, SYNTHESIS_PROMPT
from app.rag.retrieval import retrieve
from app.services.generator import get_chat_model


logger = logging.getLogger(__name__)
NO_EVIDENCE_ANSWER = "The provided documents do not contain enough information to answer this question."
MAX_EXPERTS = 3

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the routing manager for a Hero Motocorp internal-document research system.
Choose the smallest useful set of specialists from: company, product, sales, pricing, market, general.
Select up to 3 only when a question truly crosses domains. Use general when no specialist clearly fits.
Rewrite the question as a short retrieval query while preserving its intent and named entities.
Return only valid JSON with this exact shape: {{\"agents\":[\"sales\"],\"retrieval_query\":\"...\"}}.
Do not answer the question. Never request web browsing or use outside knowledge.""",
        ),
        ("human", "Question: {question}"),
    ]
)


def _content_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _fallback_plan(question: str) -> dict:
    normalized = question.casefold()
    scores = {
        key: sum(1 for term in expert["keywords"] if term in normalized)
        for key, expert in EXPERTS.items()
        if key != "general"
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [key for key, score in ranked if score > 0][:MAX_EXPERTS]
    return {
        "agents": selected or ["general"],
        "retrieval_query": question,
    }


def _plan(question: str) -> dict:
    fallback = _fallback_plan(question)
    try:
        response = get_chat_model().invoke(
            PLANNER_PROMPT.format_messages(question=question)
        )
        raw = _content_text(response)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return fallback

        candidate = json.loads(match.group(0))
        agents = candidate.get("agents", [])
        valid_agents = list(dict.fromkeys(
            key for key in agents if isinstance(key, str) and key in EXPERTS
        ))[:MAX_EXPERTS]
        retrieval_query = candidate.get("retrieval_query")
        if not isinstance(retrieval_query, str) or not retrieval_query.strip():
            retrieval_query = question
        return {
            "agents": valid_agents or fallback["agents"],
            "retrieval_query": retrieval_query.strip()[:1000],
        }
    except Exception:
        logger.warning("Manager planning failed; using deterministic domain routing.", exc_info=True)
        return fallback


def _format_context(documents) -> str:
    blocks = []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("filename", "unknown")
        location = metadata.get("page", metadata.get("row_index"))
        location_label = f", page/row {location}" if location is not None else ""
        blocks.append(f"[S{number}] {source}{location_label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _run_expert(agent_key: str, question: str, context: str) -> dict:
    expert = EXPERTS[agent_key]
    answer = get_chat_model().invoke(
        EXPERT_PROMPT.format_messages(
            expert_name=expert["name"],
            focus=expert["focus"],
            context=context,
            question=question,
        )
    )
    return {"key": agent_key, "name": expert["name"], "output": _content_text(answer)}


def run_agentic_rag(question: str, top_k: int | None = None) -> dict:
    started = perf_counter()
    plan_started = perf_counter()
    plan = _plan(question)
    planning_ms = (perf_counter() - plan_started) * 1000

    retrieval_started = perf_counter()
    documents = retrieve(plan["retrieval_query"], top_k=top_k)
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    context = _format_context(documents)

    trace = [
        {
            "name": "Research Manager",
            "role": "Planning",
            "status": "completed",
            "output": "Delegated to " + ", ".join(EXPERTS[key]["name"] for key in plan["agents"]) + ".",
        },
        {
            "name": "Document Retrieval Agent",
            "role": "Evidence retrieval",
            "status": "completed",
            "output": f"Retrieved {len(documents)} evidence chunks from the internal knowledge base.",
        },
    ]

    if not documents:
        for key in plan["agents"]:
            trace.append({
                "name": EXPERTS[key]["name"],
                "role": "Domain analysis",
                "status": "skipped",
                "output": "No evidence was retrieved, so this specialist was not called.",
            })
        return {
            "mode": "agentic",
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "agents": trace,
            "metrics": {
                "planning_ms": round(planning_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "agent_ms": 0.0,
                "synthesis_ms": 0.0,
                "total_ms": round((perf_counter() - started) * 1000, 2),
                "retrieved_chunks": 0,
            },
        }

    agent_started = perf_counter()
    results_by_key = {}
    failed_agents = []
    with ThreadPoolExecutor(max_workers=min(MAX_EXPERTS, len(plan["agents"]))) as pool:
        futures = {
            pool.submit(_run_expert, key, question, context): key
            for key in plan["agents"]
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results_by_key[key] = future.result()
            except Exception:
                logger.exception("Domain expert failed: %s", key)
                failed_agents.append(key)
    results = [results_by_key[key] for key in plan["agents"] if key in results_by_key]
    for key in failed_agents:
        trace.append({
            "name": EXPERTS[key]["name"],
            "role": "Domain analysis",
            "status": "failed",
            "output": "This specialist could not complete the analysis.",
        })
    if not results:
        raise RuntimeError("All selected research specialists failed.")
    agent_ms = (perf_counter() - agent_started) * 1000

    for result in results:
        trace.append({
            "name": result["name"],
            "role": "Domain analysis",
            "status": "completed",
            "output": result["output"],
        })

    synthesis_ms = 0.0
    if len(results) == 1:
        answer = results[0]["output"]
    else:
        synthesis_started = perf_counter()
        findings = "\n\n".join(
            f"{result['name']}:\n{result['output']}" for result in results
        )
        try:
            response = get_chat_model().invoke(
                SYNTHESIS_PROMPT.format_messages(
                    context=context,
                    findings=findings,
                    question=question,
                )
            )
            answer = _content_text(response)
            synthesis_status = "completed"
            synthesis_output = "Combined specialist findings and retained source citations."
        except Exception:
            logger.exception("Manager synthesis failed; returning specialist findings.")
            answer = findings
            synthesis_status = "fallback"
            synthesis_output = "Synthesis was unavailable; returned the grounded specialist findings separately."
        synthesis_ms = (perf_counter() - synthesis_started) * 1000
        trace.append({
            "name": "Research Manager",
            "role": "Synthesis",
            "status": synthesis_status,
            "output": synthesis_output,
        })

    sources = []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        sources.append({
            "source_id": f"S{number}",
            "filename": metadata.get("filename", "unknown"),
            "page": metadata.get("page"),
            "row_index": metadata.get("row_index"),
            "source_type": metadata.get("source_type"),
            "chunk_id": metadata.get("chunk_id"),
            "score": metadata.get("score"),
            "content": doc.page_content[:500],
        })

    return {
        "mode": "agentic",
        "answer": answer,
        "sources": sources,
        "agents": trace,
        "metrics": {
            "planning_ms": round(planning_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "agent_ms": round(agent_ms, 2),
            "synthesis_ms": round(synthesis_ms, 2),
            "total_ms": round((perf_counter() - started) * 1000, 2),
            "retrieved_chunks": len(documents),
        },
    }


def run_single_pass_rag(question: str, top_k: int | None = None) -> dict:
    """One retrieval and one general grounded generation call for comparison."""
    started = perf_counter()
    retrieval_started = perf_counter()
    documents = retrieve(question, top_k=top_k)
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    context = _format_context(documents)

    if not documents:
        answer = NO_EVIDENCE_ANSWER
        generation_ms = 0.0
    else:
        generation_started = perf_counter()
        response = get_chat_model().invoke(
            EXPERT_PROMPT.format_messages(
                expert_name="General Business Intelligence Analyst",
                focus="answer the user's question using relevant Hero Motocorp business documents",
                context=context,
                question=question,
            )
        )
        answer = _content_text(response)
        generation_ms = (perf_counter() - generation_started) * 1000

    sources = []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        sources.append({
            "source_id": f"S{number}",
            "filename": metadata.get("filename", "unknown"),
            "page": metadata.get("page"),
            "row_index": metadata.get("row_index"),
            "source_type": metadata.get("source_type"),
            "chunk_id": metadata.get("chunk_id"),
            "score": metadata.get("score"),
            "content": doc.page_content[:500],
        })

    return {
        "mode": "baseline",
        "answer": answer,
        "sources": sources,
        "agents": [
            {
                "name": "Single-pass RAG baseline",
                "role": "Retrieve and generate",
                "status": "completed" if documents else "skipped",
                "output": (
                    "One general grounded answer call; provided for comparison with the specialist workflow."
                    if documents
                    else "No evidence was retrieved, so generation was skipped."
                ),
            }
        ],
        "metrics": {
            "planning_ms": 0.0,
            "retrieval_ms": round(retrieval_ms, 2),
            "agent_ms": round(generation_ms, 2),
            "synthesis_ms": 0.0,
            "total_ms": round((perf_counter() - started) * 1000, 2),
            "retrieved_chunks": len(documents),
        },
    }
