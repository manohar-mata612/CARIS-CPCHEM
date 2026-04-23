"""
agents/diagnostic_agent.py
--------------------------
Agent 2 — RAG Diagnostician

WHAT IT DOES:
  Takes the anomaly detected by Agent 1, queries the ChromaDB
  knowledge base with Nvidia Nemotron, and writes a structured
  diagnosis to AgentState.

WHY RAG HERE:
  The LLM alone would hallucinate part numbers, thresholds,
  and procedures. RAG grounds every answer in the actual
  CPChem maintenance manual and historical failure records.
  Every claim the agent makes cites a real source document.

WHAT IT WRITES TO STATE:
  diagnosis, root_causes, recommended_actions, parts_required,
  estimated_labor_hours, safety_notes, source_documents
"""

import os
import sys
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.state import AgentState

DB_DIR             = "chroma_db"
COLLECTION_NAME    = "caris_cpchem_knowledge"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_LLM_MODEL   = "nvidia/nemotron-mini-4b-instruct"
TOP_K = 4

# module-level singletons — loaded once when container starts
_vectorstore = None
_llm         = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings = NVIDIAEmbeddings(
            model=NVIDIA_EMBED_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            truncate="NONE",
        )
        _vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def get_llm():
    global _llm
    if _llm is None:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        _llm = ChatNVIDIA(
            model=NVIDIA_LLM_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.1,
            max_tokens=1024,
        )
    return _llm


def _build_query(state: AgentState) -> str:
    """Convert sensor state into a natural language diagnostic query."""
    return (
        f"Equipment {state.get('equipment_id', 'unknown')} shows "
        f"vibration RMS of {state.get('vibration_rms', 0):.2f} mm/s, "
        f"kurtosis of {state.get('kurtosis', 0):.1f}, "
        f"crest factor of {state.get('crest_factor', 0):.1f}. "
        f"Anomaly model classified this as {state.get('fault_type', 'unknown')} "
        f"fault with {state.get('severity', 'warning')} severity. "
        f"What is the diagnosis, recommended priority, required parts, "
        f"and immediate actions?"
    )


def _build_prompt(query: str, chunks: list) -> str:
    """Build RAG prompt combining retrieved context with query."""
    context = "\n\n---\n\n".join([
        f"SOURCE: {c.metadata.get('source', 'unknown')}\n"
        f"CONTENT: {c.page_content}"
        for c in chunks
    ])

    return f"""You are CARIS, the Cedar Bayou Agentic Reliability Intelligence System.
Diagnose the equipment fault using ONLY the context documents below.
Always cite the source document. Return ONLY valid JSON, no other text.

CONTEXT:
{context}

QUERY: {query}

Return this exact JSON:
{{
  "diagnosis": "one sentence description of the fault",
  "root_causes": [{{"cause": "...", "confidence": "high/medium/low", "source": "filename"}}],
  "severity": "P1/P2/P3",
  "recommended_actions": [{{"action": "...", "timeline": "immediate/24hrs/72hrs", "source": "filename"}}],
  "parts_required": ["part name and number"],
  "estimated_labor_hours": 8,
  "safety_notes": "key safety requirement",
  "source_documents": ["filename1", "filename2"]
}}"""


def diagnostic_agent(state: AgentState) -> AgentState:
    """
    Agent 2 node function for LangGraph.

    Only runs if Agent 1 detected an anomaly.
    Queries ChromaDB + Nemotron and writes diagnosis to state.
    """
    log_entry = {
        "agent":     "diagnostic_agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action":    "rag_diagnosis",
    }

    try:
        from langchain_core.messages import HumanMessage

        query    = _build_query(state)
        vs       = get_vectorstore()
        llm      = get_llm()

        # retrieve top-k relevant chunks
        results  = vs.similarity_search_with_score(query, k=TOP_K)
        chunks   = [doc for doc, _ in results]
        sources  = [doc.metadata.get("source", "unknown") for doc in chunks]

        print(f"[Agent 2] Retrieved {len(chunks)} chunks from: "
              f"{list(set(sources))}")

        # query Nemotron
        prompt   = _build_prompt(query, chunks)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()

        # parse JSON — strip markdown if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        parsed = json.loads(raw)

        print(f"[Agent 2] Diagnosis: {parsed.get('diagnosis', '')[:80]}")
        print(f"[Agent 2] Priority:  {parsed.get('severity', 'P2')}")
        print(f"[Agent 2] Parts:     {parsed.get('parts_required', [])}")

        log_entry["diagnosis"] = parsed.get("diagnosis", "")
        log_entry["severity"]  = parsed.get("severity", "P2")
        log_entry["sources"]   = sources

        return {
            **state,
            "diagnosis":             parsed.get("diagnosis", ""),
            "root_causes":           parsed.get("root_causes", []),
            "recommended_actions":   parsed.get("recommended_actions", []),
            "parts_required":        parsed.get("parts_required", []),
            "estimated_labor_hours": parsed.get("estimated_labor_hours", 8),
            "safety_notes":          parsed.get("safety_notes", ""),
            "source_documents":      parsed.get("source_documents", sources),
            "final_priority":        parsed.get("severity", "P2"),
            "agent_log":             state.get("agent_log", []) + [log_entry],
        }

    except json.JSONDecodeError as e:
        print(f"[Agent 2] JSON parse error: {e}. Using fallback.")
        log_entry["error"] = str(e)
        return {
            **state,
            "diagnosis":           "Automated diagnosis failed — manual inspection required",
            "recommended_actions": [{"action": "Manual bearing inspection",
                                     "timeline": "24hrs"}],
            "parts_required":      ["Inspect before ordering"],
            "final_priority":      "P2",
            "agent_log":           state.get("agent_log", []) + [log_entry],
        }

    except Exception as e:
        print(f"[Agent 2] ERROR: {e}")
        log_entry["error"] = str(e)
        return {
            **state,
            "diagnosis": f"Diagnosis error: {str(e)}",
            "error":     str(e),
            "agent_log": state.get("agent_log", []) + [log_entry],
        }