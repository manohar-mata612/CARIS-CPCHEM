"""
rag/retriever.py
----------------
Queries the ChromaDB vector store and generates answers
using Nvidia Nemotron via LangChain.

This is what Agent 2 (RAG Diagnostician) calls when an
anomaly is detected. It answers questions like:
  "What does kurtosis 11.3 on CB-CGC-001 indicate?"
  "What are the recommended actions for inner race fault?"
  "What parts are needed for bearing replacement?"

WHY NVIDIA NEMOTRON FOR GENERATION:
  nemotron-mini-4b-instruct is purpose-built for agentic
  workflows — it follows tool-call formats reliably and
  generates structured JSON outputs without hallucinating.
  It runs free at 40 RPM on build.nvidia.com.

  The OpenAI-compatible API means switching to Azure OpenAI
  in production is one config line change.
"""

import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

DB_DIR          = "chroma_db"
COLLECTION_NAME = "caris_cpchem_knowledge"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_LLM_MODEL   = "nvidia/nemotron-mini-4b-instruct"
TOP_K = 4  # number of chunks to retrieve


def check_dependencies():
    try:
        import chromadb
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        return True
    except ImportError as e:
        print(f"Missing: {e}")
        print("Run: pip install chromadb langchain-nvidia-ai-endpoints langchain-community langchain")
        sys.exit(1)


def load_vectorstore():
    """Load the ChromaDB vector store from disk."""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    from langchain_community.vectorstores import Chroma

    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at {DB_DIR}\n"
            "Run: python -m rag.ingest first"
        )

    embeddings = NVIDIAEmbeddings(
        model=NVIDIA_EMBED_MODEL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="NONE",
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    print(f"Loaded ChromaDB from {DB_DIR}/")
    return vectorstore


def build_prompt(query: str, context_chunks: list) -> str:
    """
    Build the RAG prompt that combines retrieved context
    with the user query.

    The prompt instructs the LLM to:
    1. Answer ONLY from the provided context
    2. Cite the source document for every claim
    3. Return structured JSON for Agent 3 to process
    """
    context = "\n\n---\n\n".join([
        f"SOURCE: {chunk.metadata.get('source', 'unknown')}\n"
        f"CONTENT: {chunk.page_content}"
        for chunk in context_chunks
    ])

    prompt = f"""You are CARIS, the Cedar Bayou Agentic Reliability Intelligence System.
You help CPChem reliability engineers diagnose equipment faults and recommend actions.

RULES:
1. Answer ONLY using the provided context documents below.
2. Always cite the source document for each claim.
3. If the context does not contain enough information, say so clearly.
4. Always include specific numbers (thresholds, part numbers, labor hours).
5. Return your answer as valid JSON in the exact format shown.

CONTEXT DOCUMENTS:
{context}

ENGINEER QUERY: {query}

Respond with this exact JSON format:
{{
  "diagnosis": "One sentence summary of what is wrong",
  "root_causes": [
    {{"cause": "description", "confidence": "high/medium/low", "source": "filename"}}
  ],
  "severity": "P1/P2/P3",
  "recommended_actions": [
    {{"action": "description", "timeline": "immediate/24hrs/72hrs", "source": "filename"}}
  ],
  "parts_required": ["part name and number"],
  "estimated_labor_hours": 0,
  "safety_notes": "key safety requirements",
  "source_documents": ["list of source files used"]
}}"""
    return prompt


class CARISRetriever:
    """
    Main retriever class used by Agent 2 in Phase 4.

    Usage:
      retriever = CARISRetriever()
      result = retriever.diagnose({
          "equipment_id": "CB-CGC-001",
          "vibration_rms": 4.8,
          "kurtosis": 11.3,
          "crest_factor": 6.8,
          "anomaly_type": "inner_race"
      })
    """

    def __init__(self):
        check_dependencies()
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langchain_core.messages import HumanMessage
        from langchain_core.documents import Document

        self.vectorstore = load_vectorstore()
        self.llm = ChatNVIDIA(
            model=NVIDIA_LLM_MODEL,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.1,   # low temp = consistent structured output
            max_tokens=1024,
        )
        self.HumanMessage = HumanMessage
        print(f"LLM loaded: {NVIDIA_LLM_MODEL}")

    def retrieve(self, query: str, k: int = TOP_K) -> list:
        """
        Retrieve top-k most relevant document chunks for a query.
        Uses cosine similarity in embedding space.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        chunks  = [doc for doc, score in results]
        scores  = [score for doc, score in results]

        print(f"\nRetrieved {len(chunks)} chunks for query:")
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            source = chunk.metadata.get("source", "unknown")
            preview = chunk.page_content[:80].replace("\n", " ")
            print(f"  [{i+1}] score={score:.3f} | {source} | {preview}...")

        return chunks

    def diagnose(self, sensor_alert: dict) -> dict:
        """
        Full RAG diagnosis pipeline for a sensor anomaly alert.

        Parameters
        ----------
        sensor_alert : dict with keys:
          equipment_id  : e.g. "CB-CGC-001"
          vibration_rms : float
          kurtosis      : float
          crest_factor  : float
          severity      : "warning" or "critical"
          fault_type    : e.g. "inner_race" (from anomaly model)

        Returns
        -------
        dict with diagnosis, recommended actions, parts, labor
        """
        equipment_id = sensor_alert.get("equipment_id", "CB-CGC-001")
        rms          = sensor_alert.get("vibration_rms", 0)
        kurtosis     = sensor_alert.get("kurtosis", 0)
        crest_factor = sensor_alert.get("crest_factor", 0)
        severity     = sensor_alert.get("severity", "warning")
        fault_type   = sensor_alert.get("fault_type", "unknown")

        # build natural language query from sensor data
        query = (
            f"Equipment {equipment_id} shows vibration RMS of {rms:.2f} mm/s, "
            f"kurtosis of {kurtosis:.1f}, and crest factor of {crest_factor:.1f}. "
            f"Anomaly model classified this as {fault_type} fault with {severity} severity. "
            f"What is the diagnosis, recommended action, priority level, and required parts?"
        )

        print(f"\n{'='*60}")
        print(f"DIAGNOSING: {equipment_id}")
        print(f"  RMS={rms:.2f} | Kurtosis={kurtosis:.1f} | Severity={severity}")
        print(f"{'='*60}")

        # retrieve relevant chunks
        chunks = self.retrieve(query)

        # build and send prompt to Nemotron
        prompt = build_prompt(query, chunks)

        print(f"\nQuerying {NVIDIA_LLM_MODEL}...")
        response = self.llm.invoke([self.HumanMessage(content=prompt)])
        raw_text = response.content.strip()

        # parse JSON response
        result = self._parse_response(raw_text, sensor_alert)
        return result

    def _parse_response(self, raw_text: str, sensor_alert: dict) -> dict:
        """Parse LLM JSON response with fallback for malformed output."""
        # strip markdown code blocks if present
        text = raw_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
            parsed["equipment_id"]   = sensor_alert.get("equipment_id")
            parsed["sensor_readings"] = {
                "rms":          sensor_alert.get("vibration_rms"),
                "kurtosis":     sensor_alert.get("kurtosis"),
                "crest_factor": sensor_alert.get("crest_factor"),
            }
            return parsed
        except json.JSONDecodeError:
            # fallback: return raw text wrapped in dict
            print("WARNING: LLM returned non-JSON. Using raw text fallback.")
            return {
                "diagnosis":          raw_text[:300],
                "severity":           sensor_alert.get("severity", "P2"),
                "recommended_actions": [{"action": "Manual inspection required",
                                         "timeline": "24hrs"}],
                "parts_required":     ["Inspect before ordering"],
                "raw_response":       raw_text,
                "equipment_id":       sensor_alert.get("equipment_id"),
            }

    def ask(self, question: str) -> str:
        """
        Simple question-answer interface for testing.
        Ask any maintenance question in plain English.
        """
        chunks = self.retrieve(question)
        prompt = build_prompt(question, chunks)
        response = self.llm.invoke([self.HumanMessage(content=prompt)])
        return response.content


def main():
    """Test the retriever with sample queries."""
    print("=== CARIS RAG Retriever Test ===\n")

    retriever = CARISRetriever()

    # test 1: sensor alert diagnosis
    print("\n--- Test 1: Sensor Alert Diagnosis ---")
    alert = {
        "equipment_id":  "CB-CGC-001",
        "vibration_rms": 4.8,
        "kurtosis":      11.3,
        "crest_factor":  6.8,
        "severity":      "critical",
        "fault_type":    "inner_race",
    }
    result = retriever.diagnose(alert)
    print("\nDiagnosis result:")
    print(json.dumps(result, indent=2))

    # test 2: plain question
    print("\n--- Test 2: Plain Question ---")
    answer = retriever.ask(
        "What bearing parts are needed to replace the charge gas compressor bearing?"
    )
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
