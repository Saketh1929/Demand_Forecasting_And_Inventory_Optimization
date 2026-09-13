"""
Forwarding wrapper pointing to backend.agent
"""
from backend.agent import run_agent_pipeline, query_gemini_llm, generate_fallback_recommendation

__all__ = ["run_agent_pipeline", "query_gemini_llm", "generate_fallback_recommendation"]
