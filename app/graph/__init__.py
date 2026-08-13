"""LangGraph RAG orchestration.

pipeline.py wires nodes; nodes.py does the work; state.py is the shared dict
passed between nodes. Prefer get_graph() / get_baseline_graph() over building
a new graph on every request.
"""
