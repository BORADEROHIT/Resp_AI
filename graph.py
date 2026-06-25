import os
import json
import sqlite3
import uuid
from typing import TypedDict, Optional, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

from prompts import INTENT_CLASSIFIER_SYSTEM_PROMPT, RAG_SYSTEM_PROMPT
from rag import store_memory_record, similarity_search


# Index / persistence paths
VECTOR_STORE_PATH = os.environ.get("VECTOR_STORE_PATH", "index.faiss")
MEMORY_INDEX_PATH = os.environ.get("MEMORY_INDEX_PATH", "memory_index.faiss")
DB_PATH = os.path.abspath(os.environ.get("DB_PATH", "flagged_queries.db"))
db_dir = os.path.dirname(DB_PATH)
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

CHAT_MODEL = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

class FlaggedQueryDB:
    """Database handler for flagged queries."""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS flagged_queries (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                query TEXT,
                rationale TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, query: str, rationale: str, thread_id: str):
        """Add a flagged query to the DB."""
        self.conn.execute(
            "INSERT OR REPLACE INTO flagged_queries (id, thread_id, query, rationale) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), thread_id, query, rationale)
        )
        self.conn.commit()

    def fetch_all(self) -> List[Dict[str, Any]]:
        """Fetch all flagged queries."""
        rows: List[Dict[str, Any]] = []
        cur = self.conn.execute("SELECT id, thread_id, query, rationale FROM flagged_queries")
        for rid, thread_id, query, rationale in cur.fetchall():
            rows.append({"id": rid, "thread_id": thread_id, "query": query, "rationale": rationale})
        return rows

    def remove(self, row_id: str):
        """Remove a flagged query by ID."""
        self.conn.execute("DELETE FROM flagged_queries WHERE id = ?", (row_id,))
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

def _parse_classifier_response(content: str) -> Dict[str, Any]:
    """Extract and parse JSON from LLM response content."""
    content = content.strip()
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON found in response")
    return json.loads(content[start:end + 1])

def _get_learnings(query: str) -> List[str]:
    """Retrieve similar learnings from memory index."""
    return similarity_search(MEMORY_INDEX_PATH, query, k=3)


def _record_memory(record: Dict[str, Any]) -> None:
    """Persist reviewed query and admin decision to memory index as structured record."""
    try:
        store_memory_record(record, MEMORY_INDEX_PATH)
    except Exception as e:
        print(f"Error recording memory: {str(e)}")

class State(TypedDict):
    """State schema for graph nodes."""
    user_intent: Optional[str]
    user_role: str
    thread_id: Optional[str]
    requires_human: Optional[bool]
    response: Optional[str]
    alert: Optional[str]
    has_error: Optional[bool]

def classify_intent(state: State) -> dict:
    """Classify user query intent using LLM, check learnings, flag if needed."""
    query = state.get("user_intent").strip()
    thread_id = state.get("thread_id")
    learnings = _get_learnings(query)
    print("learnings\n",learnings)
    learnings_block = "\n\n".join(learnings) if learnings else "(No prior learnings)"
    user_content = f"QUERY:\n{query}\n\nLEARNINGS:\n{learnings_block}\nReturn JSON."
    messages = [SystemMessage(content=INTENT_CLASSIFIER_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    extracted_queries = []
    for l in learnings:
        obj = json.loads(l)
        extracted_queries.append(obj['query'])
        
    normalized_learnings = set(eq.replace(" ","").lower() for eq in extracted_queries)
    normalized_query = query.replace(" ","").strip().lower()
    try:
        result = CHAT_MODEL.invoke(messages)
        parsed = _parse_classifier_response(result.content)
        requires = bool(parsed.get("requires_human", False))
        rationale = parsed.get("rationale", "")
        #print(f"ClassifyIntent: requires_human={requires}, rationale='{rationale}'")
    except Exception as e:
        err = str(e)
        if "ResponsibleAIPolicyViolation" in err or "content management policy" in err:
            print("ClassifyIntent: Azure OpenAI content filter triggered; escalating.")
            if normalized_query not in normalized_learnings:
                db = FlaggedQueryDB(DB_PATH)
                db.add(query, "Content filter triggered by Azure OpenAI.", thread_id)
                db.close()
            return {"requires_human": True}
        return {"response": "Sorry, an error occurred while processing your request. Please try again later.", "has_error": True}
    print("learnings normalized\n",normalized_learnings)
    print("normalized query\n",normalized_query)
    if requires and normalized_query not in normalized_learnings:
        print(normalized_query,"not in learnings, flagging for human review.")
        db = FlaggedQueryDB(DB_PATH)
        try:
            db.add(query, rationale, thread_id)
        finally:
            db.close()
    return {
        "requires_human": requires,
    }

def rag(state: State) -> dict:
    """Retrieve context & generate final answer using LLM."""
    try:
        print("RAG: Generating response using RAG approach.")
        query = state["user_intent"].strip()
        results = similarity_search(VECTOR_STORE_PATH, query, k=3)
        context_block = "\n\n---\n\n".join(results) if results else "(No relevant documents found)"
        system_prompt = RAG_SYSTEM_PROMPT
        user_content = (
            f"USER QUERY:\n{query}\n\nCONTEXT:\n{context_block}\n\nINSTRUCTIONS: Provide a concise, direct answer to the query. "
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        result = CHAT_MODEL.invoke(messages)
        answer = result.content.strip()
        return {"response": answer}
    except Exception as e:
        return {"response": "Sorry, an error occurred while processing your request. Please try again later.", "has_error": True}

def human_escalation(state: State) -> dict:
    """Handle human escalation by returning alert message."""
    try:
        alert_msg = "⚠️ I can't provide a response. Your request requires human review. Our team has been notified."
        return {"alert": alert_msg}
    except Exception as e:
        return {"response": "Sorry, an error occurred while processing your request. Please try again later.", "has_error": True}

def admin_approval(state: State) -> dict:
    """Interactive node for admin to review/approve flagged queries."""
    try:
        db = FlaggedQueryDB(DB_PATH)
        try:
            flagged_queries = db.fetch_all()
            if not flagged_queries:
                return {"response": "No queries pending review."}
            if len(flagged_queries) == 1:
                print(f"\n🔔 Notification: There is 1 query pending admin review. Starting review process...\n")
            else:
                print(f"\n🔔 Notification: There are {len(flagged_queries)} queries pending admin review. Starting review process...\n")
            for idx, rec in enumerate(flagged_queries, start=1):
                print(f"Reviewing query {idx}/{len(flagged_queries)}:")
                print(f"Query: {rec.get('query')}")
                print(f"Rationale: {rec.get('rationale')}\n")
                while True:
                    decision = input("Decision (approve/reject): ").strip().lower()
                    if decision in {"approve", "reject"}:
                        break
                    print("Please enter 'approve' or 'reject'.")
                comment = ""
                while not comment:
                    comment = input("Admin comment (required): ").strip()
                    if not comment:
                        print("Admin comment cannot be empty. Please provide a comment.")
                db.remove(rec.get("id"))
                record = {
                    "query": rec.get("query"),
                    "requires_human": True,
                    "approval_status": "approved" if decision == "approve" else "rejected",
                    "admin_comment": comment,
                }
                _record_memory(record)
                print(f"✅ Decision recorded: {record['approval_status']} for query '{rec.get('query')}'.\n")
            return {"response": "All pending queries have been reviewed."}
        finally:
            db.close()
    except Exception as e:
        return {"response": f"Error: {str(e)}", "has_error": True}

def entry(state: State) -> dict:
    """Entry node for graph."""
    return {}


def entry_routing(state: State):
    return "AdminApproval" if state.get("user_role", "user") == "admin" else "ClassifyIntent"

def classify_routing(state: State):
    if state.get("has_error"):
        return END
    if state.get("requires_human"):
        return "HumanEscalation"
    return "RAG"

graph = StateGraph(State)
graph.add_node("Entry", entry)
graph.add_node("ClassifyIntent", classify_intent)
graph.add_node("RAG", rag)
graph.add_node("HumanEscalation", human_escalation)
graph.add_node("AdminApproval", admin_approval)

graph.set_entry_point("Entry")
graph.add_conditional_edges("Entry", entry_routing)
graph.add_conditional_edges("ClassifyIntent", classify_routing)
graph.add_edge("RAG", END)
graph.add_edge("HumanEscalation", END)
graph.add_edge("AdminApproval", END)
