from rag_retrieve import retrieve
from rag_guardrails import decide_answer_or_abstain, build_citations, suggest_clarifying_question

def main():
    q = input("Ask a question: ").strip()
    matches = retrieve(q, top_k=5)

    decision = decide_answer_or_abstain(matches, user_query=q)
    citations = build_citations(matches, max_citations=3)

    print("\nDECISION:", decision.action)
    print("REASON:", decision.reason)
    print("SIGNALS:", decision.signals)

    print("\nTOP MATCHES:")
    for c in citations:
        print(f"- score={c['score']} {c['source']}:{c['chunk_id']}")
        print(f"  {c['snippet']}\n")

    if decision.action == "ABSTAIN":
        print("CLARIFYING QUESTION:", suggest_clarifying_question(q))
    else:
        # For the demo, you can keep "answer" as a simple grounded summary from the best chunk.
        # Later we’ll plug an LLM for a nicer answer.
        best = citations[0]["snippet"] if citations else ""
        print("DRAFT ANSWER (grounded):", best)

if __name__ == "__main__":
    main()
