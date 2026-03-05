from ddgs import DDGS


def web_search(query: str) -> str:
    """
    Performs a quick web search and returns a short summary.
    """
    query = (query or "").strip()
    if not query:
        return "What should I search for, sir?"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "I couldn't find anything useful online, sir."

        lines = []
        for r in results[:3]:
            title = r.get("title", "")
            snippet = r.get("body", "")
            lines.append(f"{title} — {snippet}")

        return "Here's what I found online, sir: " + " ".join(lines)

    except Exception:
        return "Web search failed, sir."