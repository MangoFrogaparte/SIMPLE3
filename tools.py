from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search_tool = DuckDuckGoSearchRun(name="DuckDuckGo_Search")

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

def _save_content(content: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_content_{timestamp}.txt"
    try:
        with open(filename, "w") as f:
            f.write(content)
        return f"Content successfully saved to {filename}"
    except Exception as e:
        return f"Error saving content: {e}"

save_tool = Tool(
    name="Save_Content",
    description="Use this tool to save important text content to a file. Input should be the content to save.",
    func=_save_content
)