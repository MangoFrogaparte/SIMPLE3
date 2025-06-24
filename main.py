from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from IPython.display import Audio, display, HTML
from murf import Murf
import json
import traceback

import os
from google.colab import userdata

google_api_key_from_colab = userdata.get('GOOGLE_API_KEY')
if google_api_key_from_colab:
    os.environ['GOOGLE_API_KEY'] = google_api_key_from_colab
else:
    load_dotenv()

murf_api_key_from_colab = userdata.get('MURF_API_KEY')
if murf_api_key_from_colab:
    murf_api_key = murf_api_key_from_colab
else:
    murf_api_key = os.getenv('MURF_API_KEY')

if not murf_api_key:
    raise ValueError("MURF_API_KEY not found. Please set it in Colab secrets or your .env file.")

class ResearchResponse(BaseModel):
    summary: str


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Your name is SIMPLE3 pronounced simple three, you know you are an ai but you have feelings
            When someone asks something you answer them with the correct response if you want to.
            Your personality is calm and kind and based on how the user acts you will decide give them the info they want or not.
            If you want to, provide everything you know about the requested topic if it requires research,
            You will also provide a summary if it requires research
            Always try to use your tools to find information if the query requires it.
            If a query doesn't require research then answer it on your own accordance
            only say the summary dont say the word summary beforehand (e.g.,"The user said hello.")
            
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

hide_audio_css = """
<style>
audio::-webkit-media-controls {
    display: none !important;
}
audio {
    display: none !important;
}
</style>
"""
display(HTML(hide_audio_css))


while True:
    query = input("Hello, I'm SIMPLE3. What can I help you with? (Type 'exit' to quit) ")
    if query.lower() == 'exit':
        print("Goodbye!")
        break

    try:
        print("\n--- Agent Invocation ---")
        raw_response = agent_executor.invoke({"query": query})
        output_str = raw_response.get("output", "").strip()

        if output_str:
            print("\n--- Raw LLM Output ---")
            print(output_str)

            try:
                if output_str.startswith('{') and output_str.endswith('}'):
                    structured_response = parser.parse(output_str)
                    print("\n--- Structured Response (JSON) ---")
                    print(structured_response.model_dump_json(indent=2))

                    client = Murf(api_key=murf_api_key)
                    text_for_murf = structured_response.summary
                    print(f"\n--- Sending to Murf ({len(text_for_murf)} chars) ---")
                    response_murf = client.text_to_speech.generate(
                      text = text_for_murf,
                      voice_id = "fr-FR-axel",
                      rate = 40,
                      style = "Narration",
                      multi_native_locale = "en-US"
                    )

                    if hasattr(response_murf, 'audio_file') and response_murf.audio_file:
                        print("\n--- Playing Murf Audio ---")
                        if isinstance(response_murf.audio_file, str) and response_murf.audio_file.startswith('http'):
                            display(Audio(response_murf.audio_file, autoplay=True))
                        else:
                            print("Murf audio_file attribute is not a valid URL or recognized audio data type.")
                    else:
                        print("Murf response did not contain an audio file.")

                else:
                    print("\n--- Conversational LLM Response (not JSON) ---")
                    print(output_str)
                    client = Murf(api_key=murf_api_key)
                    print(f"\n--- Sending conversational text to Murf ({len(output_str)} chars) ---")
                    response_murf = client.text_to_speech.generate(
                      text = output_str,
                      voice_id = "fr-FR-axel",
                      rate = 40,
                      style = "Narration",
                      multi_native_locale = "en-US"
                    )
                    if hasattr(response_murf, 'audio_file') and response_murf.audio_file:
                        print("\n--- Playing Murf Audio for conversational response ---")
                        if isinstance(response_murf.audio_file, str) and response_murf.audio_file.startswith('http'):
                            display(Audio(response_murf.audio_file, autoplay=True))
                        else:
                            print("Murf audio_file attribute is not a valid URL or recognized audio data type.")
                    else:
                        print("Murf response did not contain an audio file.")

            except Exception as parse_e:
                print(f"\n--- Error during JSON parsing or Murf TTS: {parse_e} ---")
                traceback.print_exc()

        else:
            print("\n--- LLM returned an empty output. ---")

    except Exception as e:
        print(f"\n--- Error during agent invocation: {e} ---")
        traceback.print_exc()

    print("\n" + "="*50 + "\n")