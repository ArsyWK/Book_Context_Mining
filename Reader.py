from pypdf import PdfReader
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph,END,START
from typing import TypedDict
from langchain.messages import HumanMessage,SystemMessage

import pathlib

Output = pathlib.Path("./Output/Output.txt")
Output_harry =pathlib.Path("./Output/Output_harry.txt")


# fastllm = Anthropic(model="claude-3-haiku-20240307")
# primaryllm = MistralAI(model='mistral-large-latest')
helperllm = ChatOllama(model='llama3.2',temperature=0)

class automation(TypedDict):
    raw_text: str
    preprocessed_text: str
    summary_draft: str
    iterative_compression: str

path = pathlib.Path("./BookExample/85jkr_harrypotter_1.pdf")
book = PdfReader(path)
extracted_text = "\n".join([page.extract_text() for page in book.pages if page.extract_text()])

# def Tokenize(sentence):
#     keyword =[]
#     postag = ["PROPN","ADJ","NOUN","VERB"]
#
#
#     nlp = model(sentence)
#     stop = stop_words.STOP_WORDS
#
#
#     punct = list(punctuation)
#     # lemma = [word.lemma_.lower().strip() for word in nlp]
#
#     # filtered_tokens = [word for word in lemma if word not in stop and word not in punct]
#
#     for token in nlp:
#         if(token.text in stop or token.text in punctuation):
#             continue
#         if(token.pos_ in postag):
#             keyword.append(token.text)
#
#     freq_word = Counter(keyword)
#     freq_word.most_common(5)
#     max_freq = Counter(keyword).most_common(1)[0][1]
#     for word in freq_word.keys():
#         freq_word[word]= (freq_word[word]/max_freq)
#     freq_word.most_common(5)
#
#     sent_strength = {}
#     for sent in nlp.sents:
#         for word in sent:
#             if word.text in freq_word.keys():
#                 if sent in sent_strength.keys():
#                     sent_strength[sent]+= freq_word[word.text]
#                 else:
#                     sent_strength[sent] = freq_word[word.text]
#
#     return sent_strength



#def chunk_book(text: str, chunk_words: int = 2500) -> list[str]:
#     """Pack consecutive paragraphs into chunks of up to ~chunk_words words.
#     A single oversized paragraph becomes its own chunk (never cut mid-paragraph)."""
#     paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
#     if len(paragraphs) <= 1:
#         paragraphs = [p.strip() for p in text.splitlines() if p.strip()] or [text.strip()]
#
#     chunks, current, size = [], [], 0
#     for para in paragraphs:
#         n = count_words(para)
#         if current and size + n > chunk_words:
#             chunks.append("\n\n".join(current))
#             current, size = [], 0
#         current.append(para)
#         size += n
#     if current:
#         chunks.append("\n\n".join(current))
#     return chunks



def preprocessing(state:automation)-> dict:
    text_content = state["raw_text"]
    system_prompt = SystemMessage(
        content="""you are an expert script-to-narrative converter. Transform input script into engaging prose narratives
while preserving the essence of the original work.""")
    human_prompt = HumanMessage(content=f"""
{text_content}
1. Convert dialogue to reported speech, including emotions and speaker traits.
2. Integrate narration and stage directions seamlessly.
3. Preserve original structure, pacing, and character voices.
4. Capture emotional tone and subtext.
5. Use varied language for different speaking styles.
6. Include relevant context from stage directions.
7. Create a cohesive narrative retaining key dramatic elements.
8. Maintain original language and formal written style.
9. Use third-person perspective.""")
    response = helperllm.invoke([system_prompt, human_prompt])
    return {"preprocessed_text": response.content}

def iterative_compression(state:automation)-> dict:
    prev_sum = state["summary_draft"]

    system_prompt = SystemMessage(content="You are an expert storyteller. Create a concise meta summary of the given previous summar")

    human_prompt = HumanMessage(
        content=f"""f
{prev_sum}
- Write a meta summary of the PREVIOUS_SUMMARY.
- Focus on key events, character traits, and interactions.
[Your meta summary here]"""
    )
    response = helperllm.invoke([system_prompt, human_prompt])

    return {"iterative_compression": response.content}


def summarizing(state:dict)-> dict:
    preprocs = state["preprocessed_text"]
    system_prompt = SystemMessage(content="You are an expert storyteller. Create a concise summary of the given part of the script.")
    human_prompt = HumanMessage(
        content=f"""
{preprocs}
- Write a summary of the PART_OF_SCRIPT.
- Focus on key events, character traits, and interactions.
[Your summary here]""")
    response = helperllm.invoke([system_prompt, human_prompt])
    return {"summary_draft": response.content}



agent_builder= StateGraph(automation)

agent_builder.add_node('preprocess',preprocessing)
agent_builder.add_node('summarizing',summarizing)
agent_builder.add_node('iterative_compression',iterative_compression)

agent_builder.add_edge(START, "preprocess")
agent_builder.add_edge("preprocess", "summarizing")
agent_builder.add_edge("summarizing", "iterative_compression")
agent_builder.add_edge("iterative_compression", END)

llm = agent_builder.compile()
initial_intput ={'raw_text':extracted_text}
final_output = llm.invoke(initial_intput)

iter= final_output["iterative_compression"]
sum= final_output["summary_draft"]
pre= final_output["preprocessed_text"]

with open(Output_harry, "a", encoding="utf-8") as op:
    op.write(pre)
    op.write(sum)
    op.write(iter)














