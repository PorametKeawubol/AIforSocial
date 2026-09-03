"""Step 9 from the lab sheet, kept in its original parameter form."""

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:1b",
    num_gpu=99,
    num_ctx=1024,
    num_thread=4,
)

response = llm.invoke("Explain HTTP/3 in simple terms.")
print(response.content)
