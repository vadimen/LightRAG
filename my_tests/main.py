import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import AzureOpenAI
import tiktoken

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

WORKING_DIR = "./jobs_graphs"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)

total_llm_tokens = 0
total_embedding_tokens = 0

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    global total_llm_tokens
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    input_tokens = 0
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        input_tokens += count_tokens(system_prompt)
    if history_messages:
        messages.extend(history_messages)
        for msg in history_messages:
            input_tokens += count_tokens(msg["content"])
    
    messages.append({"role": "user", "content": prompt})
    input_tokens += count_tokens(prompt)

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
        max_tokens=500,
    )
    
    response = chat_completion.choices[0].message.content
    output_tokens = count_tokens(response)
    
    print(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
    total_llm_tokens += input_tokens + output_tokens
    return response

async def embedding_func(texts: list[str]) -> np.ndarray:
    global total_embedding_tokens
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    
    total_tokens = sum(count_tokens(text) for text in texts)
    total_embedding_tokens += total_tokens
    print(f"Embedding token usage: {total_tokens}")
    
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Resposta do llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Resultado do embedding_func: ", result.shape)
    print("Dimensão da embedding: ", result.shape[1])


asyncio.run(test_funcs())

embedding_dimension = 1536

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

job1 = open("./job1.txt", encoding="utf-8")

rag.insert([job1.read()])

# query_text = "What are the main themes?"

# print("Result (Naive):")
# print(rag.query(query_text, param=QueryParam(mode="naive")))

# print("\nResult (Local):")
# print(rag.query(query_text, param=QueryParam(mode="local")))

# print("\nResult (Global):")
# print(rag.query(query_text, param=QueryParam(mode="global")))

# print("\nResult (Hybrid):")
# print(rag.query(query_text, param=QueryParam(mode="hybrid")))

print(f"\nTotal tokens used:")
print(f"LLM tokens: {total_llm_tokens}")
print(f"Embedding tokens: {total_embedding_tokens}")
print(f"Grand total: {total_llm_tokens + total_embedding_tokens}")
