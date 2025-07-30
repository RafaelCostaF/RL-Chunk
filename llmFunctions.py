from openai import OpenAI
import os
import re
from datetime import datetime
import json

# Path to GCP credentials (replace with your actual path or environment variable)
gcp_key_file_path = "./path-to-your-gcp-key.json"

# Setup DeepSeek (OpenAI-compatible) API
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)


def get_response_from_llm(query: str, chunks) -> tuple[str, int, int]:
    # GENERATE A RESPONSE BASED ON THE SELECTED CHUNKS


    # Create a prompt to help the LLM generate a response based on the query and chunks
    prompt = (
        "You are an intelligent assistant that answers questions exclusively based on the information provided below.\n\n"
        f"User query:\n{query}\n\n"
        f"Available sources (chunks):\n{chunks}\n\n"
        "Respond clearly, objectively, and only using the sources. Return ONLY the answer, without any additional explanations or context.\n\n"
        "if there's no answer, return empty string"
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        content = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return content, input_tokens, output_tokens

    except Exception as e:
        print(f"[LLM Error] {e}")
        return "Error generating response.", 0, 0
    
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import vertexai

def get_response_from_llm_gemini(query: str, chunks: str) -> tuple[str, int, int]:
    key_path = "path-to-your-gcp-key.json"

    try:
        # Set up credentials and initialize Vertex AI
        credentials = service_account.Credentials.from_service_account_file(key_path)
        vertexai.init(
            project="poc-743d9",
            location="us-central1",
            credentials=credentials
        )

        # Load the Gemini model
        model = GenerativeModel("gemini-2.0-flash")  # or "gemini-1.5-pro"

        # Prepare prompt
        prompt = (
            "You are an intelligent assistant that answers questions exclusively based on the information provided below.\n\n"
            f"User query:\n{query}\n\n"
            f"Available sources (chunks):\n{chunks}\n\n"
            "Respond clearly, objectively, and only using the sources.\n"
            "Return ONLY the answer, using a maximum of 10 words, without any additional explanations or context.\n"
            "If there's no answer, return an empty string.\n\n"
            "Answer:"
        )

        # Generate response
        response = model.generate_content(prompt)
        content = response.text.strip()

        # Estimate token counts (not directly provided by Vertex AI SDK)
        input_tokens = len(prompt.split())  # Rough estimate
        output_tokens = len(content.split())  # Rough estimate

        return content, input_tokens, output_tokens

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Error generating response.", 0, 0

    
def clean_response(llm_response) -> str:
    # GENERATE A RESPONSE BASED ON THE SELECTED CHUNKS


    # Create a prompt to help the LLM generate a response based on the query and chunks
    prompt = (
        "You are an intelligent assistant that answers questions exclusively based on the information provided below.\n\n"
        f"llm_response:\n{llm_response}\n\n"
        "if there's no answer using the chunks, return empty string, otherwise return the llm_response as it is."
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        content = response.choices[0].message.content.strip()

        return content

    except Exception as e:
        print(f"[LLM Error] {e}")
        return ""


def clean_llm_response(query: str, answer: str) -> str:
    """
    Uses the LLM to clean a raw answer based solely on the original query.
    Ensures the response is direct, minimal, and free of extra context.

    Args:
        query (str): The original question asked.
        answer (str): The original raw answer to be cleaned.

    Returns:
        str: A cleaned, direct answer from the LLM.
    """
    prompt = (
        "You are a helpful assistant. Your task is to refine the following answer "
        "so that it is direct, concise, and responds precisely to the question.\n\n"
        f"Question:\n{query}\n\n"
        f"Original answer:\n{answer}\n\n"
        "Rewrite the answer to be clean, precise, and only address the question. "
        "Return ONLY the cleaned answer—no commentary, no explanations.\n\n"
        "Cleaned answer:"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        cleaned = response.choices[0].message.content.strip()
        return cleaned

    except Exception as e:
        print(f"[LLM Error] {e}")
        return "Error generating cleaned answer."