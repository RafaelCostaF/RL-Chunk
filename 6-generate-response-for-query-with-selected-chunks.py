# !pip install openai

from llmFunctions import *
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

def process_llm_response(input_file, output_file, response_column, llm_function):
    """
    Applies a given LLM function to queries and selected chunks,
    saving the results to a Parquet file.
    
    Parameters:
    - input_file: Path to the input Parquet file with queries and selected chunks.
    - output_file: Path to save the resulting Parquet file with LLM responses.
    - response_column: Name of the new column for storing the LLM responses.
    - llm_function: The function used to generate the responses (e.g., Gemini or OpenAI).
    """
    df = pd.read_parquet(input_file)
    
    df[response_column] = df.progress_apply(
        lambda row: llm_function(row["query"], row["bm25_chunks_selected"]),
        axis=1
    )
    
    df = df.astype(str)
    df.to_parquet(output_file)

# Example usage:
# For Gemini
process_llm_response(
    input_file="response_metrics_file_6_with_calculated_metrics_selected_bm25.parquet",
    output_file="response_metrics_file_6_with_calculated_metrics_selected_bm25_com_response_bm25_gemini.parquet",
    response_column="llm_response_bm25",
    llm_function=get_response_from_llm_gemini
)

# For another LLM (e.g., OpenAI)
process_llm_response(
    input_file="response_metrics_file_6_with_calculated_metrics_selected_bm25.parquet",
    output_file="rresponse_metrics_file_6_with_calculated_metrics_selected_bm25_com_response_bm25.parquet",
    response_column="llm_response_faiss",
    llm_function=get_response_from_llm
)
