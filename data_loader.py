import pandas as pd
from glob import glob
import json
from tqdm import tqdm
from config import Paths

def load_and_preprocess_data(paths):
    """
    Load and preprocess code and CPG data from parquet files.
    
    Args:
        paths: Paths configuration object
        
    Returns:
        DataFrame containing merged code and CPG data
    """
    code_files = sorted(glob(f"{paths.code_dir}/*.parquet"))
    cpg_files = sorted(glob(f"{paths.cpg_dir}/*.parquet"))

    if not code_files or not cpg_files:
        raise ValueError("No parquet files found in the specified directories")

    # Initialize an empty DataFrame to store results
    df = pd.DataFrame()

    # Process each corresponding pair of files from code_dir and cpg_dir
    for code_file, cpg_file in tqdm(zip(code_files, cpg_files), desc="Processing files"):
        # Load code data for the current batch
        code_df = pd.read_parquet(code_file)

        # Load cpg data for the current batch
        cpg_df = pd.read_parquet(cpg_file)

        # Ensure code_df has the required columns
        required_code_cols = ['id', 'code', 'docstring', 'code_tokens']
        if not all(col in code_df.columns for col in required_code_cols):
            raise ValueError(f"Code file {code_file} missing required columns")

        code_df = code_df[required_code_cols]

        # Merge the code and CPG data on the common 'id' column
        merged_df = pd.merge(code_df, cpg_df, on='id', how='left')

        # Filter out rows where CPG data is missing
        merged_df = merged_df.dropna(subset=['cpg'])

        # Ensure the final DataFrame has the required columns
        required_merged_cols = ['id', 'code', 'docstring', 'cpg', 'code_tokens']
        if all(col in merged_df.columns for col in required_merged_cols):
            batch_df = merged_df[required_merged_cols]
            df = pd.concat([df, batch_df], ignore_index=True)

    if df.empty:
        raise ValueError("No valid data found after merging and filtering")

    return df