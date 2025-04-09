import pandas as pd
import openpyxl
import streamlit as st
import instructor
from huggingface_hub import snapshot_download
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field
from typing import List
import json
import logging


# Define the schema for the critique output
class CritiqueOutput(BaseModel):
    job_title_critique: str = Field(..., description="Critique of the job title based on the job description")
    job_title_critique_metric: float = Field(..., description="Give a number, indicating how accurate is the job title based on the job description")

# Initialize Azure client for GPT-4o
azure_client = AzureOpenAI(
    api_key="45KAZBei7wyoEoM0Ac3YGpKlDRvEEtLXrH9kqBkoHMOYmwCqA7YIJQQJ99ALACfhMk5XJ3w3AAAAACOGidRc",
    api_version="2024-12-01-preview",
    azure_endpoint="https://ai-viennas5522ai344254808993.openai.azure.com/"
)

def truncate_text(text: str, max_tokens: int = 4096) -> str:
    """Truncate the input text to fit within the model's context length."""
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

def generate_prompt_template(job_title: str, job_description: str) -> str:
    """Generate the default prompt template that can be modified"""
    return (
        f"Critique the following job posting:\n\n"
        f"Title: {job_title}\n"
        f"Description: {job_description}\n\n"
        "1. Is the job title accurate and relevant to the job description content?\n"
        "2. Measure with a float value 1 to 5 how accurate is the job title.\n"
        "Provide detailed reasoning for your assessment."
    )

def critique_metadata_fields(prompt: str, client, model: str, temperature: float, selected_model: str) -> CritiqueOutput:
    prompt = truncate_text(prompt, max_tokens=4096)

    # Prepare base request parameters
    request_params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Only add temperature if the model DOESN'T explicitly forbid it
    if not model_options[selected_model].get("no_temperature", False):
        request_params["temperature"] = temperature  # Include only for supported models

    if "azure" in model.lower():
        response = azure_client.chat.completions.create(**request_params)
    else:
        request_params["response_format"] = CritiqueOutput
        response = client.beta.chat.completions.parse(**request_params)

    response_text = response.choices[0].message.content.strip()
    return CritiqueOutput(**json.loads(response_text))

@st.cache_resource  # Cache the Chroma client
def initialize_chroma_db(
    hf_repo_id="gthivaios/streamlit-embeddings",
    local_dir="./job_title_chroma_db",
    collection_name="job_title_lexicon", 
    embedding_model_name="all-mpnet-base-v2"
):
    """
    Initializes ChromaDB from Hugging Face repository.
    Args:
        hf_repo_id: Hugging Face repo containing Chroma DB files
        local_dir: Local directory to store downloaded DB
        collection_name: Name of Chroma collection
        embedding_model_name: SentenceTransformer model name
    """
    try:
        # 1. Download Chroma DB files from Hugging Face
        hf_token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
        repo_path = snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            token=hf_token,  # Optional for private repos
            local_dir=local_dir,
            allow_patterns=["*.parquet", "*.sqlite*", "*.bin"]  # Chroma file patterns
        )
        
        # 2. Initialize Chroma client pointing to downloaded files
        chroma_client = chromadb.PersistentClient(path=repo_path)
        
        # 3. Set up embedding function
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # 4. Get existing collection (must match name from uploaded DB)
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn
        )
        
        return collection
        
    except Exception as e:
        logging.error(f"Chroma initialization failed: {str(e)}")
        raise

def load_lexicon_data(collection):
    try:
        results = collection.get()
        documents = results['documents']
        metadatas = results['metadatas']
        ids = results['ids']
        
        processed_lexicon_df = pd.DataFrame(metadatas)
        processed_lexicon_df['combined_label'] = documents
        return processed_lexicon_df
    except Exception as e:
        logging.error(f"Error loading lexicon data: {e}")
        return pd.DataFrame()

def find_top_k_similar(job_title, collection, k, threshold=None):
    """
    Find top-k similar skills using ChromaDB.
    - If `threshold` is None, returns `k` most similar results.
    - Otherwise, only returns results with similarity >= threshold.
    """
    try:
        results = collection.query(
            query_texts=[job_title],  # ChromaDB handles embedding computation internally
            n_results=k
        )
        
        # Process results
        top_k_results = []
        for i in range(len(results['ids'][0])):  # Iterate through available results
            preferred_label = results['metadatas'][0][i]['Occupation Title']
            alternative_label = results['metadatas'][0][i]['Alternative Occupation Titles']
            similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity

            if threshold is None or similarity_score >= threshold:
                top_k_results.append((preferred_label, alternative_label, similarity_score))
            else:
                top_k_results.append((preferred_label, alternative_label, similarity_score))

        return top_k_results
    except Exception as e:
        logging.error(f"Error finding top-k similar job titles for '{job_title}': {e}")
        return []

# Define response schema
class JobTitleClassificationResponse(BaseModel):
    job_title: str = Field(..., description="The job title provided as input.")
    preferredLabel: str = Field(..., description="Primary label of the described job title/occupation.")
    altLabels: List[str] = Field(..., description="Alternative labels or synonyms for the job title/occupation.")
    description: str = Field(..., description="Reasoning for the label or proposed new label.")

def process_job_title_esco(job_title, processed_lexicon_df, client, model, temperature, collection, processed_job_title, k, threshold):
    try:
        if job_title in processed_job_title:
            existing_classification = processed_job_title[job_title]
            return existing_classification

        # Step 1: Find top-k similar labels
        top_k_results = find_top_k_similar(job_title, collection, k, threshold)

        # Check for valid results
        valid_results = [result for result in top_k_results if result[2] >= threshold]
        if valid_results:
            preferred_label = valid_results[0][0]
            alt_labels = processed_lexicon_df.loc[
                processed_lexicon_df["Occupation Title"] == preferred_label, "Alternative Occupation Titles"
            ].values[0].split(";")
            response = JobTitleClassificationResponse(
                job_title=job_title,
                preferredLabel=f"{preferred_label}(lexicon)",
                altLabels=alt_labels,
                description="job title matched directly to an existing occupation cluster (lexicon)."
            )
            processed_job_title[job_title] = response
            return response

        # Step 2: Use LLM for classification
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=temperature,
            response_format=JobTitleClassificationResponse,
            messages=[
                {"role": "system", "content": "You are an expert in job title classification for job postings."},
                {
                    "role": "user",
                    "content": (
                        "You are tasked with classifying the input job title into the most relevant preferred label with the according alternative labels from the provided clusters.\n\n"
                        + "\n".join(
                            [
                                f"{i+1}. Preferred Label: {result[0]}, Alternative Label: {result[1]}, Similarity: {result[2]:.2f}"
                                for i, result in enumerate(top_k_results)
                            ]
                        )
                        + f"\n\job title to classify: '{job_title}'"
                    ),
                },
            ],
        )
        structured_response = completion.choices[0].message.content.strip()
        parsed_response = json.loads(structured_response)
        preferred_label = parsed_response.get("preferredLabel", job_title).strip()
        alt_labels = parsed_response.get("altLabels", [])
        description = parsed_response.get("description", "").strip()

        # Check if the preferred label matches existing labels or alternatives
        is_from_top_k = any(preferred_label == result[0] or preferred_label in result[1].split(";") for result in top_k_results)

        if is_from_top_k:
            for result in top_k_results:
                if preferred_label in result[1].split(";"):
                    response = JobTitleClassificationResponse(
                        job_title=job_title,
                        preferredLabel=f"{result[0]}(model_top_k_from_alt)",
                        altLabels=result[1].split(";"),
                        description=f"The suggested preferred label '{preferred_label}' matches an alternative label. Returning the corresponding preferred label."
                    )
                    processed_job_title[job_title] = response
                    return response
                
        response = JobTitleClassificationResponse(
            job_title=job_title,
            preferredLabel=f"{preferred_label}(model_top_k)",
            altLabels=alt_labels,
            description=description
        )   
        processed_job_title[job_title] = response
        return response
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e}, Response: {structured_response}")
        return JobTitleClassificationResponse(
            job_title=job_title,
            preferredLabel="error",
            altLabels=[],
            description=f"Failed to parse JSON: {e}"
        )
    except Exception as e:
        logging.error(f"Error processing job title '{job_title}': {e}")
        return JobTitleClassificationResponse(
            job_title=job_title,
            preferredLabel="error",
            altLabels=[],
            description=f"Error occurred during processing: {e}"
        )


# Streamlit App
st.title("Golden Data Evaluation Tool")



df1 = pd.read_excel('./Data/golden_dataset_tourism.xlsx')

# Initialize ChromaDB and lexicon data
@st.cache_resource
def load_normalization_resources():
    try:
        collection = initialize_chroma_db()
        processed_lexicon_df = load_lexicon_data(collection)
        return {
            "collection": collection,
            "lexicon_df": processed_lexicon_df
        }
    except Exception as e:
        st.error(f"Failed to load normalization resources: {str(e)}")
        return None


# Initialize session state
if 'show_evaluation' not in st.session_state:
    st.session_state.show_evaluation = False
if 'show_normalization' not in st.session_state:
    st.session_state.show_normalization = False
if 'current_id' not in st.session_state:
    st.session_state.current_id = 1
if 'custom_prompt' not in st.session_state:
    st.session_state.custom_prompt = ""
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'normalization_results' not in st.session_state:
    st.session_state.normalization_results = None

# Input for job posting ID
job_id = st.number_input(
    "Enter Job Posting ID", 
    min_value=1, 
    max_value=df1['id'].max(), 
    value=st.session_state.current_id
)

# Update current ID and reset all related states when ID changes
if job_id != st.session_state.current_id:
    st.session_state.current_id = job_id
    st.session_state.show_evaluation = False
    st.session_state.show_normalization = False
    st.session_state.normalization_results = None
    st.session_state.current_prompt = ""
    st.rerun()  # Optional: forces UI to refresh immediately

# Filter job posting by current ID
job_posting = df1[df1["id"] == st.session_state.current_id]

if not job_posting.empty:
    # Display job details
    st.subheader("Job Posting Details")
    st.write(f"**Title:** {job_posting['title'].values[0]}")
    st.write(f"**Description:**")
    st.text_area("", value=job_posting['job_description_cleaned'].values[0], height=300, key=f"desc_{st.session_state.current_id}")
    st.write(f"**Occupation ISCO Code:** {job_posting['occupation_isco_code'].values[0]}")

    # Toggle evaluation controls
    if st.button("Run Evaluation"):
        st.session_state.show_evaluation = not st.session_state.show_evaluation
        # Generate default prompt when first opening evaluation
        st.session_state.current_prompt = generate_prompt_template(
            job_posting['title'].values[0],
            job_posting['job_description_cleaned'].values[0]
        )

    # Evaluation controls
    if st.session_state.show_evaluation:
        st.subheader("Evaluation Settings")
        
        model_options = {
            "deepseek-r1-distill-qwen-14b": {
                "base_url": "http://150.140.28.122:5010/v1",
                "api_key": "xxxxxx",
                "client_type": "openai"
            },
            "gpt-4o-azure": {
                "client_type": "azure",
                "model_name": "gpt-4o"  # Azure deployment name
            },
            "o1-azure": {
            "client_type": "azure",
            "model_name": "o1",  # Azure deployment name
            "no_temperature": True  # Explicitly mark models that reject temperature
            }
        }
        
        selected_model = st.selectbox(
            "Select Model",
            list(model_options.keys()),
            index=1  # Default to GPT-4o
        )
        
        # Only show temperature slider if the model supports it
        if not model_options[selected_model].get("no_temperature", False):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        else:
            temperature = None  # Will be ignored
            st.info("Note: o1-reasoning doesn't support temperature control")

        # Custom prompt editor - now will show the current job's prompt
        st.subheader("Prompt Customization")
        st.session_state.custom_prompt = st.text_area(
            "Edit the evaluation prompt:",
            value=st.session_state.current_prompt,
            height=300,
            key=f"prompt_{st.session_state.current_id}"  # Unique key per job
        )

        if st.button("Execute Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    client_config = model_options[selected_model]
                    
                    if client_config["client_type"] == "azure":
                        client = azure_client
                        model_name = client_config["model_name"]
                    else:
                        client = instructor.from_openai(OpenAI(
                            base_url=client_config["base_url"],
                            api_key=client_config["api_key"],
                            timeout=520,
                        ), mode=instructor.Mode.JSON_SCHEMA)
                        model_name = selected_model
                    
                    critique = critique_metadata_fields(
                        prompt=st.session_state.current_prompt,
                        client=client,
                        model=model_name,
                        temperature=temperature,
                        selected_model=selected_model
                    )

                    # Display results
                    st.subheader("Evaluation Results")
                    st.write(f"**Job Title Critique:** {critique.job_title_critique}")
                    st.write(f"**Critique Metric:** {critique.job_title_critique_metric:.2f}")
                    
                    with st.expander("Show Raw Response"):
                        st.json(critique.model_dump())
                        
                    # Enable normalization button after successful evaluation
                    st.session_state.show_normalization = True

                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
    # Normalization section
    if st.session_state.show_normalization:
        st.subheader("Normalization Settings")
        # Model selection for normalization
        norm_model_options = {
            "deepseek-r1-distill-qwen-14b": {
                "base_url": "http://150.140.28.122:5010/v1",
                "api_key": "xxxxxx",
                "client_type": "openai"
            },
            "gpt-4o-azure": {
                "client_type": "azure",
                "model_name": "gpt-4o"
            },
            "o1-azure": {
            "client_type": "azure",
            "model_name": "o1",
            "no_temperature": True  # Explicitly mark models that reject temperature
            }
        }
        
        selected_norm_model = st.selectbox(
            "Select Normalization Model",
            list(norm_model_options.keys()),
            index=1,
            key="norm_model_select"
        )

        # Normalization parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            if not norm_model_options[selected_norm_model].get("no_temperature", False):
                norm_temperature = st.slider(
                    "Temperature", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.2, 
                    step=0.1,
                    key="norm_temp"
                )
            else:
                norm_temperature = None
        with col2:
            k_value = st.slider(
                "Top K matches", 
                min_value=1, 
                max_value=10, 
                value=4,
                key="k_value"
            )
        with col3:
            similarity_threshold = st.slider(
                "Similarity threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6, 
                step=0.05,
                key="sim_threshold"
            )       

        # In your Streamlit app initialization:
        norm_resources = load_normalization_resources()
        if norm_resources is None:
            st.error("Critical error - failed to load normalization resources")
            st.stop()
        if st.button("Run Normalization"):
            with st.spinner("Running job title normalization..."):
                try:
                    job_title = job_posting['title'].values[0]
                    # Initialize the appropriate client
                    norm_client_config = norm_model_options[selected_norm_model]
                    if norm_client_config["client_type"] == "azure":
                        norm_client = azure_client
                        norm_model_name = norm_client_config["model_name"]
                    else:
                        norm_client = instructor.from_openai(OpenAI(
                            base_url=norm_client_config["base_url"],
                            api_key=norm_client_config["api_key"],
                            timeout=520,
                        ), mode=instructor.Mode.JSON_SCHEMA)
                        norm_model_name = selected_norm_model
                    response = process_job_title_esco(
                        job_title, 
                        processed_lexicon_df=norm_resources["lexicon_df"], 
                        client=norm_client, 
                        model=norm_model_name, 
                        temperature=norm_temperature, 
                        collection=norm_resources["collection"], 
                        processed_job_title={},  # Empty dict since we're not caching across runs
                        k=k_value, 
                        threshold=similarity_threshold
                    )
                    # Properly handle the response object
                    # Handle the response properly
                    processed_lexicon_df=norm_resources["lexicon_df"]
                    occupation_title = response.preferredLabel.split('(')[0].strip() if isinstance(response.preferredLabel, str) else job_title
                    lexicon_row = processed_lexicon_df[processed_lexicon_df['Occupation Title'].str.lower() == occupation_title.lower()]
                    
                    esco_code = lexicon_row['ESCO_Code'].iloc[0] if not lexicon_row.empty else "Not found"
                    isco_code = lexicon_row['ISCO_Code'].iloc[0] if not lexicon_row.empty else "Not found"
                    
                    # Store results
                    st.session_state.normalization_results = {
                        "original_title": job_title,
                        "normalized_title": occupation_title,
                        "esco_code": esco_code,
                        "isco_code": isco_code,
                        "alternative_labels": response.altLabels,
                        "description": response.description,
                        "parameters": {
                            "model": selected_norm_model,
                            "temperature": norm_temperature,
                            "top_k": k_value,
                            "threshold": similarity_threshold
                        }
                    }

                    # Display results
                    st.subheader("Normalization Results")
                    st.write(f"**Original Title:** {job_title}")
                    st.write(f"**Normalized Title:** {occupation_title}")
                    st.write(f"**ESCO Code:** {esco_code}")
                    st.write(f"**ISCO Code:** {isco_code}")
                    st.write(f"**Alternative Labels:** {', '.join(response.altLabels) if response.altLabels else 'None'}")
                    st.write(f"**Description:** {response.description}")
                    
                    # Show parameters used
                    with st.expander("Normalization Parameters"):
                        st.json(st.session_state.normalization_results["parameters"])
                    
                except Exception as e:
                    st.error(f"Normalization failed: {str(e)}")
                    st.session_state.normalization_results = None

        # Show previous normalization results if available
        if st.session_state.normalization_results:
            st.subheader("Previous Normalization Results")
            st.json(st.session_state.normalization_results)                               

else:
    st.warning("No job posting found with the given ID.")