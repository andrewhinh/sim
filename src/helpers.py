import json
import os
import tempfile
from pathlib import Path, PurePosixPath

import modal
import numpy as np
import plotly.graph_objects as go
import requests
import torch
import umap
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from rerankers import Reranker
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from unstructured.partition.pdf import partition_pdf
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from db.models import DataPointBase, PaperBase, SearchParamsBase
from utils import (
    APP_NAME,
    MINUTES,
    PYTHON_VERSION,
    SECRETS,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# -----------------------------------------------------------------------------

# Modal
CUDA_VERSION = "12.8.0"
FLAVOR = "devel"
OS = "ubuntu22.04"
TAG = f"nvidia/cuda:{CUDA_VERSION}-{FLAVOR}-{OS}"

PRETRAINED_VOLUME = f"{APP_NAME}-pretrained"
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {
    f"/{PRETRAINED_VOLUME}": modal.Volume.from_name(
        PRETRAINED_VOLUME, create_if_missing=True
    ),
}
if modal.is_local():
    PRETRAINED_VOL_PATH = None
else:
    PRETRAINED_VOL_PATH = Path(f"/{PRETRAINED_VOLUME}")


small_llm_name = "Qwen/Qwen3-1.7B"
small_llm_enforce_eager = False
small_llm_max_num_seqs = 1 if modal.is_local() else 1024
small_llm_trust_remote_code = True
small_llm_max_model_len = 32768
small_llm_enable_chunked_prefill = True
small_llm_max_num_batched_tokens = small_llm_max_model_len

medium_llm_name = "Qwen/Qwen3-4B"
medium_llm_enforce_eager = False
medium_llm_max_num_seqs = 1 if modal.is_local() else 512
medium_llm_trust_remote_code = True
medium_llm_max_model_len = 32768
medium_llm_enable_chunked_prefill = True
medium_llm_max_num_batched_tokens = medium_llm_max_model_len

large_llm_name = medium_llm_name if modal.is_local() else "Qwen/Qwen3-30B-A3B-FP8"
large_llm_enforce_eager = False
large_llm_max_num_seqs = 1 if modal.is_local() else 4
large_llm_trust_remote_code = True
large_llm_max_model_len = 32768
large_llm_enable_chunked_prefill = True
large_llm_max_num_batched_tokens = large_llm_max_model_len

reranker_name = "answerdotai/answerai-colbert-small-v1"
reranker_batch_size = 16


def download_models():
    for llm_name in [
        small_llm_name,
        medium_llm_name,
        large_llm_name,
    ]:
        snapshot_download(
            repo_id=llm_name,
            local_dir=PRETRAINED_VOL_PATH,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        )


GPU_IMAGE = (
    modal.Image.from_registry(TAG, add_python=PYTHON_VERSION)
    .apt_install(
        "git", "poppler-utils", "tesseract-ocr", "ffmpeg", "libsm6", "libxext6"
    )
    .pip_install(
        "accelerate>=1.6.0",
        "flashinfer-python==0.2.2.post1",
        "hf-transfer>=0.1.9",
        "huggingface-hub>=0.30.2",
        "ninja>=1.11.1.4",  # required to build flash-attn
        "packaging>=24.2",  # required to build flash-attn
        "plotly>=6.0.1",
        "rerankers>=0.9.1.post1",
        "sqlalchemy>=2.0.40",
        "sqlmodel>=0.0.24",
        "torch>=2.6.0",
        "tqdm>=4.67.1",
        "transformers>=4.51.1",
        "umap-learn>=0.5.7",
        "unstructured[local-inference,pdf]>=0.17.2",
        "vllm>=0.8.3",
        "wheel>=0.45.1",  # required to build flash-attn
    )
    .run_commands("pip install flash-attn==2.7.4.post1 --no-build-isolation")
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .run_function(
        download_models,
        secrets=SECRETS,
        volumes=VOLUME_CONFIG,
    )
    .add_local_python_source("_remote_module_non_scriptable", "src", "db", "utils")
)
app = modal.App(f"{APP_NAME}-helpers")

# -----------------------------------------------------------------------------


@app.function(
    image=GPU_IMAGE,
    cpu=1,  # cores
    memory=1024,  # MB
    gpu="l40s:1",  # 1 L40s GPU
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=5 * MINUTES,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=small_llm_max_num_seqs)
def check_question_threaded(question: str) -> list[str] | None:
    max_num_suggestions = 3

    small_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=small_llm_name,
        tokenizer=small_llm_name,
        enforce_eager=small_llm_enforce_eager,
        max_num_seqs=small_llm_max_num_seqs,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=small_llm_trust_remote_code,
        max_model_len=small_llm_max_model_len,
        enable_chunked_prefill=small_llm_enable_chunked_prefill,
        max_num_batched_tokens=small_llm_max_num_batched_tokens,
        guided_decoding_backend="xgrammar",
    )

    system_prompt = """
        You are a specialized research methodology expert with extensive experience evaluating academic research questions. 
        Your ONLY task is to assess if a research question meets specific quality criteria and return a structured JSON response in the EXACT format required. 
        Do not deviate from the response format under any circumstances.
    """
    user_prompt = f"""
        Research question to evaluate: "{question}"

        Carefully evaluate this question against the following criteria:
        
        1. CLARITY: Is the question specific, focused, and well-defined? Does it clearly articulate what is being studied?
        2. FEASIBILITY: Can this question reasonably be answered with typical research resources and methods?
        3. ANALYTICAL DEPTH: Does this question encourage analysis rather than yielding a simple yes/no answer?
        4. ETHICAL SOUNDNESS: Does this question generally follow ethical research principles?

        Response Instructions:
        - If the question is generally sound as a research question, even if it could use minor improvements, respond EXACTLY:
        {{
            "suggestions": null,
        }}
        - Only if the question has significant issues that would make it difficult to research effectively, identify up to THREE specific improvement areas from this list:
        - Specificity: Clarify the research question to make it more specific and focused
        - Scope: Narrow or broaden the scope appropriately for the research context
        - Population: Define the target population or subjects more clearly
        - Variables: Specify variables or factors to be examined
        - Timeframe: Add a clear timeframe for the research
        - Resources: Consider available research resources and constraints
        - Methods: Specify appropriate research methods or approaches
        - Sample: Define the sample size or sampling approach
        - Causality: Address issues with causal relationships or correlations
        - Comparison: Add comparative elements if needed
        - Factors: Identify relevant factors or dimensions for analysis
        - Ethics: Address ethical research concerns
        - Consent: Consider participant consent and autonomy issues
        - Privacy: Address data privacy and confidentiality concerns
        - Return your response in EXACTLY this JSON format with NO additional text:
        {{
            "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
        }}
        Note that each suggestion should be a SINGLE word or phrase, not a full sentence.
    """

    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p = 0
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 2048

    class CheckQuestion(BaseModel):
        suggestions: list[str] | None

    guided_decoding = GuidedDecodingParams(json=CheckQuestion.model_json_schema())
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
        guided_decoding=guided_decoding,
    )

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    ]
    outputs = small_llm.chat(conversations, sampling_params, use_tqdm=True)
    generated_text = outputs[0].outputs[0].text.split("</think>")[-1].strip()
    try:
        response_dict = json.loads(generated_text)
        response = CheckQuestion.model_validate(response_dict)
        suggestions = response.suggestions
        if suggestions is not None:
            suggestions = suggestions[:max_num_suggestions]
    except Exception as e:
        print(
            f"Warning: Skipping suggestions for question {question} due to error: {e}"
        )
        suggestions = []
    return suggestions


@app.function(
    image=GPU_IMAGE,
    cpu=1,
    memory=1024,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=5 * MINUTES,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=small_llm_max_num_seqs)
def modify_question_threaded(question: str, suggestion: str) -> str:
    small_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=small_llm_name,
        tokenizer=small_llm_name,
        enforce_eager=small_llm_enforce_eager,
        max_num_seqs=small_llm_max_num_seqs,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=small_llm_trust_remote_code,
        max_model_len=small_llm_max_model_len,
        enable_chunked_prefill=small_llm_enable_chunked_prefill,
        max_num_batched_tokens=small_llm_max_num_batched_tokens,
    )

    system_prompt = """
        You are a specialized research methodology expert who transforms flawed research questions into high-quality ones.
        Your ONLY task is to improve a given question based on a specific suggestion.
        Your output should ONLY be the improved question with no additional explanations.
    """
    user_prompt = f"""
        Original research question: "{question}"

        Improvement area to address: "{suggestion}"
        
        Task: Rewrite the question to incorporate improvements related to "{suggestion}" while maintaining the core research focus.

        A high-quality research question must satisfy these criteria:
        1. CLARITY: Specific, focused, well-defined, and articulates exactly what will be studied
        2. FEASIBILITY: Answerable with reasonable resources, time constraints, and available methods
        3. ANALYTICAL DEPTH: Requires critical analysis rather than simple yes/no answers or facts
        4. ETHICAL SOUNDNESS: Follows ethical research principles for participants and data

        Here's how to improve the question based on the suggestion:
        
        - If suggestion is "Specificity": Clarify the research question to make it more specific and focused
        - If suggestion is "Scope": Narrow or broaden the scope appropriately for the research context
        - If suggestion is "Population": Define the target population or subjects more clearly
        - If suggestion is "Variables": Specify the variables, factors, or phenomena being examined
        - If suggestion is "Timeframe": Add a specific time period for the research
        - If suggestion is "Resources": Consider practical limitations and available resources
        - If suggestion is "Methods": Specify or clarify appropriate research methods
        - If suggestion is "Sample": Define the sample size or sampling approach
        - If suggestion is "Causality": Clarify causal relationships or correlations being studied
        - If suggestion is "Comparison": Add comparative elements if appropriate
        - If suggestion is "Factors": Identify relevant factors or dimensions to consider
        - If suggestion is "Ethics": Address ethical research concerns
        - If suggestion is "Consent": Consider participant consent and autonomy issues
        - If suggestion is "Privacy": Address data privacy and confidentiality concerns
        
        IMPORTANT: Return ONLY the improved research question as a single sentence or paragraph. Do not include any explanations, bullet points, or meta-commentary.
    """

    temperature = 0.7
    top_p = 0.8
    top_k = 20
    min_p = 0
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 2048

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
    )

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    ]
    outputs = small_llm.chat(conversations, sampling_params, use_tqdm=True)
    generated_text = outputs[0].outputs[0].text.split("</think>")[-1].strip()
    return generated_text


def query_to_papers(
    query: str,
    min_results: int,
    max_results: int,
) -> dict[str, list[dict] | str]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    url += f"?query={query}"
    url += "&fields=paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles,authors"
    url += "&openAccessPdf"

    r = requests.get(url)
    try:
        r.raise_for_status()
        r = r.json()
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        return {"error": error_msg}

    papers = []
    while True:
        if "data" not in r:
            break
        for paper_data in r["data"]:
            paper = PaperBase(
                paper_id=paper_data["paperId"],
                corpus_id=paper_data["corpusId"],
                external_ids=paper_data["externalIds"],
                url=paper_data["url"],
                title=paper_data["title"],
                abstract=paper_data["abstract"],
                venue=paper_data["venue"],
                publication_venue=paper_data["publicationVenue"],
                year=paper_data["year"],
                reference_count=paper_data["referenceCount"],
                citation_count=paper_data["citationCount"],
                influential_citation_count=paper_data["influentialCitationCount"],
                is_open_access=paper_data["isOpenAccess"],
                open_access_pdf=paper_data["openAccessPdf"],
                fields_of_study=paper_data["fieldsOfStudy"],
                s2_fields_of_study=paper_data["s2FieldsOfStudy"],
                publication_types=paper_data["publicationTypes"],
                publication_date=paper_data["publicationDate"],
                journal=paper_data["journal"],
                citation_styles=paper_data["citationStyles"],
                authors=paper_data["authors"],
            )
            papers.append(paper.model_dump())
        if "token" not in r or r["token"] is None or len(papers) >= max_results:
            break

        r = requests.get(f"{url}&token={r['token']}")
        try:
            r.raise_for_status()
            r = r.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error: {e}"
            print(error_msg)
            return {"error": error_msg}

    if len(papers) < min_results:
        return {"error": f"Not enough results: {len(papers)} < {min_results}"}
    return {"success": papers[:max_results]}


@app.function(
    image=GPU_IMAGE,
    cpu=1,
    memory=1024,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=medium_llm_max_num_seqs)
def question_to_query_and_papers_threaded(
    question: str,
    min_results: int = 10,
    max_results: int = 100,
) -> tuple[dict, list[dict]]:  # search params, papers
    medium_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=medium_llm_name,
        tokenizer=medium_llm_name,
        enforce_eager=medium_llm_enforce_eager,
        max_num_seqs=medium_llm_max_num_seqs,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=medium_llm_trust_remote_code,
        max_model_len=medium_llm_max_model_len,
        enable_chunked_prefill=medium_llm_enable_chunked_prefill,
        max_num_batched_tokens=medium_llm_max_num_batched_tokens,
    )

    system_prompt = """
        You are a specialized academic search expert who transforms research questions into an effective search query for the Semantic Scholar Paper bulk search API.
        Your output should ONLY be the search query with no additional explanations.
    """
    user_prompt = f"""
        Original research question: "{question}"
        
        Task: Convert this research question into an effective search query for the Semantic Scholar Paper bulk search API.

        query:
        - Will be matched against the paper's title and abstract.
        - All terms are stemmed in English.
        - By default all terms in the query must be present in the paper.
        - The match query supports the following syntax:
            - + for AND operation
            - | for OR operation
            - - negates a term
            - " collects terms into a phrase
            - * can be used to match a prefix
            - ( and ) for precedence
            - ~N after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
            - ~N after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)
        - examples:
            - fish ladder matches papers that contain "fish" and "ladder"
            - fish -ladder matches papers that contain "fish" but not "ladder"
            - fish | ladder matches papers that contain "fish" or "ladder"
            - "fish ladder" matches papers that contain the phrase "fish ladder"
            - (fish ladder) | outflow matches papers that contain "fish" and "ladder" OR "outflow"
            - fish~ matches papers that contain "fish", "fist", "fihs", etc.
            - "fish ladder"~3 matches papers that contain the phrase "fish ladder" or "fish is on a ladder"

        Ensure your query is broad enough to return at least {min_results} results.
        Do so by using the minimum number of terms and broad operators.

        Return ONLY the search query without additional text.

        If your parameters do not return at least {min_results} results, your query is too narrow.
        To broaden your query, try:
        - using more '|' (OR) operators
        - using less '+' (AND) operators
        - removing unnecessary query terms
        - etc.
    """

    temperature = 0.6
    top_p = 0.95
    top_k = 20
    min_p = 0
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 2048

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
    )

    # retries in case no results
    papers = []
    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
    ]
    num_retries = 3
    for i in range(num_retries):
        outputs = medium_llm.chat(conversations, sampling_params, use_tqdm=True)
        generated_text = outputs[0].outputs[0].text.split("</think>")[-1].strip()

        search_params = SearchParamsBase.model_validate({"query": generated_text})
        search_params_dict = search_params.model_dump()
        result = query_to_papers(
            **search_params_dict, min_results=min_results, max_results=max_results
        )

        if "success" in result and len(result["success"]) >= min_results:
            papers = result["success"]
            break
        else:
            if i == num_retries - 1:
                print(
                    f"Warning: Skipping search for question {question} due to {result['error']}"
                )
                return {}, []
            else:
                conversations[0].append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": generated_text}],
                    }
                )
                conversations[0].append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (f"Error: {result['error']}"),
                            },
                        ],
                    }
                )

    return search_params_dict, papers


@app.function(
    image=GPU_IMAGE,
    cpu=1,
    memory=1024,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=10 * MINUTES,
    scaledown_window=10 * MINUTES,
)  # called by screen_papers_threaded
def paper_to_chunks(paper: dict) -> list[str] | None:
    try:
        open_access_pdf = paper.get("open_access_pdf", {})
        if not open_access_pdf:
            raise ValueError("Paper has no open access PDF")
        pdf_url = open_access_pdf.get("url", "")
        if not pdf_url:
            raise ValueError("Paper has no open access PDF URL")

        # check pdf
        resp = requests.get(pdf_url)
        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            raise ValueError(f"Paper has invalid Content-Type: {content_type}")
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(resp.content)
            tmp_file.flush()

            with open(tmp_file.name, "rb") as f:
                magic = f.read(4)
            if magic != b"%PDF":
                raise ValueError("Paper is not a valid PDF")

            chunk_elements = partition_pdf(
                tmp_file.name,
                url=None,
                chunking_strategy="by_title",
                max_characters=large_llm_max_model_len,  # hard limit, keeping in mind 1) 3 retries in mind and 2) ~4x chars/token
                new_after_n_chars=large_llm_max_model_len // 1.5,  # soft limit
            )

        chunks = [element.text for element in chunk_elements]
        return chunks
    except Exception as e:
        print(
            f"Warning: Skipping paper {paper.get('paper_id', 'unknown')} due to error: {e}"
        )
        return None


@app.function(
    image=GPU_IMAGE,
    cpu=0.125,
    memory=128,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=10 * MINUTES,
    scaledown_window=10 * MINUTES,
)
def screen_papers_threaded(
    papers: list[dict],
) -> list[dict]:
    if modal.is_local():
        chunk_lists = list(
            tqdm(thread_map(paper_to_chunks.local, papers), desc="Processing papers")
        )
    else:
        chunk_lists = list(paper_to_chunks.map(papers))

    screened_papers = []
    for paper, chunks in zip(papers, chunk_lists):
        if chunks is not None and paper["abstract"]:
            paper["chunks"] = chunks
            screened_papers.append(paper)
    return screened_papers


@app.function(
    image=GPU_IMAGE,
    cpu=1,
    memory=1024,
    gpu="h100:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=15 * MINUTES,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=large_llm_max_num_seqs)
def extract_data_points_threaded(question: str, papers: list[dict]) -> list[list[dict]]:
    large_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=large_llm_name,
        tokenizer=large_llm_name,
        enforce_eager=large_llm_enforce_eager,
        max_num_seqs=large_llm_max_num_seqs,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=large_llm_trust_remote_code,
        max_model_len=large_llm_max_model_len,
        enable_chunked_prefill=large_llm_enable_chunked_prefill,
        max_num_batched_tokens=large_llm_max_num_batched_tokens,
        guided_decoding_backend="xgrammar",
    )

    system_prompt = """
        You are a specialized research methodology expert that extracts key data points from academic papers with a research question in mind.
        Your ONLY task is to extract key data points from academic papers and return a structured JSON response in the EXACT format required. 
        Do not deviate from the response format under any circumstances.
        Your output should ONLY be the data points with no additional explanations.
    """
    user_prompt = """
        Task: Given this research question:

        {question}

        and the following chunk of text from an academic paper below, extract all relevant data points and format them as per the schema described.

        Paper content:
        {paper}

        Use the following schema for each data point:
        - name: string (the name of the metric or measurement)
        - value: number (the numeric result or measurement)
        - unit: string or null (unit of measurement)
        - excerpt: string or null (text snippet where the data point occurs)

        The final output must be a JSON array of objects, for example:
        [
            {{
                "name": "average response time",
                "value": 123.45,
                "unit": "ms",
                "excerpt": "The mean response time was 123.45 ms across all trials."
            }},
            {{
                "name": "sample size",
                "value": 200,
                "unit": "participants",
                "excerpt": "A total of 200 participants were recruited for the study."
            }}
        ]

        Return only a JSON array of objects conforming to the schema with no additional text.

        If there are no data points, return an empty array.
    """

    temperature = 0.6
    top_p = 0.95
    top_k = 20
    min_p = 0
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 8192
    guided_decoding = GuidedDecodingParams(
        json={"type": "array", "items": DataPointBase.model_json_schema()}
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
        guided_decoding=guided_decoding,
    )

    data_points_all: list[list[dict]] = []
    for paper in papers:
        paper_points: list[dict] = []
        for idx, chunk in enumerate(paper["chunks"]):
            conversation = [
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt.format(
                                    question=question, paper=chunk
                                ),
                            }
                        ],
                    },
                ]
            ]

            max_retries = 3
            for attempt in range(max_retries):
                outputs = large_llm.chat(conversation, sampling_params, use_tqdm=True)
                generated_text = (
                    outputs[0].outputs[0].text.split("</think>")[-1].strip()
                )

                try:
                    paper_points.extend(
                        [
                            DataPointBase.model_validate(item).model_dump()
                            for item in json.loads(generated_text)
                        ]
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(
                            f"Warning: Skipping data points extraction for paper_id {paper['paper_id']}, chunk {idx} due to {e}"
                        )
                    else:
                        conversation[0].append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": generated_text}],
                            }
                        )
                        conversation[0].append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"""
                                            Error: {e} for paper_id {paper['paper_id']}, chunk {idx} 
                                            Please fix the error and return a valid JSON array of objects conforming to the schema.
                                        """,
                                    },
                                ],
                            }
                        )

        data_points_all.append(paper_points)

    return data_points_all


@app.function(
    image=GPU_IMAGE,
    cpu=1,
    memory=1024,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=10 * MINUTES,
    scaledown_window=5 * MINUTES,
)
def rerank_and_visualize_papers_threaded(
    question: str, papers: list[dict]
) -> tuple[list[dict], dict]:  # papers, visualization as plotly figure json
    ranker = Reranker(
        reranker_name,
        model_type="colbert",
        verbose=0,
        dtype=torch.bfloat16,
        device="cuda",
        batch_size=reranker_batch_size,
        model_kwargs={"cache_dir": PRETRAINED_VOL_PATH},
    )

    str_papers = [str(PaperBase(**paper)) for paper in papers]

    results = ranker.rank(query=question, docs=str_papers)
    top_k_idxs = [doc.doc_id for doc in results.top_k(len(papers))]
    reranked_papers = [papers[i] for i in top_k_idxs]

    # umap of embeddings into 3D
    paper_embeddings = (
        ranker._to_embs(ranker._document_encode(str_papers))
        .to(torch.float32)
        .cpu()
        .numpy()
    )  # (N_docs, seq_len, hidden_dim)
    paper_embeddings = np.concatenate(
        [paper_embeddings.mean(axis=1), paper_embeddings.max(axis=1)], axis=1
    )  # concatenate mean and max token embeddings -> (N_docs, 2*hidden_dim)

    umap_inst = umap.UMAP(
        n_neighbors=min(15, len(papers) - 1), n_components=3, metric="cosine"
    )
    data_transform = umap_inst.fit_transform(paper_embeddings)

    paper_scatter = go.Scatter3d(
        x=data_transform[:, 0],
        y=data_transform[:, 1],
        z=data_transform[:, 2],
        mode="markers",
        marker=dict(color="cyan", size=4),
        text=[
            f"Rank: {i}\n{str(PaperBase(**paper))}" for i, paper in enumerate(papers)
        ],
        hoverinfo="text",
    )

    connectivity = umap_inst.graph_
    edges_x, edges_y, edges_z = [], [], []
    connectivity_coo = connectivity.tocoo()

    for i, j, v in zip(
        connectivity_coo.row, connectivity_coo.col, connectivity_coo.data
    ):
        # Only draw edges for strong connections (adjust threshold as needed)
        if v > 0.1 and i < j:  # Avoid duplicate edges (i<j) and self-connections
            edges_x.extend([data_transform[i, 0], data_transform[j, 0], None])
            edges_y.extend([data_transform[i, 1], data_transform[j, 1], None])
            edges_z.extend([data_transform[i, 2], data_transform[j, 2], None])

    edges_trace = go.Scatter3d(
        x=edges_x,
        y=edges_y,
        z=edges_z,
        mode="lines",
        line=dict(color="rgba(100, 100, 100, 0.2)", width=1),
        hoverinfo="none",
    )

    fig = go.Figure(data=[edges_trace, paper_scatter])
    fig.update_layout(
        hovermode="closest",
        scene=dict(
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(showticklabels=False, title=""),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return reranked_papers, fig.to_json()


# -----------------------------------------------------------------------------


@app.function(
    image=GPU_IMAGE,
    cpu=0.125,
    memory=128,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=60 * MINUTES,
)
def main():
    default_question = (
        "What is the relationship between climate change and global health?"
    )

    # suggestions = (
    #     check_question_threaded.local(default_question)
    #     if modal.is_local()
    #     else check_question_threaded.remote(default_question)
    # )
    # print(suggestions)

    # modified_question = (
    #     modify_question_threaded.local(default_question, suggestions[0])
    #     if modal.is_local()
    #     else modify_question_threaded.remote(default_question, suggestions[0])
    # )
    # print(modified_question)

    modified_question = default_question

    search_params, papers = (
        question_to_query_and_papers_threaded.local(modified_question)
        if modal.is_local()
        else question_to_query_and_papers_threaded.remote(modified_question)
    )
    print(search_params)
    print(len(papers))
    print(papers[0] if papers else None)

    screened_papers = (
        screen_papers_threaded.local(papers)
        if modal.is_local()
        else screen_papers_threaded.remote(papers)
    )
    print(len(screened_papers))
    print(screened_papers[0] if screened_papers else None)

    # data_points = (
    #     extract_data_points_threaded.local(modified_question, screened_papers)
    #     if modal.is_local()
    #     else extract_data_points_threaded.remote(modified_question, screened_papers)
    # )
    # print(len(data_points))
    # print(len(data_points[0]) if data_points else None)
    # # Print first non-null data point from all papers
    # found_point = False
    # for paper_points in data_points:
    #     for dp in paper_points:
    #         if dp and dp.get("name") and dp.get("value") is not None:
    #             print(
    #                 f"First data point: name={dp.get('name')}, value={dp.get('value')}, "
    #                 f"unit={dp.get('unit')}, excerpt={dp.get('excerpt')}"
    #             )
    #             found_point = True
    #             break
    #     if found_point:
    #         break
    # if not found_point:
    #     print("No non-null data points found in any paper")

    reranked_papers, visualization = (
        rerank_and_visualize_papers_threaded.local(modified_question, screened_papers)
        if modal.is_local()
        else rerank_and_visualize_papers_threaded.remote(
            modified_question, screened_papers
        )
    )
    print(len(reranked_papers))
    print(reranked_papers[0] if reranked_papers else None)
    print(visualization)


@app.local_entrypoint()
def main_modal():
    main.remote()


if __name__ == "__main__":
    download_models()
    main.local()
