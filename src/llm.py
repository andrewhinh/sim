import json
import os
from pathlib import Path, PurePosixPath

import modal
import requests
from pydantic import BaseModel

from db.models import PaperBase, SearchParamsBase
from utils import (
    APP_NAME,
    CPU,
    MEM,
    MINUTES,
    PYTHON_VERSION,
    SECRETS,
)

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


small_llm_name = "Qwen/Qwen2.5-3B-Instruct-AWQ"
medium_llm_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
reranker_name = "answerdotai/answerai-colbert-small-v1"


def download_models():
    snapshot_download(
        repo_id=small_llm_name,
        local_dir=PRETRAINED_VOL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    snapshot_download(
        repo_id=medium_llm_name,
        local_dir=PRETRAINED_VOL_PATH,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    snapshot_download(
        repo_id=reranker_name,
        local_dir=PRETRAINED_VOL_PATH,
        ignore_patterns=["*.pt", "*.bin"],
    )  # using safetensors


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
        "rerankers[transformers]>=0.9.1.post1",
        "sqlalchemy>=2.0.40",
        "sqlmodel>=0.0.24",
        "torch>=2.6.0",
        "tqdm>=4.67.1",
        "transformers>=4.51.1",
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
app = modal.App(f"{APP_NAME}-llm")

# -----------------------------------------------------------------------------

with GPU_IMAGE.imports():
    import torch
    from huggingface_hub import snapshot_download
    from rerankers import Reranker
    from unstructured.partition.pdf import partition_pdf
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class CheckQuestion(BaseModel):
    suggestions: list[str] | None


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=2)
def check_question_threaded(question: str) -> list[str] | None:
    max_num_suggestions = 3

    small_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=small_llm_name,
        tokenizer=small_llm_name,
        enforce_eager=False,
        max_num_seqs=1,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        max_model_len=32768,
    )

    system_prompt = """
    You are a specialized research methodology expert with extensive experience evaluating academic research questions. 
    Your ONLY task is to assess if a research question meets specific quality criteria and return a structured JSON response in the EXACT format required. 
    Do not deviate from the response format under any circumstances.
    """
    user_prompt = f"""
    Research question to evaluate: "{question}"

    Carefully evkaluate this question against the following criteria:
    
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

        Return your response in EXACTLY this JSON format with NO additional text:
        {{
            "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
        }}

        Note that each suggestion should be a SINGLE word or phrase, not a full sentence.
    """

    temperature = 0.0
    top_p = 0.8
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 512
    guided_decoding = GuidedDecodingParams(json=CheckQuestion.model_json_schema())
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
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
    generated_text = outputs[0].outputs[0].text.strip()
    response_dict = json.loads(generated_text)
    response = CheckQuestion.model_validate(response_dict)
    suggestions = response.suggestions
    if suggestions is not None:
        suggestions = suggestions[:max_num_suggestions]
    return suggestions


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=2)
def modify_question_threaded(question: str, suggestion: str) -> str:
    small_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=small_llm_name,
        tokenizer=small_llm_name,
        enforce_eager=False,
        max_num_seqs=1,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        max_model_len=32768,
    )

    system_prompt = """
    You are a specialized research methodology expert who transforms flawed research questions into high-quality ones. Your ONLY task is to improve a given question based on a specific suggestion. Your output should ONLY be the improved question with no additional explanations.
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
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 512

    modify_sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
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
    outputs = small_llm.chat(conversations, modify_sampling_params, use_tqdm=True)
    generated_text = outputs[0].outputs[0].text.strip().strip('"')
    return generated_text


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=2)
def question_to_query_threaded(question: str) -> dict:
    medium_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=medium_llm_name,
        tokenizer=medium_llm_name,
        enforce_eager=False,
        max_num_seqs=1,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        max_model_len=32768,
    )

    system_prompt = """
    You are a specialized academic search expert who transforms research questions into effective search queries for the Semantic Scholar Paper bulk search API.
    """
    user_prompt = f"""
    Original research question: "{question}"
    
    Task: Convert this research question into effective search parameters for the Semantic Scholar Paper bulk search API.

    Available API parameters:
    - query: The text query with Boolean operators (required)
    - sort: Sort results by publicationDate or citationCount (e.g., publicationDate:desc, citationCount:asc, etc.) (required)
    - publicationTypes: Review, JournalArticle, CaseReport, ClinicalTrial, Conference, Dataset, Editorial, LettersAndComments, MetaAnalysis, News, Study, Book, BookSection
    - year: Specific publication year or range (e.g., "2019", "2015-2020", etc.)
    - fieldsOfStudy: Computer Science, Medicine, Chemistry, Biology, Materials Science, Physics, Geology, Psychology, Art, History, Geography, Sociology, Business, Political Science, Economics, Philosophy, Mathematics, Engineering, Environmental Science, Agricultural and Food Sciences, Education, Law, Linguistics
    
    Tips for query construction:
    - Search for a phrase: remember that, when we search in library databases, we are searching for words, not topics (e.g., if looking for 'James Earl Ray', use 'James Earl Ray' instead of 'james, and earl, and ray,').
    - Search for variant forms of words via truncation: enter a common stem, followed by the search engine’s truncation symbol, an asterisk [*]. (e.g., 'immigra*' -> immigrant, immigration, immigrants, immigrate, etc.)
    - Combine multiple search terms into a single query via Boolean Searching: | for OR, + for AND (e.g., 'immigrant | immigration' -> immigrant OR immigration)

    Format response as JSON:
    {{
        "query": <query>,
        "sort": <type:order>,
        "publicationTypes": <comma-separated list of relevant publication types (if applicable)>,
        "year": <year (or range)> (if applicable),
        "fieldsOfStudy": <comma-separated list of relevant fields of study (if applicable)>,
    }}

    Return ONLY the valid JSON object without additional text.
    """

    temperature = 0.0
    top_p = 0.8
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 512
    guided_decoding = GuidedDecodingParams(json=SearchParamsBase.model_json_schema())

    modify_sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
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
    outputs = medium_llm.chat(conversations, modify_sampling_params, use_tqdm=True)
    generated_text = outputs[0].outputs[0].text.strip()
    response_dict = json.loads(generated_text)
    search_params = SearchParamsBase.model_validate(response_dict)
    return search_params.model_dump()


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=1000)
def search_papers_threaded(
    query: str,
    sort: str | None = None,
    publicationTypes: str | None = None,
    year: str | None = None,
    venue: str | None = None,
    fieldsOfStudy: str | None = None,
) -> list[dict]:
    max_results = 10000

    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
    url += f"?query={query}"
    url += "&fields=matchScore,paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles,authors,citations,references,embedding,tldr"
    if sort:
        url += f"&sort={sort}"
    if publicationTypes:
        url += f"&publicationTypes={publicationTypes}"
    url += "&isOpenAccess"
    url += "&minCitationCount=0"
    if year:
        url += f"&year={year}"
    if venue:
        url += f"&venue={venue}"
    if fieldsOfStudy:
        url += f"&fieldsOfStudy={fieldsOfStudy}"

    r = requests.get(url).json()
    papers = []
    while True:
        if "data" in r:
            for paper_data in r["data"]:
                is_open_access = paper_data["isOpenAccess"]
                if is_open_access:
                    paper = PaperBase(
                        match_score=paper_data["matchScore"],
                        paper_id=paper_data["paperId"],
                        corpusId=paper_data["corpusId"],
                        external_ids=paper_data["externalIds"],
                        url=paper_data["url"],
                        title=paper_data["title"],
                        abstract=paper_data["abstract"],
                        venue=paper_data["venue"],
                        publicationVenue=paper_data["publicationVenue"],
                        year=paper_data["year"],
                        reference_count=paper_data["referenceCount"],
                        citation_count=paper_data["citationCount"],
                        influential_citation_count=paper_data[
                            "influentialCitationCount"
                        ],
                        is_open_access=is_open_access,
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
        if "token" not in r or len(papers) >= max_results:
            break
        r = requests.get(f"{url}&token={r['token']}").json()
    return papers


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=4)
def rerank_papers_threaded(question: str, papers: list[dict]) -> list[dict]:
    max_results = 100

    ranker = Reranker(
        reranker_name,
        model_type="colbert",
        verbose=0,
        dtype=torch.bfloat16,
        device="cuda",
        batch_size=16,
        model_kwargs={"cache_dir": PRETRAINED_VOL_PATH},
    )
    docs = [
        f"""
        Title: {paper["title"]}
        Abstract: {paper["abstract"]}
        Venue: {paper["venue"]}
        Publication Venue: 
            Name: {paper["publicationVenue"]["name"]}
            Type: {paper["publicationVenue"]["type"]}
        Year: {paper["year"]}
        Reference Count: {paper["referenceCount"]}
        Citation Count: {paper["citationCount"]}
        Influential Citation Count: {paper["influentialCitationCount"]}
        Fields of Study: {paper["fieldsOfStudy"]}
        Publication Types: {paper["publicationTypes"]}
        Publication Date: {paper["publicationDate"]}
        Journal: {paper["journal"]}
        Authors: {"\n\n".join(
            [
                f"""
        Name: {author["name"]}
        Affiliations: {', '.join(author["affiliations"])}
        Paper Count: {author["paperCount"]}
        Citation Count: {author["citationCount"]}
        H-Index: {author["hIndex"]}
        """
                for author in paper["authors"]
            ]
        )}
        Text: {"\n\n".join([element.text for element in partition_pdf(
            paper["openAccessPdf"]["url"],
            url=None,
        )])}
        """
        for paper in papers
    ]
    results = ranker.rank(query=question, docs=docs)
    top_k_idxs = [doc.doc_id for doc in results.top_k(max_results)]
    return [papers[i] for i in top_k_idxs]


# -----------------------------------------------------------------------------

default_question = "What is the relationship between climate change and global health?"


@app.local_entrypoint()
def main():
    suggestions = check_question_threaded.remote(default_question)
    print(suggestions)

    modified_question = modify_question_threaded.remote(
        default_question,
        suggestions[0],
    )
    print(modified_question)

    search_params = question_to_query_threaded.remote(modified_question)
    print(search_params)

    papers = search_papers_threaded.remote(**search_params)
    print(len(papers))
    print(papers[0] if papers else None)

    reranked_papers = rerank_papers_threaded.remote(
        modified_question,
        papers,
    )
    print(len(reranked_papers))
    print(reranked_papers[0] if reranked_papers else None)


if __name__ == "__main__":
    suggestions = check_question_threaded.local(default_question)
    print(suggestions)

    modified_question = modify_question_threaded.local(
        default_question,
        suggestions[0],
    )
    print(modified_question)

    search_params = question_to_query_threaded.local(modified_question)
    print(search_params)

    papers = search_papers_threaded.local(**search_params)
    print(len(papers))
    print(papers[0] if papers else None)

    reranked_papers = rerank_papers_threaded.local(
        modified_question,
        papers,
    )
    print(len(reranked_papers))
    print(reranked_papers[0] if reranked_papers else None)
