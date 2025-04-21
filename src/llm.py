import json
import os
import tempfile
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
small_llm_enforce_eager = False
small_llm_max_num_seqs = 1
small_llm_trust_remote_code = True
small_llm_max_model_len = 32768

medium_llm_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
medium_llm_enforce_eager = False
medium_llm_max_num_seqs = 1
medium_llm_trust_remote_code = True
medium_llm_max_model_len = 32768

reranker_name = "answerdotai/answerai-colbert-small-v1"


def download_models():
    for llm_name in [small_llm_name, medium_llm_name, reranker_name]:
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
    from tqdm import tqdm
    from tqdm.contrib.concurrent import thread_map
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
@modal.concurrent(max_inputs=20)
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
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
)
@modal.concurrent(max_inputs=20)
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
@modal.concurrent(max_inputs=100)
def question_to_search_params_and_papers_threaded(
    question: str,
    min_results: int = 10,
    max_results: int = 100,
) -> tuple[str, list[dict]]:  # search params, papers
    medium_llm = LLM(
        download_dir=PRETRAINED_VOL_PATH,
        model=medium_llm_name,
        tokenizer=medium_llm_name,
        enforce_eager=medium_llm_enforce_eager,
        max_num_seqs=medium_llm_max_num_seqs,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=medium_llm_trust_remote_code,
        max_model_len=medium_llm_max_model_len,
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

    temperature = 0.0
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

    # retries in case no results
    num_retries = 3
    for i in range(num_retries):
        outputs = medium_llm.chat(conversations, modify_sampling_params, use_tqdm=True)
        generated_text = outputs[0].outputs[0].text.strip()

        search_params = SearchParamsBase.model_validate({"query": generated_text})
        search_params_dict = search_params.model_dump()
        papers = (
            search_papers_threaded.local(**search_params_dict)
            if modal.is_local()
            else search_papers_threaded.remote(**search_params_dict)
        )
        if len(papers) >= min_results:
            break
        else:
            if i == num_retries - 1:
                print(
                    f"Warning: Skipping search for question {question} due to too few results"
                )
                search_params_dict = {}
                papers = []
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
                                "text": (
                                    f"Error: Need {min_results} papers, only {len(papers)} found using these parameters: {generated_text}. "
                                    "Please reformulate the search parameters for broader coverage and return only the JSON with new parameters."
                                ),
                            },
                        ],
                    }
                )

    return search_params_dict, papers[:max_results]


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
@modal.concurrent(max_inputs=1000)
def search_papers_threaded(
    query: str,
    max_results: int = 50,
) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    url += f"?query={query}"
    url += "&fields=paperId,corpusId,externalIds,url,title,abstract,venue,publicationVenue,year,referenceCount,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,journal,citationStyles,authors"
    url += "&openAccessPdf"

    r = requests.get(url).json()
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
        if "token" not in r or len(papers) >= max_results:
            break
        r = requests.get(f"{url}&token={r['token']}").json()
    return papers[:max_results]


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
@modal.concurrent(max_inputs=100)
def paper_to_text(paper: dict) -> str | None:
    try:
        open_access_pdf = paper.get("open_access_pdf", {})
        if not open_access_pdf:
            return None
        pdf_url = open_access_pdf.get("url", "")
        if not pdf_url:
            return None

        # check pdf
        resp = requests.get(pdf_url)
        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            print(
                f"Warning: Skipping paper {paper.get('paper_id', 'unknown')} because Content-Type is not PDF: {content_type}"
            )
            return None
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
            tmp_file.write(resp.content)
            tmp_file.flush()

            with open(tmp_file.name, "rb") as f:
                magic = f.read(4)
            if magic != b"%PDF":
                print(
                    f"Warning: Skipping paper {paper.get('paper_id', 'unknown')} because file magic number is not %PDF"
                )
                return None

            text = "\n\n".join(
                [
                    element.text
                    for element in partition_pdf(
                        tmp_file.name,
                        url=None,
                    )
                ]
            )

        return text
    except Exception as e:
        print(
            f"Warning: Skipping paper {paper.get('paper_id', 'unknown')} due to error: {e}"
        )
        return None


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
@modal.concurrent(max_inputs=100)
def rerank_papers_threaded(
    question: str, papers: list[dict], max_results: int = 50
) -> list[dict]:
    paper_idxs = range(len(papers))
    ranker = Reranker(
        reranker_name,
        model_type="colbert",
        verbose=0,
        dtype=torch.bfloat16,
        device="cuda",
        batch_size=16,
        model_kwargs={"cache_dir": PRETRAINED_VOL_PATH},
    )
    if modal.is_local():
        texts = list(
            tqdm(thread_map(paper_to_text.local, papers), desc="Processing papers")
        )
    else:
        texts = list(paper_to_text.map(papers))

    filtered = [(text, idx) for text, idx in zip(texts, paper_idxs) if text is not None]
    if not filtered:
        return []
    texts, paper_idxs = zip(*filtered)
    docs = [
        f"""
            Title: {paper.get("title", "")}
            Abstract: {paper.get("abstract", "")}
            Venue: {paper.get("venue", "")}
            Publication Venue: {paper.get("publication_venue", {})}
            Year: {paper.get("year", "")}
            Reference Count: {paper.get("reference_count", "")}
            Citation Count: {paper.get("citation_count", "")}
            Influential Citation Count: {paper.get("influential_citation_count", "")}
            Text: {text}
            Fields of Study: {paper.get("fields_of_study", "")}
            Publication Types: {paper.get("publication_types", "")}
            Publication Date: {paper.get("publication_date", "")}
            Journal: {paper.get("journal", "")}
            Authors: {"\n\n".join(
                [
                    f"""
                    Name: {author.get("name", "")}
                    Affiliations: {', '.join(author.get("affiliations", []))}
                    Paper Count: {author.get("paper_count", "")}
                    Citation Count: {author.get("citation_count", "")}
                    H-Index: {author.get("h_index", "")}
                    """
                    for author in paper.get("authors", [])
                ]
            )}
        """
        for paper, text in zip(papers, texts)
    ]

    results = ranker.rank(query=question, docs=docs)
    top_k_idxs = [doc.doc_id for doc in results.top_k(max_results)]
    screened_papers = []
    for i in top_k_idxs:
        paper = papers[paper_idxs[i]]
        paper["text"] = texts[i]
        screened_papers.append(paper)
    return screened_papers


# -----------------------------------------------------------------------------


@app.function(
    image=GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=60 * MINUTES,
)
def main():
    default_question = (
        "What is the relationship between climate change and global health?"
    )

    suggestions = (
        check_question_threaded.local(default_question)
        if modal.is_local()
        else check_question_threaded.remote(default_question)
    )
    print(suggestions)

    modified_question = (
        modify_question_threaded.local(default_question, suggestions[0])
        if modal.is_local()
        else modify_question_threaded.remote(default_question, suggestions[0])
    )
    print(modified_question)

    modified_question = default_question

    search_params, papers = (
        question_to_search_params_and_papers_threaded.local(modified_question)
        if modal.is_local()
        else question_to_search_params_and_papers_threaded.remote(modified_question)
    )
    print(search_params)
    print(len(papers))
    print(papers[0] if papers else None)

    reranked_papers = (
        rerank_papers_threaded.local(modified_question, papers)
        if modal.is_local()
        else rerank_papers_threaded.remote(modified_question, papers)
    )
    print(len(reranked_papers))
    print(reranked_papers[0] if reranked_papers else None)


@app.local_entrypoint()
def main_modal():
    main.remote()


if __name__ == "__main__":
    main.local()
