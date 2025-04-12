import json
from pathlib import PurePosixPath

import modal
from pydantic import BaseModel

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

GPU_IMAGE = (
    modal.Image.from_registry(TAG, add_python=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(
        "accelerate>=1.6.0",
        "flashinfer-python==0.2.2.post1",
        "hf-transfer>=0.1.9",
        "huggingface-hub>=0.30.2",
        "ninja>=1.11.1.4",  # required to build flash-attn
        "packaging>=24.2",  # required to build flash-attn
        "torch>=2.6.0",
        "tqdm>=4.67.1",
        "transformers>=4.51.1",
        "vllm>=0.8.3",
        "wheel>=0.45.1",  # required to build flash-attn
    )
    .run_commands("pip install flash-attn==2.7.4.post1 --no-build-isolation")
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HUGGINGFACE_HUB_CACHE": f"/{PRETRAINED_VOLUME}",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_python_source("src", copy=True)
)
app = modal.App(f"{APP_NAME}-llm")

# -----------------------------------------------------------------------------

with GPU_IMAGE.imports():
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class CheckQuestion(BaseModel):
    suggestions: list[str] | None


global small_llm, check_sampling_params, modify_sampling_params


@app.function(
    image=GPU_IMAGE,
    cpu=CPU,
    memory=MEM,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=SECRETS,
    timeout=2 * MINUTES,
    scaledown_window=15 * MINUTES,
    allow_concurrent_inputs=1000,
)
def check_question_threaded(question: str) -> list[str] | None:
    max_num_suggestions = 3

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

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    processor = "Qwen/Qwen2.5-1.5B-Instruct"
    # quantization = None  # "awq_marlin"
    # kv_cache_dtype = None  # "fp8_e5m2"
    enforce_eager = False
    max_num_seqs = 1
    max_model_len = 2048

    temperature = 0.7
    top_p = 0.8
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 512
    guided_decoding = GuidedDecodingParams(json=CheckQuestion.model_json_schema())

    global small_llm, check_sampling_params
    if "small_llm" not in globals():
        small_llm = LLM(
            download_dir=f"/{PRETRAINED_VOLUME}",
            model=model_name,
            tokenizer=processor,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
    if "check_sampling_params" not in globals():
        check_sampling_params = SamplingParams(
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
    outputs = small_llm.chat(conversations, check_sampling_params, use_tqdm=True)
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
    allow_concurrent_inputs=1000,
)
def modify_question_threaded(question: str, suggestion: str) -> str:
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

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    processor = "Qwen/Qwen2.5-1.5B-Instruct"
    # quantization = None  # "awq_marlin"
    # kv_cache_dtype = None  # "fp8_e5m2"
    enforce_eager = False
    max_num_seqs = 1
    max_model_len = 2048

    temperature = 0.7
    top_p = 0.8
    repetition_penalty = 1.05
    stop_token_ids = []
    max_tokens = 512

    global small_llm, modify_sampling_params
    if "small_llm" not in globals():
        small_llm = LLM(
            download_dir=f"/{PRETRAINED_VOLUME}",
            model=model_name,
            tokenizer=processor,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
    if "modify_sampling_params" not in globals():
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


# -----------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    suggestions = check_question_threaded.remote(
        "What is the relationship between climate change and global health?"
    )
    print(suggestions)

    modified_question = modify_question_threaded.remote(
        "What is the relationship between climate change and global health?",
        suggestions[0],
    )
    print(modified_question)


if __name__ == "__main__":
    suggestions = check_question_threaded.local(
        "What is the relationship between climate change and global health?"
    )
    print(suggestions)

    modified_question = modify_question_threaded.local(
        "What is the relationship between climate change and global health?",
        suggestions[0],
    )
    print(modified_question)
