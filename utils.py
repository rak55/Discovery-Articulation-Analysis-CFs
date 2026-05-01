from __future__ import annotations
 
import json
import logging
import os
import re
import time
from pathlib import Path
from string import Formatter
from typing import Any
 
import openai
import pandas as pd
import yaml
 
import config
from schemas import IssueFrame
 
logger = logging.getLogger(__name__)
 
 
 
class PromptLoader:
     
 
    def __init__(self, prompts_dir: str | None = None) -> None:
        self._dir   = Path(prompts_dir or config.PROMPTS_DIR)
        self._cache: dict[str, dict] = {}
 
 
    def load(self, agent_name: str) -> dict:
        """
        Return the raw parsed YAML dict for agent_name.
        Results are cached in-process so each file is read only once.
        """
        if agent_name in self._cache:
            return self._cache[agent_name]
 
        filename = config.AGENT_PROMPT_MAP.get(agent_name, f"{agent_name}.yaml")
        path     = self._dir / filename
 
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
            )
 
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
 
        if not isinstance(data, dict):
            raise ValueError(f"Prompt file {path} must be a YAML mapping, got {type(data)}.")
        if "system" not in data:
            raise ValueError(f"Prompt file {path} is missing the required 'system' key.")
 
        self._cache[agent_name] = data
        logger.debug("Loaded prompt for agent '%s' from %s", agent_name, path)
        return data
 
    def get_system(self, agent_name: str) -> str:
        return self.load(agent_name)["system"].strip()
 
    def get_user(self, agent_name: str, **kwargs) -> str:
        data = self.load(agent_name)
        if "user_template" not in data:
            raise ValueError(
                f"Agent '{agent_name}' has no 'user_template' in its YAML file. "
                f"Either add one or build the user string manually."
            )
        template = data["user_template"]
        required = {
            fname
            for _, fname, _, _ in Formatter().parse(template)
            if fname is not None
        }
        missing = required - set(kwargs.keys())
        if missing:
            raise KeyError(
                f"Missing placeholder(s) for agent '{agent_name}': {missing}.\n"
            )
        return template.format_map(kwargs).strip()
 
    def get_output_format(self, agent_name: str) -> str:
        return self.load(agent_name).get("output_format", "")
 
    def reload(self, agent_name: str | None = None) -> None:
        """Invalidate the cache (for one agent or all) to pick up edited files."""
        if agent_name:
            self._cache.pop(agent_name, None)
        else:
            self._cache.clear()
  
    def available_agents(self) -> list[str]:
        return list(config.AGENT_PROMPT_MAP.keys())
 
    def missing_files(self) -> list[str]:
        """Return a list of agent names whose YAML files don't exist yet."""
        missing = []
        for name, fname in config.AGENT_PROMPT_MAP.items():
            if not (self._dir / fname).exists():
                missing.append(name)
        return missing
 
 
prompt_loader = PromptLoader()
 
 
_client: openai.OpenAI | None = None
 
 
def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()   
    return _client
 
 
def call_llm(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = config.MAX_TOKENS,
    temperature:   float = config.TEMPERATURE,
    max_retries:   int   = 4,
) -> str:
    """Call GPT-5-mini and return the assistant's text. Retries on rate-limit."""
    client = get_client()
    delay  = 2.0
 
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model       = config.MODEL_NAME,
                max_tokens  = max_tokens,
                temperature = temperature,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            logger.warning("Rate limit (attempt %d/%d). Sleeping %.1fs.", attempt, max_retries, delay)
            time.sleep(delay)
            delay *= 2
        except openai.APIStatusError as exc:
            logger.error("API error %s: %s", exc.status_code, exc.message)
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
 
    raise RuntimeError("OpenAI API call failed after all retries.")
 
 
 
def parse_json_response(text: str) -> Any:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Cannot parse LLM output as JSON.\nError: {exc}\nText:\n{text[:500]}"
        ) from exc
 
 

def load_known_isfs(topic: str) -> list[IssueFrame]:
    """
    Load known Issue Frames for *topic* from ``problems/<topic>.json``.
    """
    filename = config.TOPIC_PROBLEMS_MAP.get(topic, f"{topic}.json")
    path     = Path(config.PROBLEMS_DIR) / filename
 
    if not path.exists():
        raise FileNotFoundError(
            f"Known-IsF file not found: {path}\n"
            f"Create a JSON file there with a list of "
            f'{{ "name": "...", "definition": "..." }} objects.'
        )
 
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
 
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON *array*, got {type(data).__name__}.")
 
    frames: list[IssueFrame] = []
    for i, item in enumerate(data):
        if "name" not in item or "definition" not in item:
            raise ValueError(
                f"Entry #{i} in {path} is missing 'name' or 'definition'.\n"
                f"Got: {item}"
            )
        frames.append(IssueFrame(
            name       = item["name"].strip(),
            definition = item["definition"].strip(),
            is_new     = False,
        ))
 
    logger.info("Loaded %d known IsFs for topic '%s' from %s", len(frames), topic, path)
    return frames
 
 
def load_postings(topic: str, max_rows: int | None = None) -> list[str]:
    """
    Load tweet texts from ``data/<topic>.csv``.
    """
    filename = config.TOPIC_CSV_MAP.get(topic, f"{topic}.csv")
    path     = Path(config.DATA_DIR) / filename
 
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Place a CSV with a 'text' column there."
        )
 
    df = pd.read_csv(path, nrows=max_rows)
    if "text" not in df.columns:
        raise ValueError(
            f"'{path}' must have a column named 'text'. Found: {list(df.columns)}"
        )
 
    postings = df["text"].dropna().astype(str).tolist()
    logger.info("Loaded %d postings for topic '%s'.", len(postings), topic)
    return postings
 

 
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sbert: SentenceTransformer | None = SentenceTransformer(config.SBERT_MODEL)
    _USE_SBERT = True
    logger.info("sentence-transformers loaded (%s).", config.SBERT_MODEL)
except ImportError:
    _sbert     = None
    _USE_SBERT = False
    logger.warning(
        "sentence-transformers not installed – falling back to TF-IDF similarity. "

    )
 
 
def compute_similarity(text_a: str, text_b: str) -> float:
    if _USE_SBERT and _sbert:
        emb_a = _sbert.encode(text_a, convert_to_tensor=True)
        emb_b = _sbert.encode(text_b, convert_to_tensor=True)
        return float(st_util.pytorch_cos_sim(emb_a, emb_b).item())
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer().fit_transform([text_a, text_b])
        return float(cosine_similarity(vec[0], vec[1])[0][0])
    except Exception:
        a, b = set(text_a.lower().split()), set(text_b.lower().split())
        return len(a & b) / len(a | b) if (a and b) else 0.0
 
 
def is_duplicate_definition(
    candidate_def: str,
    existing_defs: list[str],
    threshold:     float = config.SIMILARITY_THRESHOLD,
) -> tuple[bool, float]:
    if not existing_defs:
        return False, 0.0
    max_sim = max(compute_similarity(candidate_def, d) for d in existing_defs)
    return max_sim >= threshold, max_sim

 
def ensure_output_dir() -> Path:
    out = Path(config.OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    return out
 
 
def save_json(obj: Any, filename: str) -> Path:
    path = ensure_output_dir() / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    logger.info("Saved → %s", path)
    return path
 
 
def fmt_isf_list(frames: list[IssueFrame]) -> str:
    return "\n".join(f"{i}. **{f.name}**: {f.definition}" for i, f in enumerate(frames, 1))
 
 
def fmt_demonstration(posting: str, isf_name: str, rationale: str, applies: bool) -> str:
    label = "ADDRESSES" if applies else "DOES NOT ADDRESS"
    return f"Post: {posting}\nIsF:  {isf_name}\nVerdict: {label}\nRationale: {rationale}"
