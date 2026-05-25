from __future__ import annotations
 
import logging
from typing import Optional
 
import config
from schemas import (
    ICI, ICIMemory, InvestigatorVerdict,
    IssueFrame, IssueFrameMemory,
    PostingMemory, PostingRecord,
)
from utils import call_llm, fmt_isf_list, parse_json_response, prompt_loader
from id_isf import InvestigatorAgent, IntrospectiveQALoop
 
logger = logging.getLogger(__name__)
 
class ArticulationAgent:
    """
    Prompt file: prompts/articulation.yaml
    Required YAML keys:
      system        – CoT articulation instruction
      user_template – placeholders: {posting}, {addressed_isfs}
    """
 
    def articulate(self, posting: str, addressed_isfs: list[IssueFrame]) -> list[ICI]:
        if not addressed_isfs:
            return []
 
        system      = prompt_loader.get_system("articulation")
        user_prompt = prompt_loader.get_user(
            "articulation",
            posting       = posting,
            addressed_isfs= fmt_isf_list(addressed_isfs),
        )
        raw  = call_llm(system, user_prompt, max_tokens=1024)
        data = parse_json_response(raw)
 
        results: list[ICI] = []
        for item in data.get("icis", []):
            text      = item.get("text",            "").strip()
            isf_names = item.get("interpreted_isfs", [])
            rationale = item.get("rationale",       "").strip()
            if text and isf_names:
                results.append(ICI(
                    text             = text,
                    interpreted_isfs = isf_names,
                    rationale        = rationale,
                    evoked_by        = [posting],
                ))
        return results
 
 
class ParaphrasePoliceAgent:
    """
    Prompt file: prompts/paraphrase_police.yaml
    Required YAML keys:
      system        – paraphrase-detection instruction
      user_template – placeholder: {ici_list}
                      (numbered list of ICI texts for a single IsF group)
    """
 
    def deduplicate(self, icis: list[ICI]) -> list[ICI]:
        if len(icis) <= 1:
            return icis
 
        from collections import defaultdict
        isf_groups: dict[frozenset, list[int]] = defaultdict(list)
        for idx, ici in enumerate(icis):
            isf_groups[frozenset(ici.interpreted_isfs)].append(idx)
 
        kept: set[int] = set()
 
        for key, indices in isf_groups.items():
            if len(indices) == 1:
                kept.add(indices[0])
                continue
 
            group_texts = "\n".join(
                f"{j}. {icis[i].text}" for j, i in enumerate(indices)
            )
            system      = prompt_loader.get_system("paraphrase_police")
            user_prompt = prompt_loader.get_user(
                "paraphrase_police",
                ici_list = group_texts,
            )
            raw  = call_llm(system, user_prompt, max_tokens=512)
            data = parse_json_response(raw)
 
            for grp in data.get("groups", [[j] for j in range(len(indices))]):
                if grp:
                    kept.add(indices[grp[0]])
 
        result  = [icis[i] for i in sorted(kept)]
        removed = len(icis) - len(result)
        if removed:
            logger.debug("  Paraphrase-Police removed %d duplicate ICI(s).", removed)
        return result
 

 
class ArticulationRewardModel:
    """
    Loads the fine-tuned DeBERTa-v3-large binary classifier
    """
 
    def __init__(self, model_path: str | None = None) -> None:
        path = model_path or config.DEBERTA_MODEL_PATH
        self._load(path)
 
    def _load(self, path: str) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as exc:
            raise ImportError(
                ""
            ) from exc
 
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
        logger.info("Loading DeBERTa reward model from '%s' …", path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model     = AutoModelForSequenceClassification.from_pretrained(path)
        self._model.eval()
 
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        logger.info(
            "DeBERTa reward model loaded  (num_labels=%d, device=%s).",
            self._model.config.num_labels,
            self._device,
        )
 
        self._num_labels = self._model.config.num_labels
 
    def score(self, ici: ICI) -> float:
        import torch
 
        encoding = self._tokenizer(
            ici.text,
            truncation     = True,
            max_length     = config.DEBERTA_MAX_LENGTH,
            return_tensors = "pt",
        ).to(self._device)
 
        with torch.no_grad():
            logits = self._model(**encoding).logits   # shape: (1, num_labels)
 
        if self._num_labels == 1:
            score = torch.sigmoid(logits[0, 0]).item()
        else:
            # softmax over [invalid, valid]; take P(valid)
            score = torch.softmax(logits[0], dim=0)[1].item()
 
        return float(score)
 
 
 
class ExplainabilityAgent:
 
    def __init__(self, reward_model: ArticulationRewardModel) -> None:
        self.reward_model = reward_model
 
    def validate(
        self,
        ici:      ICI,
        all_isfs: list[IssueFrame],
        tau:      float = config.ICI_ACCEPTANCE_THRESHOLD,
    ) -> bool:
        # Step 1 – reward model quality 
        score          = self.reward_model.score(ici)
        ici.validity_score = score
        if score < tau:
            logger.debug("  ICI rejected by reward model (%.2f < %.2f).", score, tau)
            return False
 
        # Step 2 – reconstruction test
        reconstructed = self._reconstruct(ici, all_isfs)
        if not reconstructed:
            return False
        return self._are_paraphrases(ici.text, reconstructed)
 
    def _reconstruct(self, ici: ICI, all_isfs: list[IssueFrame]) -> Optional[str]:
        relevant = [f for f in all_isfs if f.name in ici.interpreted_isfs]
        isf_defs = "\n".join(f"- {f.name}: {f.definition}" for f in relevant)
 
        system      = prompt_loader.get_system("explainability_reconstruct")
        user_prompt = prompt_loader.get_user(
            "explainability_reconstruct",
            ici_rationale   = ici.rationale,
            isf_definitions = isf_defs,
        )
        try:
            raw  = call_llm(system, user_prompt, max_tokens=256)
            data = parse_json_response(raw)
            return data.get("reconstructed_ici", "")
        except Exception:
            return None
 
    def _are_paraphrases(self, ici_a: str, ici_b: str) -> bool:
        system      = prompt_loader.get_system("explainability_paraphrase")
        user_prompt = prompt_loader.get_user(
            "explainability_paraphrase",
            ici_a = ici_a,
            ici_b = ici_b,
        )
        try:
            raw  = call_llm(system, user_prompt, max_tokens=128)
            data = parse_json_response(raw)
            return bool(data.get("are_paraphrases", False))
        except Exception:
            return False
 
 
 
class ArtICI:
 
    def __init__(self, deberta_model_path: str | None = None) -> None:
        self.investigator = InvestigatorAgent()
        self.qa_loop      = IntrospectiveQALoop()
        self.articulator  = ArticulationAgent()
        self.para_police  = ParaphrasePoliceAgent()
        self.reward_model = ArticulationRewardModel(deberta_model_path)
        self.explainer    = ExplainabilityAgent(self.reward_model)
 
    def _run_c1(self, postings: list[str], ifm: IssueFrameMemory) -> PostingMemory:
        logger.info(
            "Art-ICI C1: %d postings × %d total IsFs …",
            len(postings), len(ifm.all_frames),
        )
        pm = PostingMemory()
        for idx, posting in enumerate(postings, 1):
            logger.debug("  Posting %d/%d", idx, len(postings))
            addressed: list[str] = []
            for isf in ifm.all_frames:
                result = self.investigator.classify_with_qa(posting, isf, self.qa_loop)
                if result.verdict == InvestigatorVerdict.APPLICABLE:
                    addressed.append(isf.name)
            pm.records.append(PostingRecord(text=posting, addressed_isfs=addressed))
        return pm
 
    def _run_cc3(self, pm: PostingMemory, ifm: IssueFrameMemory) -> ICIMemory:
        logger.info("Art-ICI CC3: articulating ICIs …")
        all_raw: list[ICI] = []
 
        for record in pm.records:
            if not record.addressed_isfs:
                continue
            relevant = [f for f in ifm.all_frames if f.name in record.addressed_isfs]
            all_raw.extend(self.articulator.articulate(record.text, relevant))
 
        logger.info("  Deduplicating %d raw ICIs …", len(all_raw))
        deduped = self.para_police.deduplicate(all_raw)
        logger.info("  → %d after deduplication.", len(deduped))
 
        ici_memory  = ICIMemory()
        valid_count = 0
        for ici in deduped:
            ici.is_valid = self.explainer.validate(ici, ifm.all_frames)
            ici_memory.add(ici)
            if ici.is_valid:
                valid_count += 1
 
        logger.info("  → %d / %d ICIs passed validation.", valid_count, len(deduped))
        return ici_memory
 
    def run(self, postings: list[str], ifm: IssueFrameMemory) -> ICIMemory:
        pm         = self._run_c1(postings, ifm)
        ici_memory = self._run_cc3(pm, ifm)
        logger.info("Art-ICI done. Total=%d  Valid=%d",
                    len(ici_memory.icis), len(ici_memory.valid_icis()))
        return ici_memory
