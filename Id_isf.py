from __future__ import annotations
 
import logging
import random
from typing import Optional
 
import config
from schemas import (
    CandidateIsF, DebateResult, DebateRound, DebateVerdict,
    InvestigatorResult, InvestigatorVerdict,
    IssueFrame, IssueFrameMemory, QAPair,
)
from utils import (
    call_llm, fmt_demonstration, fmt_isf_list,
    is_duplicate_definition, parse_json_response,
    prompt_loader,
)
 
logger = logging.getLogger(__name__)
 

#Investigator Agent

 
class InvestigatorAgent:
    """
    Prompt file: prompts/investigator.yaml
    """
 
    def classify(self, posting: str, isf: IssueFrame) -> InvestigatorResult:
        system      = prompt_loader.get_system("investigator")
        user_prompt = prompt_loader.get_user(
            "investigator",
            posting        = posting,
            isf_name       = isf.name,
            isf_definition = isf.definition,
        )
        raw     = call_llm(system, user_prompt, max_tokens=512)
        data    = parse_json_response(raw)
        verdict_str = data.get("verdict", "uncertain").strip().lower().replace(" ", "_")
        try:
            verdict = InvestigatorVerdict(verdict_str)
        except ValueError:
            verdict = InvestigatorVerdict.UNCERTAIN
 
        return InvestigatorResult(
            isf_name  = isf.name,
            verdict   = verdict,
            rationale = data.get("rationale", ""),
        )
 
    def classify_with_qa(
        self,
        posting: str,
        isf:     IssueFrame,
        qa_loop: "IntrospectiveQALoop",
    ) -> InvestigatorResult:
        result = self.classify(posting, isf)
        if result.verdict == InvestigatorVerdict.UNCERTAIN:
            logger.debug("  Investigator uncertain about '%s' → Q/A loop", isf.name)
            qa_pairs = qa_loop.run(posting, isf, result.rationale)
            result   = self._resolve_with_qa(posting, isf, result, qa_pairs)
        return result
 
    def _resolve_with_qa(
        self,
        posting:  str,
        isf:      IssueFrame,
        prior:    InvestigatorResult,
        qa_pairs: list[QAPair],
    ) -> InvestigatorResult:
        """
        Prompt file: prompts/investigator_resolver.yaml
        """
        qa_text = "\n".join(
            f"Q: {p.question}\nA: {p.answer} – {p.rationale}" for p in qa_pairs
        )
        system      = prompt_loader.get_system("investigator_resolver")
        user_prompt = prompt_loader.get_user(
            "investigator_resolver",
            posting        = posting,
            isf_name       = isf.name,
            isf_definition = isf.definition,
            prior_rationale= prior.rationale,
            qa_evidence    = qa_text,
        )
        raw  = call_llm(system, user_prompt, max_tokens=512)
        data = parse_json_response(raw)
        verdict_str = data.get("verdict", "not_applicable").strip().lower().replace(" ", "_")
        try:
            verdict = InvestigatorVerdict(verdict_str)
        except ValueError:
            verdict = InvestigatorVerdict.NOT_APPLICABLE
 
        return InvestigatorResult(
            isf_name  = isf.name,
            verdict   = verdict,
            rationale = data.get("rationale", prior.rationale),
        )
 
 

# Introspective Q/A Loop
 
class IntrospectiveQALoop:
    """
    Prompt files:
      prompts/qa_question_generator.yaml
      prompts/qa_answer_generator.yaml

    """
 
    def run(
        self,
        posting:         str,
        isf:             IssueFrame,
        prior_rationale: str,
        max_loop:        int = config.MAX_QA_LOOP,
    ) -> list[QAPair]:
        pairs:       list[QAPair] = []
        prev_answer: Optional[str] = None
 
        for _ in range(1, max_loop + 1):
            question              = self._generate_question(posting, isf, prior_rationale, pairs, prev_answer)
            answer, rationale     = self._generate_answer(posting, isf, question)
            pairs.append(QAPair(question=question, answer=answer, rationale=rationale))
            prev_answer = answer
            if answer in ("yes", "no"):
                break
 
        return pairs
 
    def _generate_question(
        self,
        posting:         str,
        isf:             IssueFrame,
        prior_rationale: str,
        prior_pairs:     list[QAPair],
        prev_answer:     Optional[str],
    ) -> str:
        qa_history = "\n".join(
            f"Q: {p.question}\nA: {p.answer} – {p.rationale}" for p in prior_pairs
        )
        feedback = (
            "The previous answer was CANNOT ANSWER – generate a more specific question."
            if prev_answer == "cannot_answer" else ""
        )
        system      = prompt_loader.get_system("qa_question_generator")
        user_prompt = prompt_loader.get_user(
            "qa_question_generator",
            posting        = posting,
            isf_name       = isf.name,
            isf_definition = isf.definition,
            prior_rationale= prior_rationale,
            qa_history     = qa_history or "(none yet)",
            feedback       = feedback,
        )
        raw  = call_llm(system, user_prompt, max_tokens=256)
        data = parse_json_response(raw)
        return data.get("question", "Does the post mention the IsF topic?")
 
    def _generate_answer(self, posting: str, isf: IssueFrame, question: str) -> tuple[str, str]:
        system      = prompt_loader.get_system("qa_answer_generator")
        user_prompt = prompt_loader.get_user(
            "qa_answer_generator",
            posting        = posting,
            isf_name       = isf.name,
            isf_definition = isf.definition,
            question       = question,
        )
        raw  = call_llm(system, user_prompt, max_tokens=256)
        data = parse_json_response(raw)
        return data.get("answer", "cannot_answer").lower(), data.get("rationale", "")
 

#Proposer Agent
 
class ProposerAgent:
    """
    Prompt file: prompts/proposer.yaml
    """
 
    def propose(self, posting: str, ifm: IssueFrameMemory) -> list[CandidateIsF]:
        demos      = self._build_demonstrations(ifm)
        known_list = fmt_isf_list(ifm.known_frames + ifm.new_frames)
        system      = prompt_loader.get_system("proposer")
        user_prompt = prompt_loader.get_user(
            "proposer",
            known_isfs     = known_list,
            demonstrations = demos,
            posting        = posting,
        )
        raw  = call_llm(system, user_prompt, max_tokens=1024)
        data = parse_json_response(raw)
        return [
            CandidateIsF(
                name       = item.get("name",       "Unnamed"),
                definition = item.get("definition", ""),
                evidence   = item.get("evidence",   ""),
                rationale  = item.get("rationale",  ""),
                posting    = posting,
            )
            for item in data.get("candidates", [])
        ]
 
    def _build_demonstrations(self, ifm: IssueFrameMemory, max_per_isf: int = 2) -> str:
        parts: list[str] = []
        for isf in ifm.all_frames:
            for p, r in zip(isf.addressed_postings[:max_per_isf],
                            isf.addressed_rationales[:max_per_isf]):
                parts.append(fmt_demonstration(p, isf.name, r, applies=True))
            for p, r in zip(isf.not_addressed_postings[:max_per_isf],
                            isf.not_addressed_rationales[:max_per_isf]):
                parts.append(fmt_demonstration(p, isf.name, r, applies=False))
        random.shuffle(parts)
        return "\n---\n".join(parts[:20]) if parts else "(no demonstrations yet)"
 
 

# Multi-Agent Debate
 
class ChampionAgent:
    """
    Prompt file: prompts/champion.yaml
    """
 
    def argue(
        self,
        candidate:    CandidateIsF,
        known_isfs:   list[IssueFrame],
        prior_critic: Optional[str] = None,
    ) -> str:
        system      = prompt_loader.get_system("champion")
        user_prompt = prompt_loader.get_user(
            "champion",
            candidate_name       = candidate.name,
            candidate_definition = candidate.definition,
            candidate_evidence   = candidate.evidence,
            candidate_rationale  = candidate.rationale,
            posting              = candidate.posting,
            known_isfs           = fmt_isf_list(known_isfs),
            prior_critic         = prior_critic or "(first round – no prior critique)",
        )
        raw  = call_llm(system, user_prompt, max_tokens=512)
        data = parse_json_response(raw)
        return data.get("argument", "")
 
 
class CriticAgent:
    """
    Prompt file: prompts/critic.yaml
    """
 
    def argue(
        self,
        candidate:         CandidateIsF,
        known_isfs:        list[IssueFrame],
        champion_argument: str,
    ) -> str:
        system      = prompt_loader.get_system("critic")
        user_prompt = prompt_loader.get_user(
            "critic",
            candidate_name       = candidate.name,
            candidate_definition = candidate.definition,
            candidate_evidence   = candidate.evidence,
            candidate_rationale  = candidate.rationale,
            posting              = candidate.posting,
            known_isfs           = fmt_isf_list(known_isfs),
            champion_argument    = champion_argument,
        )
        raw  = call_llm(system, user_prompt, max_tokens=512)
        data = parse_json_response(raw)
        return data.get("argument", "")
 
 
class JudgeAgent:
    """
    Prompt file: prompts/judge.yaml
    """
 
    def judge(
        self,
        candidate:         CandidateIsF,
        known_isfs:        list[IssueFrame],
        champion_argument: str,
        critic_argument:   str,
    ) -> tuple[DebateVerdict, str, Optional[str], Optional[str]]:
        system      = prompt_loader.get_system("judge")
        user_prompt = prompt_loader.get_user(
            "judge",
            candidate_name       = candidate.name,
            candidate_definition = candidate.definition,
            posting              = candidate.posting,
            known_isfs           = fmt_isf_list(known_isfs),
            champion_argument    = champion_argument,
            critic_argument      = critic_argument,
        )
        raw  = call_llm(system, user_prompt, max_tokens=512)
        data = parse_json_response(raw)
 
        verdict_str = data.get("verdict", "reject").lower()
        try:
            verdict = DebateVerdict(verdict_str)
        except ValueError:
            verdict = DebateVerdict.REJECT
 
        return (
            verdict,
            data.get("comment",            ""),
            data.get("refined_name",       None),
            data.get("refined_definition", None),
        )
 
 
class MultiAgentDebate:
 
    def __init__(self) -> None:
        self.champion = ChampionAgent()
        self.critic   = CriticAgent()
        self.judge    = JudgeAgent()
 
    def run(
        self,
        candidate:  CandidateIsF,
        known_isfs: list[IssueFrame],
        max_rounds: int = config.MAX_DEBATE_ROUNDS,
    ) -> DebateResult:
        rounds:        list[DebateRound] = []
        critic_arg     = ""
        final_verdict  = DebateVerdict.REJECT
        final_comment  = ""
        refined_name   = None
        refined_def    = None
 
        for r in range(1, max_rounds + 1):
            champion_arg = self.champion.argue(candidate, known_isfs,
                                               prior_critic=critic_arg if r > 1 else None)
            critic_arg   = self.critic.argue(candidate, known_isfs, champion_arg)
            verdict, comment, r_name, r_def = self.judge.judge(
                candidate, known_isfs, champion_arg, critic_arg
            )
            rounds.append(DebateRound(r, champion_arg, critic_arg, comment))
 
            if verdict in (DebateVerdict.ACCEPT, DebateVerdict.REJECT, DebateVerdict.CORRECT):
                final_verdict = verdict
                final_comment = comment
                refined_name  = r_name
                refined_def   = r_def
                break
 
        logger.debug("  Debate '%s' → %s (%d round(s))", candidate.name, final_verdict.value, len(rounds))
        return DebateResult(candidate, final_verdict, rounds, refined_name, refined_def)
 
 
#Orchestrator
 
class OrchestratorAgent:
    """Drives Steps 1-3 of C3 for every posting."""
 
    def __init__(self) -> None:
        self.proposer = ProposerAgent()
        self.debate   = MultiAgentDebate()
 
    def process_posting(self, posting: str, ifm: IssueFrameMemory) -> list[DebateResult]:
        candidates    = self.proposer.propose(posting, ifm)
        existing_defs = [f.definition for f in ifm.all_frames]
        results: list[DebateResult] = []
 
        for candidate in candidates:
            if not candidate.definition:
                continue
            is_dup, sim = is_duplicate_definition(
                candidate.definition, existing_defs, config.SIMILARITY_THRESHOLD
            )
            if is_dup:
                logger.debug("  Candidate '%s' duplicate (sim=%.2f). Skip.", candidate.name, sim)
                continue
 
            result   = self.debate.run(candidate, ifm.all_frames)
            results.append(result)
            accepted = result.accepted_frame()
            if accepted:
                ifm.add_new_frame(accepted)
                existing_defs.append(accepted.definition)
                logger.info("  ✓ New IsF accepted: '%s'", accepted.name)
            else:
                ifm.rejected_frames.append(candidate)
 
        return results
 
 
class IDIsF:
    """
    Usage::
 
        module = IDIsF()
        ifm    = module.run(postings, known_isfs)
    """
 
    def __init__(self) -> None:
        self.investigator = InvestigatorAgent()
        self.qa_loop      = IntrospectiveQALoop()
        self.orchestrator = OrchestratorAgent()
 
    def _run_c1(self, postings: list[str], ifm: IssueFrameMemory) -> None:
        logger.info("classifying %d postings × %d known IsFs", len(postings), len(ifm.known_frames))
        for idx, posting in enumerate(postings, 1):
            logger.debug("  Posting %d/%d", idx, len(postings))
            for isf in ifm.known_frames:
                result = self.investigator.classify_with_qa(posting, isf, self.qa_loop)
                if result.verdict == InvestigatorVerdict.APPLICABLE:
                    isf.addressed_postings.append(posting)
                    isf.addressed_rationales.append(result.rationale)
                else:
                    isf.not_addressed_postings.append(posting)
                    isf.not_addressed_rationales.append(result.rationale)
 
    def _run_c3(self, postings: list[str], ifm: IssueFrameMemory) -> None:
        logger.info("discovering new IsFs across %d postings", len(postings))
        for idx, posting in enumerate(postings, 1):
            logger.debug("  Proposer – posting %d/%d", idx, len(postings))
            self.orchestrator.process_posting(posting, ifm)
 
    def run(self, postings: list[str], known_isfs: list[IssueFrame]) -> IssueFrameMemory:
        ifm = IssueFrameMemory(known_frames=known_isfs)
        self._run_c1(postings, ifm)
        self._run_c3(postings, ifm)
        logger.info("ID-IsF done. Known=%d  New=%d  Rejected=%d",
                    len(ifm.known_frames), len(ifm.new_frames), len(ifm.rejected_frames))
        return ifm
