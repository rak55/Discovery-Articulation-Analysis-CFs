from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
 

 
class InvestigatorVerdict(str, Enum):
    APPLICABLE     = "applicable"
    NOT_APPLICABLE = "not_applicable"
    UNCERTAIN      = "uncertain"
 
 
class DebateVerdict(str, Enum):
    ACCEPT  = "accept"
    REJECT  = "reject"
    CORRECT = "correct"
 
  
@dataclass
class IssueFrame:
    """
    Represents a single Issue Frame – either a known (pre-existing) one
    loaded from the 'problems/' folder, or a newly discovered one.
    """
    name:       str
    definition: str
    is_new:     bool = False
 
    addressed_postings:      List[str] = field(default_factory=list)
    addressed_rationales:    List[str] = field(default_factory=list)
    not_addressed_postings:  List[str] = field(default_factory=list)
    not_addressed_rationales:List[str] = field(default_factory=list)
 
    def __repr__(self) -> str:
        tag = "NEW" if self.is_new else "KNOWN"
        return (f"IssueFrame[{tag}](name={self.name!r}, "
                f"addressed={len(self.addressed_postings)})")
 
 
 
@dataclass
class InvestigatorResult:
    isf_name:  str
    verdict:   InvestigatorVerdict
    rationale: str
 
 
 
@dataclass
class QAPair:
    question:   str
    answer:     str          # "yes" | "no" | "cannot_answer"
    rationale:  str
 
 
#Candidate new IsF
 
@dataclass
class CandidateIsF:
    name:       str
    definition: str
    evidence:   str    # span(s) from the source posting
    rationale:  str    # why existing IsFs are insufficient
    posting:    str    # the source posting text
 
 
#Debate
 
@dataclass
class DebateRound:
    round_num:         int
    champion_argument: str
    critic_argument:   str
    judge_comment:     Optional[str] = None
 
 
@dataclass
class DebateResult:
    candidate:    CandidateIsF
    verdict:      DebateVerdict
    rounds:       List[DebateRound] = field(default_factory=list)
    refined_name: Optional[str] = None        # set when verdict == CORRECT
    refined_def:  Optional[str] = None        # set when verdict == CORRECT
 
    def accepted_frame(self) -> Optional[IssueFrame]:
        """
        Return the final IssueFrame if the debate accepted (or corrected)
        the candidate, otherwise None.
        """
        if self.verdict == DebateVerdict.REJECT:
            return None
        name = self.refined_name or self.candidate.name
        defn = self.refined_def  or self.candidate.definition
        isf  = IssueFrame(name=name, definition=defn, is_new=True)
        isf.addressed_postings.append(self.candidate.posting)
        return isf
 
 
#Issue Frame Memory (IFM)
 
@dataclass
class IssueFrameMemory:
    """
    Holds all known and newly accepted Issue Frames together with the
    full evidence accumulated while processing postings.
    """
    known_frames: List[IssueFrame] = field(default_factory=list)
    new_frames:   List[IssueFrame] = field(default_factory=list)
 
   
    rejected_frames: List[CandidateIsF] = field(default_factory=list)
 
    @property
    def all_frames(self) -> List[IssueFrame]:
        return self.known_frames + self.new_frames
 
    def get_frame(self, name: str) -> Optional[IssueFrame]:
        for f in self.all_frames:
            if f.name.lower() == name.lower():
                return f
        return None
 
    def add_new_frame(self, isf: IssueFrame) -> None:
        self.new_frames.append(isf)
 
    def all_definitions(self) -> List[tuple[str, str]]:
        """Return [(name, definition), ...] for every frame in memory."""
        return [(f.name, f.definition) for f in self.all_frames]
 
    def __repr__(self) -> str:
        return (f"IFM(known={len(self.known_frames)}, "
                f"new={len(self.new_frames)}, "
                f"rejected={len(self.rejected_frames)})")
 
 
#Issue Causal Interpretation (ICI)
 
@dataclass
class ICI:
    """
    A causal interpretation of one or more Issue Frames, as articulated by
    the Articulation Agent.
    """
    text:              str
    interpreted_isfs:  List[str]   # IsF names this ICI interprets
    rationale:         str
    evoked_by:         List[str] = field(default_factory=list)  # posting texts
    validity_score:    float     = 0.0    # assigned by the Articulation Reward Model
    is_valid:          bool      = False  # True after passing all quality checks
 
    def __repr__(self) -> str:
        return (f"ICI(isfs={self.interpreted_isfs}, "
                f"valid={self.is_valid}, text={self.text[:60]!r}...)")
 
 
#Posting Memory
 
@dataclass
class PostingRecord:
    """Associates one posting with the IsFs it addresses."""
    text:          str
    addressed_isfs: List[str] = field(default_factory=list)  # IsF names
 
 
@dataclass
class PostingMemory:
    records: List[PostingRecord] = field(default_factory=list)
 
    def get_by_text(self, text: str) -> Optional[PostingRecord]:
        for r in self.records:
            if r.text == text:
                return r
        return None
 
 
#ICI Memory 
 
@dataclass
class ICIMemory:
    """
    Maps each IsF name to the list of valid ICIs that interpret it,
    and tracks which postings evoke each ICI.
    """
    icis: List[ICI] = field(default_factory=list)
 
    def add(self, ici: ICI) -> None:
        self.icis.append(ici)
 
    def get_icis_for_isf(self, isf_name: str) -> List[ICI]:
        return [i for i in self.icis if isf_name in i.interpreted_isfs]
 
    def valid_icis(self) -> List[ICI]:
        return [i for i in self.icis if i.is_valid]
 
    def __repr__(self) -> str:
        v = len(self.valid_icis())
        return f"ICIMemory(total={len(self.icis)}, valid={v})"
 
 
@dataclass
class PipelineResult:
    topic:      str
    ifm:        IssueFrameMemory
    ici_memory: ICIMemory
