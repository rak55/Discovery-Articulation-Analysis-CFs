from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- ID-TA Structures ---

class TopicAspect(BaseModel):
    """Represents a known TA from memory[cite: 434]."""
    name: str
    definition: str

class IdentificationResult(BaseModel):
    """Output for the Investigator Agent[cite: 326]."""
    ta_name: str
    status: Literal["Applicable", "Not Applicable", "Uncertain"]
    rationale: str

class CandidateTA(BaseModel):
    """Output for the Proposer Agent[cite: 444]."""
    name: str = Field(description="Name of the new candidate TA")
    definition: str = Field(description="Definition of the new candidate TA")
    evidence: str = Field(description="Specific text span from the post serving as evidence")
    rationale: str = Field(description="Reasoning for why this is a new TA")

class DebateVerdict(BaseModel):
    """Output for the Judge Agent[cite: 461]."""
    verdict: Literal["Accept", "Reject", "Refine"]
    final_ta_name: Optional[str] = None
    final_ta_definition: Optional[str] = None
    reasoning: str

# --- Art-FRAME Structures  ---

class CommunicationFrame(BaseModel):
    """Output for the Articulation Agent[cite: 625]."""
    frame_text: str = Field(description="The articulated Communication Frame")
    rationale: str = Field(description="Causal interpretation provided by this frame")
    associated_tas: List[str] = Field(description="List of TA names this frame interprets")
