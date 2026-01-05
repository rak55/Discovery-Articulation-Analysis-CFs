from crewai import Agent, Task, Crew
from schemas import IdentificationResult, CandidateTA, DebateVerdict
from config import MAX_LOOP_COUNT

class ID_TA_Module:
    def __init__(self):
        # 1. Investigator Agent 
        self.investigator = Agent(
            role='Investigator Agent',
            goal='Determine if known Topic Aspects are addressed in a post.',
            backstory="You are an expert analyst. You verify strict applicability of TAs to posts.",
            verbose=True,
            allow_delegation=False
        )

        # 2. Introspective Loop Agents 
        self.q_gen = Agent(
            role='Question Generator',
            goal='Generate Yes/No clarification questions for uncertain TAs.',
            backstory="You create precise questions to test TA criteria.",
            verbose=True
        )
        self.a_gen = Agent(
            role='Answer Generator',
            goal='Answer clarification questions based on text evidence.',
            backstory="You analyze posts to answer Yes/No/Cannot Answer.",
            verbose=True
        )

        # 3. Discovery Agents (Proposer, Champion, Critic, Judge) 
        self.proposer = Agent(
            role='Proposer Agent',
            goal='Propose NEW Topic Aspects not currently in memory.',
            backstory="You use Contrastive Chain-of-Thought to find gaps in current labels.",
            verbose=True
        )
        self.champion = Agent(
            role='Champion Agent',
            goal='Defend the validity of a new Candidate TA.',
            backstory="You argue for Applicability and Discriminant Validity.",
            verbose=True
        )
        self.critic = Agent(
            role='Critic Agent',
            goal='Invalidate the new Candidate TA.',
            backstory="You represent Parsimony. You check for Redundancy and Hallucination.",
            verbose=True
        )
        self.judge = Agent(
            role='Judge Agent',
            goal='Render a final verdict on the Candidate TA.',
            backstory="You decide: Accept, Reject, or Refine based on the debate.",
            verbose=True
        )

    def run_introspective_loop(self, post: str, ta_name: str, ta_def: str):
        """Resolves 'Uncertain' status via Q/A loop."""
        print(f"\n--- [ID-TA] Running Introspective Loop for {ta_name} ---")
        
        # Simplified simulation of the loop for this implementation
        task_q = Task(
            description=f"Generate a Yes/No question to test if '{ta_name}' ({ta_def}) is in: '{post}'",
            agent=self.q_gen,
            expected_output="A yes/no question string."
        )
        task_a = Task(
            description=f"Answer the question derived from the post content: '{post}'",
            agent=self.a_gen,
            expected_output="Yes, No, or Cannot Answer with rationale."
        )
        crew = Crew(agents=[self.q_gen, self.a_gen], tasks=[task_q, task_a])
        result = crew.kickoff()
        return result

    def identify_ta(self, post: str, ta_name: str, ta_def: str) -> IdentificationResult:
        """Run the Investigator Agent."""
        task = Task(
            description=f"Is TA '{ta_name}' ({ta_def}) addressed in post: '{post}'?",
            agent=self.investigator,
            expected_output="Applicable, Not Applicable, or Uncertain.",
            output_pydantic=IdentificationResult
        )
        crew = Crew(agents=[self.investigator], tasks=[task])
        result = crew.kickoff()
        
        # Handle Uncertainty
        if result.status == "Uncertain":
            clarification = self.run_introspective_loop(post, ta_name, ta_def)
            result.rationale += f" | QA Resolution: {clarification}"
            result.status = "Applicable" if "Yes" in str(clarification) else "Not Applicable"
            
        return result

    def discover_new_ta(self, post: str, known_tas_str: str):
        """Runs the Proposer -> Debate flow[cite: 439]."""
        
        # Step 1: Propose
        task_propose = Task(
            description=f"Given known TAs: {known_tas_str}, propose a NEW TA for post: '{post}' using Contrastive CoT.",
            agent=self.proposer,
            expected_output="Candidate TA details.",
            output_pydantic=CandidateTA
        )
        proposer_crew = Crew(agents=[self.proposer], tasks=[task_propose])
        candidate = proposer_crew.kickoff()
        
        # (Redundancy check via utils.py happens in main.py logic)

        # Step 3: Debate (Champion vs Critic -> Judge) 
        task_debate = Task(
            description=f"Debate validity of TA '{candidate.name}': {candidate.definition} for post: '{post}'. Champion defends, Critic attacks.",
            agent=self.judge, # In CrewAI, we can chain the context or use a hierarchical crew
            context=[task_propose],
            expected_output="Verdict: Accept, Reject, or Refine.",
            output_pydantic=DebateVerdict
        )
        
        # For simplicity in this file, we assume a sequential flow: Champion -> Critic -> Judge
        task_champ = Task(description=f"Defend '{candidate.name}'", agent=self.champion, expected_output="Defense arguments")
        task_crit = Task(description=f"Critique '{candidate.name}'", agent=self.critic, expected_output="Critique arguments")
        task_judge = Task(description="Issue Verdict based on Defense and Critique", agent=self.judge, context=[task_champ, task_crit], output_pydantic=DebateVerdict)

        debate_crew = Crew(agents=[self.champion, self.critic, self.judge], tasks=[task_champ, task_crit, task_judge])
        verdict = debate_crew.kickoff()
        
        return verdict
