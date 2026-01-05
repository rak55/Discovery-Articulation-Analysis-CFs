from crewai import Agent, Task, Crew
from schemas import CommunicationFrame

class Art_FRAME_Module:
    def __init__(self):
        # 1. Articulation Agent [cite: 625]
        self.articulator = Agent(
            role='Articulation Agent',
            goal='Articulate the Communication Frame (CF) evoked in the post.',
            backstory="You generate causal interpretations of TAs found in a post.",
            verbose=True
        )

        # 2. Explainability Agent (Reconstruction Test) [cite: 721]
        self.explainer = Agent(
            role='Explainability Agent',
            goal='Validate CFs by reconstructing them from their rationale.',
            backstory="You perform the Reconstruction Test to ensure CF validity.",
            verbose=True
        )

    def process_post(self, post: str, addressed_tas: list):
        """Run Articulation -> Validation."""
        
        # Step 1: Articulate
        task_art = Task(
            description=f"Post: '{post}'. TAs addressed: {addressed_tas}. Articulate the Communication Frame.",
            agent=self.articulator,
            expected_output="CF, Rationale, and Associated TAs.",
            output_pydantic=CommunicationFrame
        )
        
        # Step 2: Validate (Reconstruction) [cite: 722]
        # We pass the Rationale from the previous task, but NOT the frame text, to see if it can be reconstructed.
        task_val = Task(
            description=f"Using ONLY the rationale from the previous task, reconstruct the Communication Frame text.",
            agent=self.explainer,
            context=[task_art],
            expected_output="The reconstructed frame text."
        )

        crew = Crew(agents=[self.articulator, self.explainer], tasks=[task_art, task_val])
        result = crew.kickoff()
        
        # Result will be the output of the last task (Reconstruction). 
        # Ideally, you capture the intermediate output (the CF) via callback or parsing.
        return result
