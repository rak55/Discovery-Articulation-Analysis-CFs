import utils
from id_ta_agents import ID_TA_Module
from art_frame_agents import Art_FRAME_Module
from schemas import TopicAspect

# example post
post_text = "I own a hardware store... If they leave, local commerce takes a massive hit."
known_tas = [
    {"name": "Economic", "definition": "The costs, benefits, or monetary implications..."},
    {"name": "Health & Safety", "definition": "Health and safety outcomes of a policy issue..."}
]

def run_pipeline():
    print("### STARTING ACL 2026 FRAMEWORK PIPELINE ###")
    
    # Initialize Modules
    id_ta = ID_TA_Module()
    art_frame = Art_FRAME_Module()
    
    # --- STAGE 1: ID-TA (Identification)---
    print("\n--- 1. IDENTIFYING KNOWN TAs ---")
    active_tas = []
    for ta in known_tas:
        result = id_ta.identify_ta(post_text, ta['name'], ta['definition'])
        print(f"TA '{ta['name']}': {result.status}")
        if result.status == "Applicable":
            active_tas.append(ta['name'])

    # --- STAGE 2: ID-TA (Discovery)---
    print("\n--- 2. DISCOVERING NEW TAs ---")
    # In a real loop, we would check if active_tas is empty or if we suspect missing frames
    verdict = id_ta.discover_new_ta(post_text, str(known_tas))
    
    if verdict.verdict == "Accept":
        print(f"NEW TA ACCEPTED: {verdict.final_ta_name}")
        # Add to active TAs for the next step
        active_tas.append(verdict.final_ta_name)
    elif verdict.verdict == "Refine":
        print(f"TA REFINED: {verdict.final_ta_name}")
        active_tas.append(verdict.final_ta_name)
    else:
        print("No new TA accepted.")

    # --- STAGE 3: Art-FRAME (Articulation)  ---
    print("\n--- 3. ARTICULATING FRAMES ---")
    if active_tas:
        # Pass the list of all valid TAs (known + newly discovered)
        final_result = art_frame.process_post(post_text, active_tas)
        print(f"Final Processing Complete. Output: {final_result}")
    else:
        print("No TAs found to articulate.")

if __name__ == "__main__":
    run_pipeline()
