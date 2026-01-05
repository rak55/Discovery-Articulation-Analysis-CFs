from sentence_transformers import SentenceTransformer, util
from config import SIMILARITY_THRESHOLD

# Initialize SBERT model 
# Using 'all-MiniLM-L6-v2' as a proxy for the SBERT mentioned in the paper
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_redundant(candidate_def: str, known_tas: list) -> bool:
    """
    Checks if a candidate TA is semantically similar to known TAs.
    Matches logic in Section 2.1, Step 2[cite: 448].
    """
    if not known_tas:
        return False
        
    known_defs = [ta['definition'] for ta in known_tas]
    
    # Encode
    candidate_emb = model.encode(candidate_def, convert_to_tensor=True)
    known_embs = model.encode(known_defs, convert_to_tensor=True)
    
    # Compute Cosine Similarity
    cosine_scores = util.cos_sim(candidate_emb, known_embs)
    max_score = float(cosine_scores.max())
    
    print(f"DEBUG: Max similarity score: {max_score}")
    
    # If score > threshold, it is a duplicate [cite: 454]
    return max_score >= SIMILARITY_THRESHOLD
