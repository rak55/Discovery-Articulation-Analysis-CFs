MODEL_NAME  = "gpt-5-mini"
TEMPERATURE = 1.0   
 
#ID-IsF hyper-parameters 
MAX_QA_LOOP        = 2     # max rounds in the introspective Q/A loop
MAX_DEBATE_ROUNDS  = 3     # max rounds of the multi-agent debate
SIMILARITY_THRESHOLD = 0.75  # SK 
 
#Art-ICI hyper-parameters
ICI_ACCEPTANCE_THRESHOLD = 0.8   # minimum validity score to keep an ICI
 
#Directory paths
DATA_DIR     = "data"      
PROBLEMS_DIR = "problems"  # problems/<topic>.json – known IsF definitions
PROMPTS_DIR  = "prompts"   # prompts/<agent_name>.yaml – one file per agent
OUTPUT_DIR   = "output"   
 

SUPPORTED_TOPICS = ["immigration", "vaccine_hesitancy", "climate_change", "abortion"]
 
TOPIC_CSV_MAP: dict[str, str] = {
    "immigration":       "immigration.csv",
    "vaccine_hesitancy": "vaccine_hesitancy.csv",
    "climate_change":    "climate_change.csv",
    "abortion":          "abortion.csv",
}
 
TOPIC_PROBLEMS_MAP: dict[str, str] = {
    "immigration":       "immigration.json",
    "vaccine_hesitancy": "vaccine_hesitancy.json",
    "climate_change":    "climate_change.json",
    "abortion":          "abortion.json",
}
 

AGENT_PROMPT_MAP: dict[str, str] = {
    # ID-IsF agents
    "investigator":              "investigator.yaml",
    "investigator_resolver":     "investigator_resolver.yaml",
    "qa_question_generator":     "qa_question_generator.yaml",
    "qa_answer_generator":       "qa_answer_generator.yaml",
    "proposer":                  "proposer.yaml",
    "champion":                  "champion.yaml",
    "critic":                    "critic.yaml",
    "judge":                     "judge.yaml",
    "articulation":              "articulation.yaml",
    "paraphrase_police":         "paraphrase_police.yaml",
    "explainability_reconstruct":"explainability_reconstruct.yaml",
    "explainability_paraphrase": "explainability_paraphrase.yaml",
}
 
#Fine-tuned DeBERTa reward model
# Must contain:  config.json, tokenizer files, and model weights.
# Override via the  --deberta_model_path  CLI flag or set directly here.
DEBERTA_MODEL_PATH: str = "models/articulation_reward_model"
DEBERTA_MAX_LENGTH: int = 128
 
#Sentence-BERT model
SBERT_MODEL = "all-MiniLM-L6-v2"
 
RANDOM_SEED = 42
LOG_LEVEL   = "INFO"
 
