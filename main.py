from __future__ import annotations
 
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
 
import config
from id_isf   import IDIsF
from art_ici  import ArtICI
from schemas  import PipelineResult
from utils    import load_known_isfs, load_postings, save_json
 
 
def setup_logging(level: str = config.LOG_LEVEL) -> None:
    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
 
  
def _isf_to_dict(isf) -> dict:
    return {
        "name":                    isf.name,
        "definition":              isf.definition,
        "is_new":                  isf.is_new,
        "addressed_posting_count": len(isf.addressed_postings),
        "addressed_postings":      isf.addressed_postings,
        "addressed_rationales":    isf.addressed_rationales,
    }
 
 
def _ici_to_dict(ici) -> dict:
    return {
        "text":             ici.text,
        "interpreted_isfs": ici.interpreted_isfs,
        "rationale":        ici.rationale,
        "validity_score":   round(ici.validity_score, 3),
        "is_valid":         ici.is_valid,
        "evoked_by_count":  len(ici.evoked_by),
    }
 
 
def save_results(result: PipelineResult) -> None:
    """Persist the full pipeline output to JSON files."""
    topic = result.topic
    ifm   = result.ifm
    icim  = result.ici_memory
 
    ifm_out = {
        "topic":          topic,
        "known_isfs":     [_isf_to_dict(f) for f in ifm.known_frames],
        "new_isfs":       [_isf_to_dict(f) for f in ifm.new_frames],
        "rejected_count": len(ifm.rejected_frames),
    }
    save_json(ifm_out, f"{topic}_issue_frames.json")
 
    ici_out = {
        "topic":       topic,
        "total_icis":  len(icim.icis),
        "valid_icis":  len(icim.valid_icis()),
        "icis":        [_ici_to_dict(i) for i in icim.icis],
    }
    save_json(ici_out, f"{topic}_icis.json")
 

    analysis: list[dict] = []
    for isf in sorted(
        ifm.all_frames,
        key=lambda f: len(f.addressed_postings),
        reverse=True,
    ):
        isf_icis       = icim.get_icis_for_isf(isf.name)
        valid_isf_icis = [i for i in isf_icis if i.is_valid]
        analysis.append({
            "name":              isf.name,
            "is_new":            isf.is_new,
            "posting_count":     len(isf.addressed_postings),
            "validated_ici_count": len(valid_isf_icis),
            "most_evoked_ici":   (
                max(valid_isf_icis, key=lambda i: len(i.evoked_by)).text
                if valid_isf_icis else None
            ),
        })
 
    save_json({"topic": topic, "analysis": analysis}, f"{topic}_framing_analysis.json")
    print(f"\nResults saved to '{config.OUTPUT_DIR}/'")
 

  
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and analyse Issue Frames in social-media discourse."
    )
    parser.add_argument(
        "--topic",
        required=True,
        choices=config.SUPPORTED_TOPICS,
        help="Topic to analyse.",
    )
    parser.add_argument(
        "--max_posts",
        type=int,
        default=None,
        help="Limit the number of postings.",
    )
    parser.add_argument(
        "--skip_art_ici",
        action="store_true",
        help="Run only ID-IsF; skip the Art-ICI articulation step.",
    )
    parser.add_argument(
        "--deberta_model_path",
        default=None,
        help=(
            "Path to the fine-tuned DeBERTa reward-model directory. "
            "Defaults to config.DEBERTA_MODEL_PATH if not provided."
        ),
    )
    parser.add_argument(
        "--log_level",
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("  Topic  : %s", args.topic)
    logger.info("  Posts  : %s", args.max_posts or "all")
 
    known_isfs = load_known_isfs(args.topic)
    postings   = load_postings(args.topic, max_rows=args.max_posts)
 
    if not postings:
        logger.error("No postings loaded.")
        sys.exit(1)
 
    logger.info("\n Running ID-IsF")
    id_isf = IDIsF()
    ifm    = id_isf.run(postings, known_isfs)
 
    if args.skip_art_ici:
        from schemas import ICIMemory
        ici_memory = ICIMemory()
        logger.info("Art-ICI skipped.")
    else:
        logger.info("\n Running Art-ICI")
        art_ici    = ArtICI(deberta_model_path=args.deberta_model_path)
        ici_memory = art_ici.run(postings, ifm)
 
    result = PipelineResult(
        topic      = args.topic,
        ifm        = ifm,
        ici_memory = ici_memory,
    )
 
    save_results(result)
 
 
if __name__ == "__main__":
    main()
