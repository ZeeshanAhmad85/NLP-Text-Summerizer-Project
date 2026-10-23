import argparse
import sys
import time
from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
    STAGE_NAME as STAGE_01_NAME
)
from src.TextSummarizer.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
    STAGE_NAME as STAGE_02_NAME
)
from src.TextSummarizer.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
    STAGE_NAME as STAGE_03_NAME
)
from src.TextSummarizer.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
    STAGE_NAME as STAGE_04_NAME
)
from src.TextSummarizer.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
    STAGE_NAME as STAGE_05_NAME
)


def run_stage(stage_name: str, stage_callable):
    """Executes a single pipeline stage with execution logging and timing."""
    logger.info(f"\n{'='*20} Starting: {stage_name} {'='*20}")
    start_time = time.time()
    try:
        result = stage_callable()
        elapsed = time.time() - start_time
        logger.info(f"{'='*20} Completed: {stage_name} in {elapsed:.2f}s {'='*20}\n")
        return result
    except Exception as e:
        logger.exception(f"Stage '{stage_name}' failed with error: {e}")
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="NLP Text Summarizer Training and Evaluation Pipeline Runner"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "ingestion", "validation", "transformation", "trainer", "evaluation"],
        help="Pipeline stage to execute (default: 'all')"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Optional limit on evaluation test samples (useful for quick verification)"
    )

    args = parser.parse_args()
    logger.info(f"Pipeline invoked with stage='{args.stage}'")

    if args.stage in ["all", "ingestion"]:
        run_stage(STAGE_01_NAME, lambda: DataIngestionTrainingPipeline().main())

    if args.stage in ["all", "validation"]:
        run_stage(STAGE_02_NAME, lambda: DataValidationTrainingPipeline().main())

    if args.stage in ["all", "transformation"]:
        run_stage(STAGE_03_NAME, lambda: DataTransformationTrainingPipeline().main())

    if args.stage in ["all", "trainer"]:
        run_stage(STAGE_04_NAME, lambda: ModelTrainerTrainingPipeline().main())

    if args.stage in ["all", "evaluation"]:
        run_stage(STAGE_05_NAME, lambda: ModelEvaluationTrainingPipeline().main(sample_limit=args.eval_samples))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        logger.error(f"Execution terminated due to error: {err}")
        sys.exit(1)