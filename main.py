"""Training entry point.

Runs ingestion -> validation -> transformation -> model training, then prints what the run
actually produced. Printed rather than only logged because this is what the API's training
panel tails: a run whose outcome lives only in a log file is a run whose outcome nobody reads.
"""

import sys

from src.exception.custom_exception import CustomException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        logging.info("Starting the Cardioplace BP Alerts training pipeline")
        artifact = TrainingPipeline().run_pipeline()

        print("\n=== training pipeline complete ===")
        print(f"bundle          : {artifact.bundle_file_path}")
        print(f"model           : {artifact.trained_model_file_path}")
        print(f"reports         : {artifact.report_dir}")
        print(f"selected family : {artifact.selected_family}")

        if artifact.offset_metric_artifact:
            o = artifact.offset_metric_artifact
            print(f"offset          : {o.label} warm={o.warm} k={o.k} q={o.q} | "
                  f"MAE {o.mae} | max threshold {o.max_threshold} "
                  f"(floor {o.emergency_floor}, holds={o.floor_holds})")
        if artifact.detector_metric_artifact:
            d = artifact.detector_metric_artifact[0]
            print(f"detector        : {d.detector} @{d.budget_pct}% | precision "
                  f"{d.precision:.3f} ({d.lift:.2f}x lift) | recall {d.recall:.3f} | "
                  f"lead {d.lead_days_median:.1f} d")
        if artifact.rule_engine_artifact:
            r = artifact.rule_engine_artifact
            print(f"rule engine     : {r.alerts_fired:,} alerts over {r.rows_evaluated:,} "
                  f"readings ({r.emergency_fired:,} emergency) | {r.evaluable_rules} "
                  f"evaluable / {r.blocked_rules} blocked rules")
        if artifact.safety_gate_artifact:
            g = artifact.safety_gate_artifact
            print(f"safety gates    : {g.gates_passed}/{g.gates_total} pass | "
                  f"{g.critical_failures} critical failures | promotable={g.promotable}")
            # The exit code carries the promotion verdict, so CI, a cron job or the API's
            # training panel can act on it without parsing stdout.
            if not g.promotable:
                print("\nPROMOTION BLOCKED -- final_model/ left untouched. Gate report: "
                      f"{g.gate_report_file_path}")
                sys.exit(2)
        print("\npromoted to final_model/model.pkl")
    except Exception as e:
        raise CustomException(e, sys)
