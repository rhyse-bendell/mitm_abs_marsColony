import csv
import json
from pathlib import Path


def aggregate_run_summaries(outputs_root, output_basename="aggregate_run_summaries"):
    outputs_path = Path(outputs_root)
    rows = []

    for session_dir in sorted(p for p in outputs_path.iterdir() if p.is_dir()):
        run_summary = session_dir / "measures" / "run_summary.json"
        if not run_summary.exists():
            continue
        payload = json.loads(run_summary.read_text(encoding="utf-8"))
        metadata = payload.get("run_metadata", {})
        outcomes = payload.get("outcomes", {})
        process = payload.get("process", {})
        external = payload.get("externalization_metrics", {})

        row = {
            "session": session_dir.name,
            "experiment_name": metadata.get("experiment_name"),
            "timestamp": metadata.get("timestamp"),
            "brain_backend": metadata.get("brain_backend"),
            "num_agents": metadata.get("num_agents"),
            "total_structures_attempted": outcomes.get("total_structures_attempted"),
            "total_structures_completed": outcomes.get("total_structures_completed"),
            "total_structures_validated_correct": outcomes.get("total_structures_validated_correct"),
            "total_structures_repaired_or_corrected": outcomes.get("total_structures_repaired_or_corrected"),
            "help_requests": process.get("help_requests"),
            "artifact_consultations": process.get("artifact_consultations"),
            "mismatch_detections": process.get("mismatch_detections"),
            "repair_or_correction_episodes": process.get("repair_or_correction_episodes"),
            "artifact_validation_rate": external.get("artifact_validation_rate"),
            "artifact_revision_or_repair_rate": external.get("artifact_revision_or_repair_rate"),
            "artifact_consultation_or_use_rate": external.get("artifact_consultation_or_use_rate"),
            "knowledge_artifact_mismatch_count": external.get("knowledge_artifact_mismatch_count"),
        }
        rows.append(row)

    aggregate_json = outputs_path / f"{output_basename}.json"
    aggregate_csv = outputs_path / f"{output_basename}.csv"

    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "count": len(rows)}, f, indent=2)

    if rows:
        with aggregate_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        aggregate_csv.write_text("", encoding="utf-8")

    print(f"✅ aggregate summary written: json={aggregate_json} csv={aggregate_csv} rows={len(rows)}")

    return {"json": str(aggregate_json), "csv": str(aggregate_csv), "count": len(rows)}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "Outputs"
    result = aggregate_run_summaries(outputs_root)
    print(f"✅ Aggregate metrics files written: {result}")
