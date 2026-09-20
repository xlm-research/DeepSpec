"""Evidence summaries using durations measured within one clock domain."""

import math

from .runtime import validate_message


def summarize_metrics(plan, events):
    totals = {"feature_tokens": 0, "complete_reads": 0}
    durations = {
        "writer_seconds": [],
        "read_wait_seconds": [],
        "training_rank_seconds": [],
        "wait_seconds": [],
    }
    missing, nodes, waits, seen = [], {}, {}, set()
    elapsed = None
    for event in events:
        validate_message(event, run_id=plan["run_id"], plan_hash=plan["plan_hash"])
        if event["event_id"] in seen:
            raise ValueError("Duplicate metric event")
        seen.add(event["event_id"])
        kind, data = event["event"], event["data"]
        mapping = {
            "feature_produced": "writer_seconds",
            "feature_read": "read_wait_seconds",
            "rank_update_completed": "training_rank_seconds",
            "wait": "wait_seconds",
        }
        if kind in mapping:
            duration = data.get("duration_seconds")
            if duration is None:
                missing.append(
                    {
                        "event_id": event["event_id"],
                        "metric": mapping[kind],
                        "reason": event.get("missing", {}).get(
                            "duration_seconds", "duration not recorded"
                        ),
                    }
                )
            elif (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError("Invalid metric duration")
            else:
                durations[mapping[kind]].append(duration)
                if kind == "wait":
                    waits[data["reason"]] = waits.get(data["reason"], 0) + duration
        if kind == "feature_produced":
            totals["feature_tokens"] += data["tokens"]
        elif kind == "feature_read" and data.get("verified") is True:
            totals["complete_reads"] += 1
        elif kind == "phase_duration" and data["phase"] == "production":
            if (
                elapsed is not None
                or not math.isfinite(data["duration_seconds"])
                or data["duration_seconds"] <= 0
            ):
                raise ValueError("Invalid or duplicate production duration")
            elapsed = data["duration_seconds"]
        elif kind == "resource_sample":
            node = event["sender_identity"]["node_id"]
            if node not in plan["node_budgets"]:
                raise ValueError("Unplanned metric node")
            item = nodes.setdefault(
                node,
                {
                    "samples": 0,
                    "peak_run_rss_bytes": None,
                    "minimum_headroom_bytes": None,
                    "approved_feature_bytes": plan["node_budgets"][node][
                        "feature_bound"
                    ],
                },
            )
            item["samples"] += 1
            if data["memory"] is not None:
                headroom = data["memory"]["headroom_bytes"]
                item["minimum_headroom_bytes"] = min(
                    headroom,
                    item["minimum_headroom_bytes"]
                    if item["minimum_headroom_bytes"] is not None
                    else headroom,
                )
            if data["processes"] is not None:
                rss = data["processes"]["rss_bytes"]
                item["peak_run_rss_bytes"] = max(rss, item["peak_run_rss_bytes"] or 0)
            missing.extend(
                {"event_id": event["event_id"], "metric": field, "reason": reason}
                for field, reason in event.get("missing", {}).items()
            )
    for node in plan["node_budgets"]:
        if node not in nodes:
            missing.append(
                {
                    "node_id": node,
                    "metric": "resource_sample",
                    "reason": "no independent node samples",
                }
            )
    if elapsed is None:
        missing.append(
            {
                "metric": "feature_tokens_per_second",
                "reason": "production duration not recorded",
            }
        )
    incomplete = {entry["metric"] for entry in missing}
    return {
        **totals,
        "feature_tokens_per_second": None
        if elapsed is None
        else totals["feature_tokens"] / elapsed,
        **{
            key: math.fsum(value) if value and key not in incomplete else None
            for key, value in durations.items()
        },
        "waits_by_reason": waits,
        "nodes": nodes,
        "missing": missing,
        "duration_basis": "explicit same-process durations; rank durations sum work, not wall time",
        "memory_basis": "observed run RSS may count shared pages more than once",
    }


def require_metrics(plan, events):
    report = summarize_metrics(plan, events)
    if report["missing"] or any(n["samples"] < 2 for n in report["nodes"].values()):
        raise ValueError(
            "Required runtime metrics or independent node samples are missing"
        )
    if report["feature_tokens"] != sum(s["length"] for s in plan["samples"]):
        raise ValueError("Production metrics do not cover the input plan")
    return report
