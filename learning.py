"""Propose bounded policy changes, replay scenarios, and gate promotion."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets

from .evaluation import run_suite
from .policy import Policy
from .storage import evaluation_lock, write_json, write_text

REPAIRS = {
    "PREFERENCE_MISSED": {"field": "honor_soft_preferences", "value": True,
        "reason": "Persisted soft time preference was not reflected in ranked availability. Rank matching slots first, keeping hard filters unchanged."},
    "READ_TRANSIENT_UNRECOVERED": {"field": "read_retries", "value": 1,
        "reason": "An audited transient read failure prevented task completion. Retry read-only tools once; never retry a booking mutation blindly."},
}


def propose(before):
    patches = []
    for code, repair in REPAIRS.items():
        evidence = [{"run_id": r["id"], "transcript_hash": r["transcript_hash"], "failure_code": code}
                    for r in before["results"] if code in r["failure_codes"]]
        if evidence:
            patches.append({**repair, "failure_code": code, "evidence": evidence})
    return {"schema_version": 1, "source_policy": before["policy"]["version"],
            "generator": "failure-code repair library v1", "patches": patches,
            "unhandled_failure_codes": sorted({code for r in before["results"] for code in r["failure_codes"] if code not in REPAIRS})}


def apply_proposal(policy, proposal, before):
    if proposal.get("schema_version") != 1 or proposal.get("source_policy") != policy.version:
        raise ValueError("Proposal schema or source-policy mismatch")
    if before["policy"]["version"] != policy.version:
        raise ValueError("Evidence belongs to a different policy")
    values = asdict(policy)
    seen = set()
    for patch in proposal["patches"]:
        code = patch.get("failure_code")
        allowed = REPAIRS.get(code)
        if not allowed or patch.get("field") != allowed["field"] or type(patch.get("value")) is not type(allowed["value"]) or patch.get("value") != allowed["value"]:
            raise ValueError("Patch is outside the bounded repair allowlist")
        if patch["field"] in seen or not patch.get("evidence"):
            raise ValueError("Duplicate or unsupported patch")
        for item in patch["evidence"]:
            matching = [r for r in before["results"] if r["id"] == item.get("run_id") and r["transcript_hash"] == item.get("transcript_hash") and code in r["failure_codes"]]
            if item.get("failure_code") != code or not matching:
                raise ValueError("Patch evidence does not match an observed failure")
        seen.add(patch["field"])
        values[patch["field"]] = patch["value"]
    return Policy(**values)


def regressions(before, after):
    old = {(r["id"], c["name"]): c["passed"] for r in before["results"] for c in r["checks"]}
    new = {(r["id"], c["name"]): c["passed"] for r in after["results"] for c in r["checks"]}
    return [f"{run}:{check}" for (run, check), passed in old.items() if passed and not new.get((run, check), False)]


def promotion_gate(before, after, validation_before, validation_after):
    lost = regressions(before, after) + regressions(validation_before, validation_after)
    reasons = []
    if not after["safety_passed"] or not validation_after["safety_passed"] or any(
        c["passed"] is not True for suite in [after, validation_after]
        for run in suite["results"] for c in run["checks"] if c["category"] == "safety"
    ):
        reasons.append("A mandatory safety check failed")
    if after["score"] <= before["score"]:
        reasons.append("Training scenario score did not strictly improve")
    if validation_after["score"] < validation_before["score"]:
        reasons.append("Validation score fell")
    if lost:
        reasons.append("Previously passing checks regressed")
    for old, new in [(before, after), (validation_before, validation_after)]:
        coverage = lambda suite: {(r["id"], c["name"]) for r in suite["results"] for c in r["checks"]}
        if not old["results"] or not new["results"] or coverage(old) != coverage(new):
            reasons.append("Scenario or check coverage changed")
            break
    return {"promoted": not reasons, "reasons": reasons, "regressions": lost}


def improve(output="artifacts/latest", baseline=None):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(folder):
        return _improve(folder, baseline or Policy())


def _improve(folder, baseline):
    run_id = secrets.token_hex(12)
    before = run_suite(baseline, "training")
    proposal = propose(before)
    candidate = apply_proposal(baseline, proposal, before)
    after = run_suite(candidate, "training")
    # Validation is used only for the promotion gate, never by the proposer.
    validation_before = run_suite(baseline, "validation")
    validation_after = run_suite(candidate, "validation")
    gate = promotion_gate(before, after, validation_before, validation_after)
    active = candidate if gate["promoted"] else baseline
    summary = {"run_id": run_id, "generated_at": datetime.now(timezone.utc).isoformat(),
               "before": {k: before[k] for k in ["passed", "total", "score", "dimensions"]},
               "after": {k: after[k] for k in ["passed", "total", "score", "dimensions"]},
               "validation_before": {k: validation_before[k] for k in ["passed", "total", "score"]},
               "validation_after": {k: validation_after[k] for k in ["passed", "total", "score"]},
               "proposal": proposal, "gate": gate, "active_policy": active.to_dict()}
    files = {"before.json": before, "proposal.json": proposal, "after.json": after,
             "validation-before.json": validation_before, "validation-after.json": validation_after, "summary.json": summary}
    for name, value in files.items():
        value["run_id"] = run_id
        write_json(folder / name, value)
    rows = ["# CareLoop improvement report", "", f"Training: **{before['score']}% → {after['score']}%** ({before['passed']}/{before['total']} → {after['passed']}/{after['total']}).",
            f"Held-out validation: **{validation_before['score']}% → {validation_after['score']}%**.",
            f"Promotion: **{'accepted' if gate['promoted'] else 'rejected'}**. Regressed checks: {len(gate['regressions'])}.", "",
            "## Structured reinforcements", ""]
    for patch in proposal["patches"]:
        rows.append(f"- `{patch['failure_code']}` → `{patch['field']} = {json.dumps(patch['value'])}`. {patch['reason']} Evidence: {', '.join(e['run_id'] for e in patch['evidence'])}.")
    rows += ["", "## Same-scenario replay", "", "| Scenario | Before | After |", "|---|---|---|"]
    for old, new in zip(before["results"], after["results"]):
        rows.append(f"| {old['id']} | {'PASS' if old['passed'] else 'FAIL: ' + ', '.join(old['failure_codes'])} | {'PASS' if new['passed'] else 'FAIL'} |")
    rows += ["", "## Limits", "", "A passing finite suite is evidence of these behaviors, not clinical readiness. The oracle can inspect the synthetic DB and audit; it cannot certify medical triage, real identity, language coverage, or an external clinic API. No real patient records or medical diagnoses are used."]
    write_text(folder / "report.md", "\n".join(rows) + "\n")
    active.save(folder / "active_policy.json")
    return summary


def load_report(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    before = json.loads((folder / "before.json").read_text(encoding="utf-8"))
    after = json.loads((folder / "after.json").read_text(encoding="utf-8"))
    if any(not isinstance(value, dict) for value in [summary, before, after]):
        raise ValueError('Evaluation files must contain JSON objects')
    if any(suite.get("run_id") != summary.get("run_id") for suite in [before, after]):
        raise ValueError("Evaluation files belong to different runs")
    if summary["proposal"]["source_policy"] != before["policy"]["version"] or len(before["results"]) != len(after["results"]):
        raise ValueError("Evaluation files belong to different runs")
    summary["cases"] = []
    for initial, replay in zip(before["results"], after["results"]):
        if initial["id"] != replay["id"]:
            raise ValueError("Evaluation scenario order does not match")
        summary["cases"].append({"id": initial["id"], "description": initial["description"],
            "before_passed": initial["passed"], "after_passed": replay["passed"],
            "failure_codes": initial["failure_codes"], "checks": replay["checks"],
            "before_history": initial["history"], "after_history": replay["history"],
            "before_tools": initial["tools"], "after_tools": replay["tools"]})
    return summary
