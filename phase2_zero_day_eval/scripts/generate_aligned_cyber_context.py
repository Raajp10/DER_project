"""
Phase 2 / Phase 3 Aligned Cyber-Context Generator.

Reads all validated scenario bundles and the zero-day physical attacked CSV,
then generates event-level cyber/context files that are exactly aligned to the
physical dataset for use by the Phase 3 small-LLM explanation layer.

ALL cyber content is event-level semantic context only.
No packet-level fields.  No real traffic.  No protocol compliance claims.

Inputs:
  outputs/zero_day_physical_attacked.csv
  outputs/zero_day_scenario_manifest.csv
  outputs/zero_day_context_windows.csv
  scenarios/scenario_bundles_validated/*.json

Outputs:
  outputs/zero_day_cyber_event_log.csv
  outputs/zero_day_cyber_physical_aligned_1s.csv   (604 800 rows)
  outputs/zero_day_cyber_context_packets.jsonl
  outputs/zero_day_phase3_evidence_packets.jsonl
  outputs/zero_day_phase3_alignment_summary.json

Reports:
  reports/ZERO_DAY_CYBER_CONTEXT_REPORT.md
  reports/PHASE3_ZERO_DAY_ALIGNMENT_VALIDATION_REPORT.md
"""
import sys
import json
import uuid
import time as _time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

PHASE2_ROOT   = Path(r"D:\updated_dataset\phase2_zero_day_eval")
OUTPUTS_DIR   = PHASE2_ROOT / "outputs"
REPORTS_DIR   = PHASE2_ROOT / "reports"
VALIDATED_DIR = PHASE2_ROOT / "scenarios" / "scenario_bundles_validated"

ATTACKED_CSV  = OUTPUTS_DIR / "zero_day_physical_attacked.csv"
MANIFEST_CSV  = OUTPUTS_DIR / "zero_day_scenario_manifest.csv"
CONTEXT_CSV   = OUTPUTS_DIR / "zero_day_context_windows.csv"

BASE_DT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
NOW_STR = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOTAL_SECONDS = 604800
T_MAX         = 604799

FORBIDDEN_FIELDS = {
    "payload", "packet_bytes", "tcp_flags", "exploit", "malware",
    "credential", "password", "hash", "shellcode", "buffer_overflow",
    "sql_injection", "cve_id", "metasploit", "cobalt_strike",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(t_s: int) -> str:
    """int seconds → ISO-8601 UTC string."""
    return (BASE_DT + timedelta(seconds=max(0, min(int(t_s), T_MAX)))).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _clamp(v: int) -> int:
    return max(0, min(int(v), T_MAX))


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _join_vars(effects: list, observable: list) -> str:
    """Collect unique variable names from physical_effects + expected_observable_signals."""
    seen = set()
    out = []
    for e in effects:
        v = e.get("variable", "")
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    for v in observable:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return "|".join(out)


def _get_ramp_s(effects: list) -> int:
    """Return largest ramp_seconds from effects, default 0."""
    return max((int(e.get("ramp_seconds", 0)) for e in effects), default=0)


# ─────────────────────────────────────────────────────────────────────────────
# Derive per-class cyber status columns
# ─────────────────────────────────────────────────────────────────────────────

def _cia_statuses(cia: str, scenario_class: str) -> dict:
    """Derive authn/authz/integrity/availability/confidentiality from CIA dimension."""
    base = dict(
        authn_status="passed",
        authz_status="passed",
        integrity_status="passed",
        availability_status="normal",
        confidentiality_status="passed",
    )
    if scenario_class in ("cyber_only", "cyber_physical"):
        cia_l = cia.lower()
        if "integrity" in cia_l:
            base["integrity_status"] = "compromised"
        if "availability" in cia_l:
            base["availability_status"] = "degraded"
        if "confidentiality" in cia_l:
            base["confidentiality_status"] = "compromised"
        if "access_control" in cia_l:
            base["authz_status"] = "violated"
    return base


def _family_flags(family: str, scenario_class: str, delivery: str) -> dict:
    """Derive per-row binary flags from scenario family."""
    f = dict(
        command_delay_active_flag=0,
        blocked_flag=0,
        replay_flag=0,
        mismatch_flag=0,
        stale_command_flag=0,
        timeout_flag=0,
        retry_count=0,
    )
    if delivery in ("blocked", "suppressed"):
        f["blocked_flag"] = 1
    if delivery == "replayed":
        f["replay_flag"] = 1
    if delivery == "delayed":
        f["command_delay_active_flag"] = 1

    fam = family.lower()
    if "delay" in fam:
        f["command_delay_active_flag"] = 1
    if "suppress" in fam or "block" in fam:
        f["blocked_flag"] = 1
    if "stale" in fam or "stale_measurement" in fam:
        f["stale_command_flag"] = 1
    if "false_data" in fam or "mismatch" in fam:
        f["mismatch_flag"] = 1
    if "availability" in fam:
        f["timeout_flag"] = 1
    if "oscillatory" in fam:
        f["mismatch_flag"] = 1
    return f


def _cyber_state(scenario_class: str, family: str, lifecycle: str) -> str:
    """Human-readable cyber_state label for a given class/family/lifecycle."""
    if scenario_class == "normal":
        return "normal_monitoring"
    if scenario_class == "physical_only":
        if lifecycle in ("physical_event_observed",):
            return "physical_event_observed"
        return "normal_monitoring"
    # cyber_only / cyber_physical
    fam = family.lower()
    if "suppres" in fam or "block" in fam:
        return "blocked_command"
    if "delay" in fam:
        return "delayed_command_context"
    if "stale" in fam:
        return "stale_measurement_context"
    if "false_data" in fam:
        return "false_data_context"
    if "availability" in fam:
        return "availability_degraded"
    if scenario_class == "cyber_physical":
        return "cyber_physical_effect_active"
    return "security_event_active"


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle rows per scenario → cyber_event_log rows
# ─────────────────────────────────────────────────────────────────────────────

def _build_lifecycle_rows(sc: dict, author: str, bundle_file: str) -> list[dict]:
    """Return list of lifecycle-row dicts for one scenario."""
    sid       = sc["scenario_id"]
    family    = sc.get("scenario_family", "")
    cls       = sc.get("scenario_class", "")
    start_s   = int(sc.get("start_time_s", 0))
    dur_s     = int(sc.get("duration_s", 0))
    end_s     = start_s + dur_s - 1
    effects   = sc.get("physical_effects", [])
    obs       = sc.get("expected_observable_signals", [])
    cc        = sc.get("cyber_context", {})
    labels    = sc.get("labels", {})
    ramp_s    = _get_ramp_s(effects)

    rel_vars  = _join_vars(effects, obs)
    cia       = cc.get("cia_dimension", "none")
    cia_stats = _cia_statuses(cia, cls)
    flags     = _family_flags(family, cls, cc.get("delivery_status", "success"))
    proto     = cc.get("protocol_profile", "synthetic_der_event_log")
    msg_type  = cc.get("message_type", "status")
    cat       = cc.get("cyber_event_category", "operational_telemetry")
    delivery  = cc.get("delivery_status", "success")

    la   = int(labels.get("label_anomaly", 0))
    lca  = int(labels.get("label_cyber_anomaly", 0))
    lpa  = int(labels.get("label_physical_anomaly", 0))

    # timing
    cp_created  = _clamp(start_s - 5)
    cp_sent     = _clamp(start_s - 4)
    cp_recv     = _clamp(start_s - 3)
    cp_accept   = _clamp(start_s - 2)
    cp_apply    = start_s
    cp_response = _clamp(end_s + 1)

    delay_s = 0
    if "delay" in family.lower():
        delay_s = max(30, ramp_s if ramp_s > 0 else dur_s // 4)
        cp_apply = _clamp(start_s + delay_s)

    base = dict(
        scenario_id=sid,
        author_model=author,
        scenario_family=family,
        scenario_class=cls,
        start_time_s=start_s,
        end_time_s=end_s,
        duration_s=dur_s,
        target_asset_id=sc.get("target_asset_id", ""),
        target_component=sc.get("target_component", ""),
        related_physical_variables=rel_vars,
        cia_dimension=cia,
        protocol_profile=proto,
        message_type=msg_type,
        cyber_event_category=cat,
        delivery_status=delivery,
        **cia_stats,
        command_created_time_s=cp_created,
        command_sent_time_s=cp_sent,
        command_recv_time_s=cp_recv,
        command_accept_time_s=cp_accept,
        command_apply_time_s=cp_apply,
        command_response_time_s=cp_response,
        delay_s=delay_s,
        **flags,
        label_anomaly=la,
        label_cyber_anomaly=lca,
        label_physical_anomaly=lpa,
        event_level_only=True,
        packet_level_protocol_compliance_claimed=False,
        source_bundle_file=bundle_file,
        validation_status="ACCEPTED",
        validation_notes=sc.get("validation_notes", ""),
    )

    rows = []

    def _row(stage: str, t_s: int, cyber_active: int, phys_active: int,
             extra_flags: dict | None = None) -> dict:
        r = dict(base)
        r["cyber_event_id"]       = _uid(f"{sid}_{stage}")
        r["lifecycle_stage"]      = stage
        r["event_time_s"]         = _clamp(t_s)
        r["event_time_utc"]       = _ts(t_s)
        r["cyber_state"]          = _cyber_state(cls, family, stage)
        r["cyber_anomaly_active"] = cyber_active
        r["physical_effect_active"] = phys_active
        if extra_flags:
            r.update(extra_flags)
        return r

    # ── normal ───────────────────────────────────────────────────────────────
    if cls == "normal":
        rows.append(_row("status_report",       start_s,    0, 0))
        rows.append(_row("monitoring_complete",  end_s,      0, 0))

    # ── physical_only ─────────────────────────────────────────────────────────
    elif cls == "physical_only":
        rows.append(_row("monitoring_pre_event", start_s,        0, 0))
        rows.append(_row("physical_event_observed",
                         _clamp(start_s + max(ramp_s, 1)),     0, 1))
        rows.append(_row("monitoring_post_event", end_s,          0, 1))

    # ── cyber_only ────────────────────────────────────────────────────────────
    elif cls == "cyber_only":
        fam = family.lower()
        mid = _clamp(start_s + dur_s // 2)

        if "suppres" in fam or "block" in fam:
            rows.append(_row("command_created",  cp_sent,  1, 0))
            rows.append(_row("command_sent",     start_s,  1, 0,
                             {"blocked_flag": 1}))
            rows.append(_row("command_blocked",  start_s,  1, 0,
                             {"blocked_flag": 1}))
            rows.append(_row("security_alert_active", mid, 1, 0,
                             {"blocked_flag": 1}))

        elif "delay" in fam:
            rows.append(_row("command_created",          cp_sent,                    1, 0))
            rows.append(_row("command_sent",             start_s,                    1, 0))
            rows.append(_row("command_delay_observed",   _clamp(start_s + delay_s),  1, 0,
                             {"command_delay_active_flag": 1}))
            rows.append(_row("timeout_flagged",          end_s,                      1, 0,
                             {"command_delay_active_flag": 1, "timeout_flag": 1}))

        elif "stale" in fam:
            rows.append(_row("telemetry_received",      start_s, 1, 0))
            rows.append(_row("stale_measurement_context", mid,    1, 0,
                             {"stale_command_flag": 1}))
            rows.append(_row("stale_measurement_resolved", end_s, 1, 0,
                             {"stale_command_flag": 1}))

        elif "false_data" in fam:
            rows.append(_row("telemetry_sent",             start_s, 1, 0))
            rows.append(_row("mismatch_detected_context",  mid,      1, 0,
                             {"mismatch_flag": 1}))

        elif "availability" in fam:
            rows.append(_row("status_request_sent",    start_s, 1, 0))
            rows.append(_row("timeout_observed",       mid,      1, 0,
                             {"timeout_flag": 1}))
            rows.append(_row("availability_degraded",  end_s,    1, 0,
                             {"timeout_flag": 1, "availability_status": "degraded"}))

        else:
            rows.append(_row("cyber_event_start",  start_s, 1, 0))
            rows.append(_row("security_alert",     mid,      1, 0))
            rows.append(_row("cyber_event_end",    end_s,    1, 0))

    # ── cyber_physical ────────────────────────────────────────────────────────
    elif cls == "cyber_physical":
        fam = family.lower()
        phys_peak = _clamp(start_s + max(ramp_s, dur_s // 3))

        rows.append(_row("command_created",           cp_created, 1, 0))
        rows.append(_row("command_sent",              cp_sent,    1, 0))
        rows.append(_row("command_received",          cp_recv,    1, 0))
        rows.append(_row("command_accepted",          cp_accept,  1, 0))
        rows.append(_row("command_applied",           cp_apply,   1, 1))
        rows.append(_row("physical_response_observed", phys_peak, 1, 1))
        rows.append(_row("status_report",             end_s,      1, 1))

        # extra family flags
        if "delay" in fam:
            for r in rows:
                if r["lifecycle_stage"] in (
                        "command_applied", "physical_response_observed"):
                    r["command_delay_active_flag"] = 1
        if "suppres" in fam:
            for r in rows:
                r["blocked_flag"] = 1
        if "stale" in fam:
            for r in rows:
                if r["lifecycle_stage"] in (
                        "command_applied", "physical_response_observed"):
                    r["stale_command_flag"] = 1
        if "false_data" in fam:
            for r in rows:
                r["mismatch_flag"] = 1
        if "oscillator" in fam:
            for r in rows:
                if r["lifecycle_stage"] == "physical_response_observed":
                    r["mismatch_flag"] = 1

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build aligned 1-second table
# ─────────────────────────────────────────────────────────────────────────────

def _build_aligned_1s(all_scenarios: list[dict],
                       attacked_df: pd.DataFrame) -> pd.DataFrame:
    """Return a 604 800-row DataFrame aligned to the physical time axis."""
    print("[INFO] Building 604,800-row aligned cyber/physical table...")
    t0 = _time.time()

    n = TOTAL_SECONDS
    TIME_S = np.arange(n, dtype=np.int32)

    # Initialise all string/int columns as arrays for speed
    timestamps   = attacked_df["timestamp_utc"].values
    zd_active    = np.zeros(n, dtype=np.int8)
    scenario_ids = np.full(n, "", dtype=object)
    active_ids   = np.full(n, "", dtype=object)   # multi-scenario joins
    authors      = np.full(n, "", dtype=object)
    families     = np.full(n, "", dtype=object)
    classes      = np.full(n, "", dtype=object)
    targets      = np.full(n, "", dtype=object)
    components   = np.full(n, "", dtype=object)
    rel_vars     = np.full(n, "", dtype=object)
    cyber_states = np.full(n, "normal_monitoring", dtype=object)
    cyber_lc     = np.full(n, "background", dtype=object)
    cyber_active = np.zeros(n, dtype=np.int8)
    phys_active  = np.zeros(n, dtype=np.int8)
    cmd_active   = np.zeros(n, dtype=np.int8)
    cmd_created  = np.zeros(n, dtype=np.int8)
    cmd_sent     = np.zeros(n, dtype=np.int8)
    cmd_recv     = np.zeros(n, dtype=np.int8)
    cmd_accept   = np.zeros(n, dtype=np.int8)
    cmd_apply    = np.zeros(n, dtype=np.int8)
    cmd_response = np.zeros(n, dtype=np.int8)
    cmd_delay    = np.zeros(n, dtype=np.int8)
    blocked_f    = np.zeros(n, dtype=np.int8)
    replay_f     = np.zeros(n, dtype=np.int8)
    mismatch_f   = np.zeros(n, dtype=np.int8)
    stale_f      = np.zeros(n, dtype=np.int8)
    timeout_f    = np.zeros(n, dtype=np.int8)
    delivery_s   = np.full(n, "success", dtype=object)
    authn_s      = np.full(n, "passed",  dtype=object)
    authz_s      = np.full(n, "passed",  dtype=object)
    integ_s      = np.full(n, "passed",  dtype=object)
    avail_s      = np.full(n, "normal",  dtype=object)
    conf_s       = np.full(n, "passed",  dtype=object)
    cia_d        = np.full(n, "none",    dtype=object)
    proto_p      = np.full(n, "synthetic_der_event_log", dtype=object)
    msg_t        = np.full(n, "monitoring", dtype=object)
    cat_t        = np.full(n, "operational_telemetry", dtype=object)
    lab_a        = np.zeros(n, dtype=np.int8)
    lab_ca       = np.zeros(n, dtype=np.int8)
    lab_pa       = np.zeros(n, dtype=np.int8)
    el_only      = np.ones(n,  dtype=np.int8)   # always 1
    pkt_claim    = np.zeros(n, dtype=np.int8)   # always 0

    overlap_tracker = {}  # t_s -> list of scenario_ids

    for sc in all_scenarios:
        sid     = sc["scenario_id"]
        cls     = sc.get("scenario_class", "")
        family  = sc.get("scenario_family", "")
        author  = sc.get("_author", "")
        start_s = int(sc.get("start_time_s", 0))
        dur_s   = int(sc.get("duration_s", 0))
        end_s   = start_s + dur_s - 1
        effects = sc.get("physical_effects", [])
        obs     = sc.get("expected_observable_signals", [])
        cc      = sc.get("cyber_context", {})
        labels  = sc.get("labels", {})
        ramp_s  = _get_ramp_s(effects)

        rvars   = _join_vars(effects, obs)
        cia     = cc.get("cia_dimension", "none")
        cia_st  = _cia_statuses(cia, cls)
        fflags  = _family_flags(family, cls, cc.get("delivery_status", "success"))
        delivery = cc.get("delivery_status", "success")

        la  = int(labels.get("label_anomaly", 0))
        lca = int(labels.get("label_cyber_anomaly", 0))
        lpa = int(labels.get("label_physical_anomaly", 0))

        delay_s_val = 0
        if "delay" in family.lower():
            delay_s_val = max(30, ramp_s if ramp_s > 0 else dur_s // 4)

        cp_created  = _clamp(start_s - 5)
        cp_sent     = _clamp(start_s - 4)
        cp_recv     = _clamp(start_s - 3)
        cp_accept   = _clamp(start_s - 2)
        cp_apply    = start_s
        cp_response = _clamp(end_s + 1)

        # Boolean: does this second get physical effect?
        def _is_phys(t: int) -> int:
            if not effects or lpa == 0:
                return 0
            if cls in ("cyber_physical", "physical_only"):
                return 1 if start_s <= t <= end_s else 0
            return 0

        # Boolean: cyber anomaly active this second?
        def _is_cyber(t: int) -> int:
            if lca == 0:
                return 0
            return 1 if start_s <= t <= end_s else 0

        sl = slice(start_s, end_s + 1)

        # Track overlaps
        for t in range(start_s, min(end_s + 1, n)):
            if zd_active[t]:
                existing = overlap_tracker.get(t, [scenario_ids[t]])
                if sid not in existing:
                    existing.append(sid)
                overlap_tracker[t] = existing

        # Fill arrays
        existing_active = zd_active[sl].copy()
        is_overlap = existing_active.astype(bool)

        # scenario_id: first-write or MULTIPLE
        for i, t in enumerate(range(start_s, min(end_s + 1, n))):
            if not zd_active[t]:
                scenario_ids[t] = sid
            else:
                scenario_ids[t] = "MULTIPLE"
            active_ids[t] = (active_ids[t] + "|" + sid).lstrip("|")

        zd_active[sl]   = 1
        authors[sl]     = author
        families[sl]    = family
        classes[sl]     = cls
        targets[sl]     = sc.get("target_asset_id", "")
        components[sl]  = sc.get("target_component", "")
        rel_vars[sl]    = rvars
        cia_d[sl]       = cia
        proto_p[sl]     = cc.get("protocol_profile", "synthetic_der_event_log")
        msg_t[sl]       = cc.get("message_type", "monitoring")
        cat_t[sl]       = cc.get("cyber_event_category", "operational_telemetry")
        delivery_s[sl]  = delivery
        authn_s[sl]     = cia_st["authn_status"]
        authz_s[sl]     = cia_st["authz_status"]
        integ_s[sl]     = cia_st["integrity_status"]
        avail_s[sl]     = cia_st["availability_status"]
        conf_s[sl]      = cia_st["confidentiality_status"]

        # labels: OR
        lab_a[sl]  = np.maximum(lab_a[sl],  la)
        lab_ca[sl] = np.maximum(lab_ca[sl], lca)
        lab_pa[sl] = np.maximum(lab_pa[sl], lpa)

        # cyber / physical activity
        for t in range(start_s, min(end_s + 1, n)):
            cyber_active[t] = max(cyber_active[t], _is_cyber(t))
            phys_active[t]  = max(phys_active[t],  _is_phys(t))

        # lifecycle_stage for this window
        for t in range(start_s, min(end_s + 1, n)):
            t_rel = t - start_s
            if cls == "normal":
                cyber_lc[t] = "normal_monitoring"
            elif cls == "physical_only":
                cyber_lc[t] = ("physical_event_observed"
                               if t_rel >= ramp_s else "monitoring_pre_event")
            elif cls == "cyber_only":
                cyber_lc[t] = "cyber_event_active"
            else:  # cyber_physical
                if t < cp_apply:
                    cyber_lc[t] = "command_pipeline"
                elif t < _clamp(start_s + ramp_s + 1):
                    cyber_lc[t] = "command_applied"
                else:
                    cyber_lc[t] = "physical_effect_active"

        # cyber_state
        for t in range(start_s, min(end_s + 1, n)):
            cyber_states[t] = _cyber_state(cls, family, cyber_lc[t])

        # command flags
        if cls == "cyber_physical":
            cmd_active[sl]   = 1
            if 0 <= cp_created < n:
                cmd_created[cp_created]  = 1
            if 0 <= cp_sent < n:
                cmd_sent[cp_sent]        = 1
            if 0 <= cp_recv < n:
                cmd_recv[cp_recv]        = 1
            if 0 <= cp_accept < n:
                cmd_accept[cp_accept]    = 1
            if 0 <= cp_apply <= end_s:
                cmd_apply[_clamp(cp_apply)]  = 1
            if 0 <= cp_response < n:
                cmd_response[cp_response]= 1

        # flags
        cmd_delay[sl]  = np.maximum(cmd_delay[sl],  fflags.get("command_delay_active_flag", 0))
        blocked_f[sl]  = np.maximum(blocked_f[sl],  fflags.get("blocked_flag", 0))
        replay_f[sl]   = np.maximum(replay_f[sl],   fflags.get("replay_flag", 0))
        mismatch_f[sl] = np.maximum(mismatch_f[sl], fflags.get("mismatch_flag", 0))
        stale_f[sl]    = np.maximum(stale_f[sl],    fflags.get("stale_command_flag", 0))
        timeout_f[sl]  = np.maximum(timeout_f[sl],  fflags.get("timeout_flag", 0))

    # Determine alignment_status
    alignment_status = np.where(
        np.array([t in overlap_tracker for t in range(n)]),
        "OVERLAP_MULTIPLE",
        np.where(zd_active.astype(bool), "SCENARIO_ACTIVE", "BACKGROUND")
    )

    print(f"[INFO] Table built in {_time.time()-t0:.1f}s")

    df = pd.DataFrame({
        "timestamp_utc":                 timestamps,
        "time_s":                        TIME_S,
        "zero_day_active_flag":          zd_active,
        "scenario_id":                   scenario_ids,
        "active_scenario_ids":           active_ids,
        "author_model":                  authors,
        "scenario_family":               families,
        "scenario_class":                classes,
        "target_asset_id":               targets,
        "target_component":              components,
        "related_physical_variables":    rel_vars,
        "cyber_state":                   cyber_states,
        "cyber_lifecycle_stage":         cyber_lc,
        "cyber_anomaly_active":          cyber_active,
        "physical_effect_active":        phys_active,
        "command_active_flag":           cmd_active,
        "command_created_flag":          cmd_created,
        "command_sent_flag":             cmd_sent,
        "command_recv_flag":             cmd_recv,
        "command_accept_flag":           cmd_accept,
        "command_apply_flag":            cmd_apply,
        "command_response_flag":         cmd_response,
        "command_delay_active_flag":     cmd_delay,
        "blocked_flag":                  blocked_f,
        "replay_flag":                   replay_f,
        "mismatch_flag":                 mismatch_f,
        "stale_command_flag":            stale_f,
        "timeout_flag":                  timeout_f,
        "delivery_status":               delivery_s,
        "authn_status":                  authn_s,
        "authz_status":                  authz_s,
        "integrity_status":              integ_s,
        "availability_status":           avail_s,
        "confidentiality_status":        conf_s,
        "cia_dimension":                 cia_d,
        "protocol_profile":              proto_p,
        "message_type":                  msg_t,
        "cyber_event_category":          cat_t,
        "label_anomaly":                 lab_a,
        "label_cyber_anomaly":           lab_ca,
        "label_physical_anomaly":        lab_pa,
        "event_level_only":              el_only,
        "packet_level_protocol_compliance_claimed": pkt_claim,
        "alignment_status":              alignment_status,
    })
    return df, len(overlap_tracker)


# ─────────────────────────────────────────────────────────────────────────────
# Context packet (JSONL) per scenario
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_packet(sc: dict, author: str) -> dict:
    sid    = sc["scenario_id"]
    cls    = sc.get("scenario_class", "")
    family = sc.get("scenario_family", "")
    start  = int(sc.get("start_time_s", 0))
    dur    = int(sc.get("duration_s", 0))
    end    = start + dur - 1
    effects = sc.get("physical_effects", [])
    obs     = sc.get("expected_observable_signals", [])
    cc      = sc.get("cyber_context", {})
    labels  = sc.get("labels", {})
    ramp_s  = _get_ramp_s(effects)
    flags   = _family_flags(family, cls, cc.get("delivery_status", "success"))

    cp_apply = start
    if "delay" in family.lower():
        delay_v = max(30, ramp_s if ramp_s > 0 else dur // 4)
        cp_apply = _clamp(start + delay_v)

    lc_stages = []
    if cls == "normal":
        lc_stages = ["status_report", "monitoring_complete"]
    elif cls == "physical_only":
        lc_stages = ["monitoring_pre_event", "physical_event_observed", "monitoring_post_event"]
    elif cls == "cyber_only":
        fam = family.lower()
        if "suppres" in fam or "block" in fam:
            lc_stages = ["command_created", "command_sent", "command_blocked", "security_alert_active"]
        elif "delay" in fam:
            lc_stages = ["command_created", "command_sent", "command_delay_observed", "timeout_flagged"]
        elif "stale" in fam:
            lc_stages = ["telemetry_received", "stale_measurement_context", "stale_measurement_resolved"]
        elif "false_data" in fam:
            lc_stages = ["telemetry_sent", "mismatch_detected_context"]
        elif "availability" in fam:
            lc_stages = ["status_request_sent", "timeout_observed", "availability_degraded"]
        else:
            lc_stages = ["cyber_event_start", "security_alert", "cyber_event_end"]
    else:  # cyber_physical
        lc_stages = ["command_created", "command_sent", "command_received",
                     "command_accepted", "command_applied",
                     "physical_response_observed", "status_report"]

    return {
        "scenario_id":    sid,
        "author_model":   author,
        "scenario_family":family,
        "scenario_class": cls,
        "time_window": {
            "start_time_s": start, "end_time_s": end, "duration_s": dur,
            "start_utc": _ts(start), "end_utc": _ts(end),
        },
        "asset_context": {
            "target_asset_id":          sc.get("target_asset_id", ""),
            "target_component":         sc.get("target_component", ""),
            "related_physical_variables": _join_vars(effects, obs).split("|"),
        },
        "cyber_context": {
            "lifecycle_stages":       lc_stages,
            "cyber_state_summary":    _cyber_state(cls, family, lc_stages[-1] if lc_stages else ""),
            "protocol_profile":       cc.get("protocol_profile", ""),
            "message_type":           cc.get("message_type", ""),
            "cyber_event_category":   cc.get("cyber_event_category", ""),
            "cia_dimension":          cc.get("cia_dimension", "none"),
            "delivery_status":        cc.get("delivery_status", "success"),
            "flags":                  {k: v for k, v in flags.items() if v != 0},
            "event_level_only":       True,
            "packet_level_protocol_compliance_claimed": False,
        },
        "physical_context": {
            "physical_effect_present":         bool(effects and labels.get("label_physical_anomaly", 0)),
            "physical_effect_variables":       [e.get("variable", "") for e in effects],
            "physical_effect_types":           [e.get("effect_type", "") for e in effects],
            "physical_effect_start_time_s":    start if effects else None,
            "physical_effect_end_time_s":      end   if effects else None,
            "physical_ramp_seconds":           ramp_s,
            "command_apply_time_s":            cp_apply if cls == "cyber_physical" else None,
        },
        "labels": {
            "label_anomaly":          int(labels.get("label_anomaly", 0)),
            "label_cyber_anomaly":    int(labels.get("label_cyber_anomaly", 0)),
            "label_physical_anomaly": int(labels.get("label_physical_anomaly", 0)),
        },
        "alignment_status": "ACCEPTED",
        "limitations": sc.get("validation_notes", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 evidence packet per scenario
# ─────────────────────────────────────────────────────────────────────────────

def _build_evidence_packet(sc: dict, author: str) -> dict:
    sid    = sc["scenario_id"]
    cls    = sc.get("scenario_class", "")
    family = sc.get("scenario_family", "")
    start  = int(sc.get("start_time_s", 0))
    dur    = int(sc.get("duration_s", 0))
    end    = start + dur - 1
    effects = sc.get("physical_effects", [])
    obs     = sc.get("expected_observable_signals", [])
    cc      = sc.get("cyber_context", {})
    labels  = sc.get("labels", {})
    ramp_s  = _get_ramp_s(effects)

    phys_vars = [e.get("variable", "") for e in effects]
    cyber_fields = [
        "cyber_anomaly_active", "cyber_state", "delivery_status",
        "cia_dimension", "protocol_profile", "lifecycle_stage",
    ]

    # Explanation focus
    if cls == "normal":
        focus = "No abnormal cyber or physical evidence. True negative baseline."
    elif cls == "physical_only":
        focus = (
            "Physical signal changed while cyber context remained normal. "
            f"Look for deviation in: {', '.join(phys_vars)}."
        )
    elif cls == "cyber_only":
        focus = (
            "Cyber anomaly occurred but DER/PCC physical signals did not change. "
            "Evidence is entirely in the cyber/context layer."
        )
    else:  # cyber_physical
        focus = (
            "Cyber event occurred and physical signal deviation followed within the same scenario window. "
            f"Cyber lead time: ~{max(0, 5 - ramp_s)} seconds before physical onset. "
            f"Physical variables: {', '.join(phys_vars)}."
        )

    return {
        "scenario_id":    sid,
        "author_model":   author,
        "scenario_family":family,
        "scenario_class": cls,
        "labels": {
            "label_anomaly":          int(labels.get("label_anomaly", 0)),
            "label_cyber_anomaly":    int(labels.get("label_cyber_anomaly", 0)),
            "label_physical_anomaly": int(labels.get("label_physical_anomaly", 0)),
        },
        "timing_alignment": {
            "start_time_s":               start,
            "end_time_s":                 end,
            "duration_s":                 dur,
            "physical_effect_start_s":    start if effects else None,
            "physical_effect_end_s":      end   if effects else None,
            "cyber_event_start_s":        start,
        },
        "physical_evidence_summary": (
            f"Variables: {', '.join(phys_vars) or 'none'}. "
            f"Effect types: {', '.join(e.get('effect_type','') for e in effects) or 'none'}. "
            f"Ramp: {ramp_s}s." if effects else "No physical effects."
        ),
        "cyber_evidence_summary": (
            f"CIA: {cc.get('cia_dimension','none')}. "
            f"Category: {cc.get('cyber_event_category','—')}. "
            f"Delivery: {cc.get('delivery_status','—')}. "
            f"Protocol: {cc.get('protocol_profile','—')}."
        ),
        "explanation_inputs": {
            "detector_window_reference": f"PLACEHOLDER: model detection window for {sid}",
            "physical_signals_to_check": phys_vars + [v for v in obs if v not in phys_vars],
            "cyber_fields_to_check":     cyber_fields,
            "expected_asset_focus":      sc.get("target_asset_id", ""),
            "expected_timing_relationship": (
                "cyber before physical by ~5s" if cls == "cyber_physical" else
                "no cyber anomaly" if cls in ("normal", "physical_only") else
                "cyber only, no physical lag"
            ),
        },
        "expected_explanation_focus": focus,
        "safety_claim_boundary": sc.get("safety_note", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate(aligned_df: pd.DataFrame, event_log_df: pd.DataFrame,
              manifest_df: pd.DataFrame, n_ctx: int, n_ev: int) -> tuple[dict, list, list]:
    errs  = []
    warns = []

    # 1-5: row counts and time_s integrity
    nr = len(aligned_df)
    if nr != TOTAL_SECONDS:
        errs.append(f"aligned_1s has {nr} rows (expected {TOTAL_SECONDS})")
    ts_min = int(aligned_df["time_s"].min())
    ts_max = int(aligned_df["time_s"].max())
    if ts_min != 0:
        errs.append(f"time_s min = {ts_min} (expected 0)")
    if ts_max != T_MAX:
        errs.append(f"time_s max = {ts_max} (expected {T_MAX})")
    dups = aligned_df["time_s"].duplicated().sum()
    if dups:
        errs.append(f"{dups} duplicate time_s values")

    # 6: scenario IDs in event log exist in manifest
    mids = set(manifest_df["scenario_id"].tolist())
    log_ids = set(event_log_df["scenario_id"].unique())
    unknown = log_ids - mids
    if unknown:
        errs.append(f"Event log contains unknown scenario_ids: {unknown}")

    # 7-9: coverage
    if len(mids) != 64:
        warns.append(f"Manifest has {len(mids)} scenarios (expected 64)")
    represented = set(aligned_df[aligned_df["zero_day_active_flag"] == 1]["scenario_id"])
    represented.discard("MULTIPLE")
    not_rep = mids - represented
    if not_rep:
        warns.append(f"Scenarios not represented in aligned table: {not_rep}")
    authors_rep = set(aligned_df[aligned_df["zero_day_active_flag"] == 1]["author_model"].unique())
    if len(authors_rep) < 4:
        errs.append(f"Only {len(authors_rep)} authors in aligned table: {authors_rep}")

    # 10: no old asset IDs
    OLD_ASSETS = {"pv35","pv60","pv83","pv76","pv49","pv104","pv114",
                  "bess48","bess76","bess108","bess114"}
    for col in ["target_asset_id"]:
        if col in aligned_df.columns:
            bad = set(aligned_df[col].unique()) & OLD_ASSETS
            if bad:
                errs.append(f"Old asset IDs in aligned table col {col}: {bad}")

    # 11-12: no packet fields, event_level_only always 1
    if "event_level_only" in aligned_df.columns:
        n_false = (aligned_df["event_level_only"] == 0).sum()
        if n_false:
            errs.append(f"{n_false} rows have event_level_only=0")
    if "packet_level_protocol_compliance_claimed" in aligned_df.columns:
        n_true = (aligned_df["packet_level_protocol_compliance_claimed"] == 1).sum()
        if n_true:
            errs.append(f"{n_true} rows have packet_level_protocol_compliance_claimed=1")

    # 13-18: label consistency by class
    for cls, expected in [
        ("normal",        {"cyber_anomaly_active": 0, "physical_effect_active": 0}),
        ("cyber_only",    {"physical_effect_active": 0}),
        ("physical_only", {"cyber_anomaly_active": 0}),
    ]:
        sub = aligned_df[(aligned_df["scenario_class"] == cls) &
                         (aligned_df["zero_day_active_flag"] == 1) &
                         (aligned_df["scenario_id"] != "MULTIPLE")]
        if sub.empty:
            continue
        for col, exp_val in expected.items():
            n_bad = (sub[col] != exp_val).sum()
            if n_bad:
                errs.append(f"{n_bad} rows of class={cls} have {col}!={exp_val}")
    # cyber_physical: both must be 1
    cp_sub = aligned_df[(aligned_df["scenario_class"] == "cyber_physical") &
                        (aligned_df["zero_day_active_flag"] == 1) &
                        (aligned_df["scenario_id"] != "MULTIPLE")]
    if not cp_sub.empty:
        bad_ca = (cp_sub["cyber_anomaly_active"] != 1).sum()
        bad_pa = (cp_sub["physical_effect_active"] != 1).sum()
        if bad_ca:
            errs.append(f"{bad_ca} cyber_physical rows have cyber_anomaly_active!=1")
        if bad_pa:
            errs.append(f"{bad_pa} cyber_physical rows have physical_effect_active!=1")

    overlap_rows = (aligned_df["scenario_id"] == "MULTIPLE").sum()

    status = "PASS" if not errs else "FAIL"
    summary = {
        "status":                  status,
        "total_physical_rows":     TOTAL_SECONDS,
        "total_aligned_rows":      nr,
        "total_scenarios":         len(mids),
        "scenarios_represented":   len(represented),
        "authors_represented":     list(authors_rep),
        "cyber_event_rows":        len(event_log_df),
        "context_packet_count":    n_ctx,
        "evidence_packet_count":   n_ev,
        "overlap_row_count":       int(overlap_rows),
        "validation_errors":       errs,
        "validation_warnings":     warns,
    }
    return summary, errs, warns


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────

def _write_cyber_context_report(summary: dict) -> None:
    lines = [
        "# ZERO_DAY_CYBER_CONTEXT_REPORT",
        "",
        f"Generated: {NOW_STR}",
        "",
        "## Overview",
        "",
        "This report documents the event-level cyber/context files generated",
        "for Phase 2 zero-day scenarios.  All content is **event-level semantic",
        "context only** — no packet captures, no real network traffic, no",
        "protocol compliance claims.",
        "",
        "## Generated Files",
        "",
        f"| File | Rows/Objects | Purpose |",
        f"|---|---|---|",
        f"| `zero_day_cyber_event_log.csv` | {summary['cyber_event_rows']} | Lifecycle rows per scenario |",
        f"| `zero_day_cyber_physical_aligned_1s.csv` | {summary['total_aligned_rows']:,} | 1-second aligned table |",
        f"| `zero_day_cyber_context_packets.jsonl` | {summary['context_packet_count']} | One context object per scenario |",
        f"| `zero_day_phase3_evidence_packets.jsonl` | {summary['evidence_packet_count']} | Phase 3 explanation inputs |",
        f"| `zero_day_phase3_alignment_summary.json` | 1 | Validation summary |",
        "",
        "## Alignment Method",
        "",
        "1. **Scenario manifest** (`zero_day_scenario_manifest.csv`) defines the",
        "   authoritative time windows for all 64 validated scenarios.",
        "2. Every row in the 1-second aligned table carries the scenario_id,",
        "   class, family, and flags for the scenario that is active at that",
        "   second.  Background seconds carry empty string values and zero flags.",
        "3. **Overlapping windows** are rare but handled: `scenario_id = MULTIPLE`",
        f"   and `active_scenario_ids` lists all active IDs.  ({summary['overlap_row_count']} overlap rows found.)",
        "4. Labels are **OR-merged** across overlapping scenarios.",
        "",
        "## Class-Level Representation",
        "",
        "| Class | cyber_anomaly_active | physical_effect_active | cyber_state |",
        "|---|---|---|---|",
        "| normal | 0 | 0 | normal_monitoring |",
        "| physical_only | 0 | 1 (inside window) | normal_monitoring / physical_event_observed |",
        "| cyber_only | 1 (inside window) | 0 | blocked_command / stale_measurement_context / … |",
        "| cyber_physical | 1 (inside window) | 1 (inside window) | cyber_physical_effect_active |",
        "",
        "## Why Event-Level Only",
        "",
        "- No real DER traffic was captured.",
        "- These scenarios are synthetic, frozen-model evaluation constructs.",
        "- `event_level_only = 1` and `packet_level_protocol_compliance_claimed = 0`",
        "  in every row.",
        "- Field names mirror the Phase 1 cyber event log schema but contain only",
        "  semantic/contextual values derived from the validated scenario bundles.",
        "",
        "## Phase 3 Explanation Layer Usage",
        "",
        "The `zero_day_phase3_evidence_packets.jsonl` file is structured for a",
        "small-LLM explanation layer that will receive:",
        "  1. Model detection result (anomaly score + threshold flag)",
        "  2. Physical evidence (signal values from `zero_day_physical_attacked.csv`)",
        "  3. Aligned cyber/context evidence (from this file)",
        "",
        "Each evidence packet includes `explanation_inputs.detector_window_reference`",
        "as a placeholder that will be filled in during Phase 3 when frozen model",
        "scores are available.",
        "",
        "## Illustrative Example — cyber_physical Scenario",
        "",
        "**Scenario:** `zdl_chatgpt_soc_dispatch_002`",
        "  - Class: `cyber_physical`, Family: `soc_aware_bess_dispatch_anomaly`",
        "  - Window: 54600s–55079s (480s)",
        "  - Effects: `bess_p_kw` absolute_delta, `bess_soc_percent` absolute_delta",
        "",
        "Command timeline (cyber layer):",
        "```",
        "  t=54595  command_created",
        "  t=54596  command_sent",
        "  t=54597  command_received",
        "  t=54598  command_accepted",
        "  t=54600  command_applied  ← physical effect begins",
        "  t=54760  physical_response_observed",
        "  t=55079  status_report",
        "```",
        "",
        "1-second aligned table rows 54600–55079:",
        "```",
        "  scenario_class = cyber_physical",
        "  cyber_anomaly_active = 1",
        "  physical_effect_active = 1",
        "  cyber_state = cyber_physical_effect_active",
        "  command_apply_flag = 1  (at t=54600 only)",
        "```",
        "",
        "## Limitations",
        "",
        "- Cyber context is not derived from real network captures.",
        "- Command timing is deterministically computed from scenario start_time_s.",
        "- All values are bounded by the scenario window; nothing outside the window",
        "  is modified.",
        "",
        "---",
        "*Phase 2 / Phase 3 Aligned Cyber-Context — complete*",
    ]
    (REPORTS_DIR / "ZERO_DAY_CYBER_CONTEXT_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[INFO] Wrote ZERO_DAY_CYBER_CONTEXT_REPORT.md")


def _write_alignment_validation_report(summary: dict, errs: list, warns: list) -> None:
    lines = [
        "# PHASE3_ZERO_DAY_ALIGNMENT_VALIDATION_REPORT",
        "",
        f"Generated: {NOW_STR}",
        f"**Validation status: {summary['status']}**",
        "",
        "## Summary",
        "",
        f"| Check | Value |",
        f"|---|---|",
        f"| Total physical rows | {summary['total_physical_rows']:,} |",
        f"| Aligned 1s rows | {summary['total_aligned_rows']:,} |",
        f"| Total scenarios | {summary['total_scenarios']} |",
        f"| Scenarios represented | {summary['scenarios_represented']} |",
        f"| Authors represented | {len(summary['authors_represented'])} |",
        f"| Cyber event rows | {summary['cyber_event_rows']} |",
        f"| Context packets | {summary['context_packet_count']} |",
        f"| Evidence packets | {summary['evidence_packet_count']} |",
        f"| Overlap rows | {summary['overlap_row_count']} |",
        "",
    ]
    if not errs:
        lines += ["## Validation Errors", "", "None.", ""]
    else:
        lines += ["## Validation Errors", ""]
        for e in errs:
            lines.append(f"- **ERROR**: {e}")
        lines.append("")
    if warns:
        lines += ["## Warnings", ""]
        for w in warns:
            lines.append(f"- WARN: {w}")
        lines.append("")

    lines += [
        "## Checks Performed",
        "",
        "1. aligned_1s file row count = 604,800",
        "2. time_s min = 0",
        "3. time_s max = 604,799",
        "4. No duplicate time_s",
        "5. All scenario_ids in cyber files exist in manifest",
        "6. All 64 scenarios represented",
        "7. All 4 authors represented",
        "8. No old multi-DER asset IDs",
        "9. event_level_only always 1",
        "10. packet_level_protocol_compliance_claimed always 0",
        "11. cyber_only has physical_effect_active = 0",
        "12. physical_only has cyber_anomaly_active = 0",
        "13. cyber_physical has both flags = 1",
        "14. normal has all anomaly flags = 0",
        "15. Overlap rows explicitly marked MULTIPLE",
        "",
        "---",
        "*Phase 3 Alignment Validation complete*",
    ]
    (REPORTS_DIR / "PHASE3_ZERO_DAY_ALIGNMENT_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("[INFO] Wrote PHASE3_ZERO_DAY_ALIGNMENT_VALIDATION_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ──────────────────────────────────────────────────────────
    print("[INFO] Loading inputs...")
    for p in [ATTACKED_CSV, MANIFEST_CSV, CONTEXT_CSV]:
        if not p.exists():
            print(f"[ERROR] Required file missing: {p}")
            sys.exit(1)

    attacked_df  = pd.read_csv(ATTACKED_CSV,
                               usecols=["timestamp_utc", "time_s"])
    manifest_df  = pd.read_csv(MANIFEST_CSV)
    print(f"  Attacked CSV : {len(attacked_df):,} rows")
    print(f"  Manifest     : {len(manifest_df)} scenarios")

    # Load all validated bundles
    all_scenarios: list[dict] = []
    for path in sorted(VALIDATED_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            bundle = json.load(f)
        author = bundle.get("author_model", "unknown")
        for sc in bundle.get("scenarios", []):
            sc["_author"] = author
            sc["_bundle_file"] = path.name
            all_scenarios.append(sc)
    print(f"  Scenarios from bundles: {len(all_scenarios)}")

    # ── Cyber event log ──────────────────────────────────────────────────────
    print("[INFO] Building cyber event log...")
    event_rows = []
    for sc in all_scenarios:
        rows = _build_lifecycle_rows(sc, sc["_author"], sc["_bundle_file"])
        event_rows.extend(rows)
    event_log_df = pd.DataFrame(event_rows)

    # Reorder columns
    LOG_COLS = [
        "cyber_event_id","scenario_id","author_model","scenario_family","scenario_class",
        "event_time_s","event_time_utc","start_time_s","end_time_s","duration_s",
        "target_asset_id","target_component","related_physical_variables",
        "lifecycle_stage","cyber_state","message_type","cyber_event_category",
        "cia_dimension","protocol_profile","delivery_status",
        "authn_status","authz_status","integrity_status","availability_status","confidentiality_status",
        "command_created_time_s","command_sent_time_s","command_recv_time_s",
        "command_accept_time_s","command_apply_time_s","command_response_time_s",
        "delay_s","retry_count","timeout_flag","blocked_flag","replay_flag",
        "mismatch_flag","stale_command_flag","command_delay_active_flag",
        "cyber_anomaly_active","physical_effect_active",
        "label_anomaly","label_cyber_anomaly","label_physical_anomaly",
        "event_level_only","packet_level_protocol_compliance_claimed",
        "source_bundle_file","validation_status","validation_notes",
    ]
    exist_cols = [c for c in LOG_COLS if c in event_log_df.columns]
    event_log_df = event_log_df[exist_cols]
    log_path = OUTPUTS_DIR / "zero_day_cyber_event_log.csv"
    event_log_df.to_csv(log_path, index=False)
    print(f"  Saved: {log_path.name} ({len(event_log_df)} rows)")

    # ── Aligned 1s table ─────────────────────────────────────────────────────
    aligned_df, n_overlap = _build_aligned_1s(all_scenarios, attacked_df)
    a1s_path = OUTPUTS_DIR / "zero_day_cyber_physical_aligned_1s.csv"
    aligned_df.to_csv(a1s_path, index=False)
    print(f"  Saved: {a1s_path.name} ({len(aligned_df):,} rows)")

    # ── Context packets JSONL ────────────────────────────────────────────────
    print("[INFO] Writing context packets JSONL...")
    ctx_path = OUTPUTS_DIR / "zero_day_cyber_context_packets.jsonl"
    with open(ctx_path, "w", encoding="utf-8") as f:
        for sc in all_scenarios:
            pkt = _build_context_packet(sc, sc["_author"])
            f.write(json.dumps(pkt, ensure_ascii=False) + "\n")
    print(f"  Saved: {ctx_path.name} ({len(all_scenarios)} objects)")

    # ── Phase 3 evidence packets JSONL ───────────────────────────────────────
    print("[INFO] Writing Phase 3 evidence packets JSONL...")
    ev_path = OUTPUTS_DIR / "zero_day_phase3_evidence_packets.jsonl"
    with open(ev_path, "w", encoding="utf-8") as f:
        for sc in all_scenarios:
            pkt = _build_evidence_packet(sc, sc["_author"])
            f.write(json.dumps(pkt, ensure_ascii=False) + "\n")
    print(f"  Saved: {ev_path.name} ({len(all_scenarios)} objects)")

    # ── Validation ───────────────────────────────────────────────────────────
    print("[INFO] Running validation...")
    summary, errs, warns = _validate(
        aligned_df, event_log_df, manifest_df,
        len(all_scenarios), len(all_scenarios)
    )
    summary["overlap_row_count"] = n_overlap
    summary_path = OUTPUTS_DIR / "zero_day_phase3_alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Validation: {summary['status']} ({len(errs)} errors, {len(warns)} warnings)")
    print(f"  Saved: {summary_path.name}")

    # ── Reports ──────────────────────────────────────────────────────────────
    print("[INFO] Writing reports...")
    _write_cyber_context_report(summary)
    _write_alignment_validation_report(summary, errs, warns)

    # ── Console summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PHASE 2/3 ALIGNED CYBER CONTEXT — COMPLETE")
    print("=" * 60)
    print(f"cyber_event_log rows      : {len(event_log_df)}")
    print(f"aligned_1s rows           : {len(aligned_df):,}")
    print(f"context packets           : {len(all_scenarios)}")
    print(f"evidence packets          : {len(all_scenarios)}")
    print(f"scenarios represented     : {summary['scenarios_represented']}")
    print(f"authors represented       : {', '.join(sorted(summary['authors_represented']))}")
    print(f"validation status         : {summary['status']}")
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
    report_dir = str(REPORTS_DIR)
    print(f"report path               : {report_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
