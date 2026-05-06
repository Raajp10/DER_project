"""
Optional: Generate simple IEEE 2030.5-style XML research artifacts.
These are research-use XML-like files, NOT official IEEE 2030.5 protocol compliance.
NOT EXI-encoded. NOT packet captures.
Writes (if feasible): data_updated/raw/ieee2030_5_xml_research_artifacts/*.xml
"""
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import IEEE2030_5_XML_DIR, CYBER_ANOMALOUS_CSV

DISCLAIMER = """\
<!--
  RESEARCH USE ONLY
  This XML artifact models IEEE 2030.5-style DER control messages at a semantic level.
  It is NOT a serialized IEEE 2030.5 protocol message.
  It is NOT EXI-encoded.
  It does NOT represent official IEEE 2030.5 compliance.
  It is generated for research and educational purposes only.
-->"""

DER_CONTROL_TEMPLATE = """\
{disclaimer}
<DERControl xmlns="urn:ieee:std:2030.5:ns">
  <mRID>{mrid}</mRID>
  <DERControlBase>
    <opModTargetW><multiplier>0</multiplier><value>{target_p_w}</value></opModTargetW>
    <opModTargetVar><multiplier>0</multiplier><value>{target_q_var}</value></opModTargetVar>
  </DERControlBase>
  <creationTime>{created_ts}</creationTime>
  <interval>
    <duration>{duration}</duration>
    <start>{start_ts}</start>
  </interval>
  <randomizeDuration>0</randomizeDuration>
  <randomizeStart>0</randomizeStart>
  <!-- Research metadata -->
  <scenarioId>{scenario_id}</scenarioId>
  <lifecycleStage>{lifecycle_stage}</lifecycleStage>
  <protocolClaimLevel>semantic_ieee2030_5_style</protocolClaimLevel>
</DERControl>
"""

METER_READING_TEMPLATE = """\
{disclaimer}
<MirrorMeterReading xmlns="urn:ieee:std:2030.5:ns">
  <mRID>{mrid}</mRID>
  <lastUpdateTime>{ts}</lastUpdateTime>
  <MirrorReadingSet>
    <MirrorReading>
      <value>{p_w}</value>
      <localID>ActivePower</localID>
    </MirrorReading>
    <MirrorReading>
      <value>{q_var}</value>
      <localID>ReactivePower</localID>
    </MirrorReading>
  </MirrorReadingSet>
  <!-- Research metadata -->
  <scenarioId>{scenario_id}</scenarioId>
  <protocolClaimLevel>semantic_ieee2030_5_style</protocolClaimLevel>
</MirrorMeterReading>
"""


def main() -> bool:
    IEEE2030_5_XML_DIR.mkdir(parents=True, exist_ok=True)

    if not CYBER_ANOMALOUS_CSV.exists():
        print("Anomalous cyber log not found; skipping XML artifact generation.")
        return False

    df = pd.read_csv(CYBER_ANOMALOUS_CSV)

    # Only generate XML for a sample of control events (max 50)
    ctrl_mask = df["is_control_event"] == 1
    sample = df[ctrl_mask].head(50)

    count = 0
    for _, row in sample.iterrows():
        mrid = str(row.get("message_mrid", "unknown"))[:8]
        fname = f"DERControl_{mrid}.xml"
        target_p_w = int(float(row.get("target_p_kw", 0)) * 1000)
        target_q_var = int(float(row.get("target_q_kvar", 0)) * 1000)
        content = DER_CONTROL_TEMPLATE.format(
            disclaimer=DISCLAIMER,
            mrid=row.get("message_mrid", ""),
            target_p_w=target_p_w,
            target_q_var=target_q_var,
            created_ts=row.get("command_created_time_utc", ""),
            duration=int(float(row.get("delay_s", 60)) + 60),
            start_ts=row.get("command_sent_time_utc", ""),
            scenario_id=row.get("scenario_id", ""),
            lifecycle_stage=row.get("lifecycle_stage", ""),
        )
        (IEEE2030_5_XML_DIR / fname).write_text(content)
        count += 1

    # Generate a few meter reading XMLs
    meter_mask = df["is_monitoring_event"] == 1
    meter_sample = df[meter_mask].head(10)
    for _, row in meter_sample.iterrows():
        mrid = str(row.get("message_mrid", "unknown"))[:8]
        fname = f"MirrorMeterReading_{mrid}.xml"
        content = METER_READING_TEMPLATE.format(
            disclaimer=DISCLAIMER,
            mrid=row.get("message_mrid", ""),
            ts=row.get("event_time_utc", ""),
            p_w=int(float(row.get("target_p_kw", 0)) * 1000),
            q_var=int(float(row.get("target_q_kvar", 0)) * 1000),
            scenario_id=row.get("scenario_id", ""),
        )
        (IEEE2030_5_XML_DIR / fname).write_text(content)
        count += 1

    # Write README for the XML artifacts
    readme = f"""\
# IEEE 2030.5-Style XML Research Artifacts

**IMPORTANT DISCLAIMER:**
These XML files are RESEARCH-USE ONLY artifacts.
They are NOT:
- Official IEEE 2030.5 protocol messages
- EXI-encoded
- Network packet captures
- Officially validated IEEE 2030.5 implementations

They model the IEEE 2030.5 DER control lifecycle at the semantic/syntactic level
for research and educational purposes only.

## Contents
- DERControl_*.xml: DER active/reactive power control messages ({count} total)
- MirrorMeterReading_*.xml: Metering data messages

## Protocol Claim Level
All files carry: protocolClaimLevel = semantic_ieee2030_5_style
"""
    (IEEE2030_5_XML_DIR / "README.md").write_text(readme)

    print(f"Generated {count} XML research artifacts in {IEEE2030_5_XML_DIR}")
    print("  NOTE: These are semantic research artifacts, NOT official IEEE 2030.5 compliance.")
    return True


if __name__ == "__main__":
    main()
