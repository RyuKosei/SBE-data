from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common import atomic_write_json, atomic_write_text, sha256_text, utc_now
from gpt_common import JUDGE_SYSTEM_PROMPT, judge_schema
from prompts import GENERATION_SYSTEM_PROMPT
from run_gpt_references import SYSTEM as REFERENCE_SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    schemas = {str(count): judge_schema(count) for count in (2, 3, 4)}
    atomic_write_json(args.output_root / "config/judge_schemas_by_checkpoint_count.json", schemas)
    protocol = {
        "frozen_at": utc_now(),
        "generation_system_prompt": GENERATION_SYSTEM_PROMPT,
        "reference_system_prompt": REFERENCE_SYSTEM_PROMPT,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "hashes": {
            "generation_system_prompt_sha256": sha256_text(GENERATION_SYSTEM_PROMPT),
            "reference_system_prompt_sha256": sha256_text(REFERENCE_SYSTEM_PROMPT),
            "judge_system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
            "judge_schema_2_sha256": sha256_text(json.dumps(schemas["2"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "judge_schema_3_sha256": sha256_text(json.dumps(schemas["3"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "judge_schema_4_sha256": sha256_text(json.dumps(schemas["4"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
        },
        "content_pass_rule": "coverage >= 0.90 and not factual_contradiction and not task_result_changed_by_addition and not unjudgeable",
        "zero_checkpoint_policy": "data error; never assign automatic coverage 0 or 1",
        "numeric_boolean_policy": "booleans are rejected for style_B_intensity and naturalness",
        "blind_fields_excluded_from_judge_prompt": ["target_ratio", "model_id", "model_family", "generation_seed", "replicate_id", "endpoint_identity", "original_generation_prompt"],
    }
    atomic_write_text(args.output_root / "config/prompt_and_scoring_protocol.yaml", yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False))
    print(json.dumps(protocol["hashes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
