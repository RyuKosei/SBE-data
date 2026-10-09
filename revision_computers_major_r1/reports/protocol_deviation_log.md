# Protocol deviation log

## External generator replacement — 2026-10-09

The revision brief initially proposed a DeepSeek external generator. The
author-supplied internal routes `deepseek-r1` and `deepseek-v4-pro-0813` both
failed before inference because the relay quota configuration lacked a
`null`-to-CNY exchange rate. They produced no formal outputs and consumed no
tokens. A subsequently tested `gpt-5.4-mini` route failed at the same
pre-inference layer because a `null`-to-USD exchange rate was missing.

The author then explicitly authorized replacing the unavailable DeepSeek
condition with a lightweight GPT model. Before any formal sixth-model output
was generated, `gpt-6-luna` was locked using capability, real-prompt smoke, and
cost evidence. The 20-output smoke batch was kept separate from formal data:
20/20 valid, zero transport/content failures, and USD 0.0015422 estimated cost.
The projected 6,720-output cost was USD 0.5182 (USD 0.6218 with 20% reserve),
and the formal generation hard cap was fixed at USD 1.

The formal run then attempted all 6,720 prespecified request identities. It
produced 6,719 usable completed texts, one incomplete response marked by the
relay as `content_filter`, and no transport failures. The incomplete response
was retained as an explicit failure and was not regenerated for quality. The
response-derived formal cost estimate was USD 0.4769491.

This replacement was not selected after observing geometric rankings. The
sixth generator is non-Qwen and uses a different model alias from the
`gpt-6-sol` judge, but both are GPT-family models served by the same internal
provider. Consequently, the GPT judge is not described as family-independent
from the sixth generator. The local Qwen judge supplies the cross-family
evaluator comparison. Neither GPT alias is treated as a frozen snapshot.

Machine-readable evidence is in
`manifests/external_generator_selection.json`; blocked-route evidence remains
in `manifests/deepseek_blocker.json` and is not deleted.
