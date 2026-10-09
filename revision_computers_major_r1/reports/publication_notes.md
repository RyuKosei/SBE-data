# Publication and release notes

## What can be stated from the recorded assets

- The benchmark prompts and generated texts in this revision are Chinese. No cross-language or language-independent validation was performed.
- Local outputs use the five frozen Qwen entries and deployment settings recorded in the legacy manifests and `config/experiment.yaml`.
- External reference generation and independent scoring use the internal Alibaba Relay Responses API with requested and returned model alias `gpt-6-sol`. The alias is not represented as a frozen public checkpoint. Formal records store prompt/schema/output hashes, requested parameters, token usage, timestamps, and (for requests after request-ID capture was added) a one-way hash of the response ID.
- The final sixth generator is the author-authorized lightweight `gpt-6-luna` internal alias. It is non-Qwen and distinct from the `gpt-6-sol` judge, but both are GPT-family models served through the same internal provider; this is a limitation, not evaluator-family independence. Earlier DeepSeek and GPT-5.4-Mini candidates were blocked before inference by relay quota/exchange-rate configuration and produced no formal outputs.
- GPT-generated endpoints are labeled `auto_generated`; Qwen quality screening is labeled `auto_screened`. No reference text or automatic score is labeled human-approved or human-verified.
- Fixed endpoint-reference generation and GPT judging use the same `gpt-6-sol` source model. The sixth generator uses the distinct `gpt-6-luna` alias but remains in the same GPT family/provider. Both overlaps must be disclosed and must not be presented as independent model-family evidence; the local Qwen judge is the cross-family evaluator.

## Public-release status

The available assets do not contain the applicable Alibaba internal relay terms, employer data-release approval, or a provider determination that raw generated responses may be redistributed publicly. Therefore this worktree does **not** assert that raw API outputs are cleared for public release. Before uploading data or a repository snapshot, the author/team should verify:

1. whether prompts and raw GPT/DeepSeek response text may be redistributed;
2. whether internal route names, employee metadata, or request identifiers require removal;
3. whether model-specific attribution or acceptable-use language is required;
4. whether the Chinese scenario/output corpus needs an additional content or privacy review.

Aggregate metrics, code, hashes, and fully de-identified manifests can be considered separately, but their release also remains subject to the relevant institutional policy.

On 2026-10-09, the author explicitly instructed the execution agent to push
this completed delivery to the existing public project repository. This
records the author's publication decision; it is not an independent legal or
institutional determination about provider or employer terms.

## Credential and private-metadata handling

- Formal scripts read the relay credential only from `SBE_IAI_API_KEY`; it is absent from CLI arguments, configuration, outputs, logs, and manifests.
- `empId` is used only as an HTTP header and is not written into sample-level output. The public configuration names `SBE_IAI_EMP_ID`; the employee identifier itself is not stored in the repository.
- Raw reasoning content is not persisted. DeepSeek records only whether reasoning was present; GPT formal runs request no reasoning.
- Response/request identifiers are stored only as SHA-256 hashes where capture is available.
- `manifests/retry_policy_audit.json` records the point at which SDK automatic
  retries were explicitly disabled. Exact underlying HTTP-attempt counts for
  the earlier successful GPT calls cannot be reconstructed and must not be
  claimed; no score-dependent retry was performed. It also records two
  deliberate judge-process concurrency restarts. Returned-response usage is
  exact, while a conservative in-flight-call allowance is added to the budget
  report because the relay supplies no invoice-level reconciliation. Raw
  usage for the two inexpensive comparison-model judge pilots was not retained;
  the budget manifest leaves its exact cost null and adds a separate USD 0.10
  conservative allowance rather than inventing token counts.
- The rolling exact-credential scan reports its result in `manifests/api_usage_and_budget.json`; the current recorded scan has zero matches.

## Required model/API disclosure in a paper or data card

Report the provider route, requested model alias, returned model field, whether a snapshot was frozen, call dates, API surface, sampling fields, reasoning setting, output budget, wrapper differences, retry rule, and the fact that account balance was not exposed. Explicitly disclose the GPT-6-Luna generator / GPT-6-Sol judge family overlap. API latency and token usage are service observations and must not be combined with the matched local-GPU efficiency comparison.
