# B03 — Decision-Aware KV-State Signaling for Cache-Aware LLM Dispatch

B03 studies the question that sits one level above B02:

> B02 answers whether a KV-state update is **semantically** worth keeping.
> B03 asks whether it is **worth transmitting for the dispatch decision**.

A state update can be fresh, valid, and already reduced by B02's semantic
gateway (merge / tombstone priority / replica cap / adaptive admission), yet
still change nothing about the dispatcher's routing decision — or flip a
decision with near-zero or negative net serving benefit.  B03's motivation
experiment quantifies this decision redundancy with **counterfactual
evaluation** on the real B02 hybrid platform (live vLLM + real kernel
signaling path), *without* suppressing any update and *without* modifying
dispatch execution.

## Repository provenance

The platform code in this repository was copied from
[`B02_LIS`](https://github.com/BianYanhui/B02_LIS) at commit `c917c25`
(branch `b02-sota-policy-matrix-20260715`), **code only** — no B02 run
artifacts (results / traces / analysis / server logs) were copied.  B02
originals are never modified; B03 work lives in `b03_motivation/`.

| Path | Origin | Role |
|---|---|---|
| `shared_link_exp/run_live_shared_link_v3.py` | B02 | hybrid live harness: 4x vLLM (T4) + real TCP/tc signaling path |
| `shared_link_exp/net/` | B02 | gateway relay container, docker network, tc HTB setup, cluster restart |
| `shared_link_exp/sim/` | B02 | pure-simulation reference |
| `experiments/4t4/` | B02 | 4xT4 formal experiment code (frozen-4T4 platform) |
| `experiments/scripts/` | B02 | original simulation runner stack |
| `b03_motivation/` | **B03 (new)** | counterfactual instrumentation + analysis |

## B03 motivation experiment

See `b03_motivation/EXPERIMENT_DESIGN.md` for the design and
`b03_motivation/MOTIVATION_REPORT.md` for results.  The four research
questions:

- **RQ1** — after B02 semantic reduction, how many forwarded updates are
  decision-irrelevant (`decision_flip_rate`)?
- **RQ2** — does a decision flip imply meaningful serving benefit
  (heterogeneity of oracle value among flips)?
- **RQ3** — is update value highly skewed (Pareto concentration)?
- **RQ4** — can update value be predicted from pre-transmission observable
  features?

Core rule: the real experiment always runs the unmodified B02 policy;
the counterfactual evaluator only *observes and records*, evaluating each
forwarded update on a cloned state (World 0 without the update vs. World 1
with it) — the live dispatcher index is never mutated by instrumentation.
