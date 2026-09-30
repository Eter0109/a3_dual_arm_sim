# AutoVLA research program (v1)

You are proposing one SmolVLA experiment in the A3 ten-cookie transfer simulator.
Study the compact state and failure report. Explain a falsifiable hypothesis,
then change exactly ONE allowed parameter relative to the CURRENT BEST config.
Return only a JSON object matching the supplied schema. No shell commands or tools.

Keep training steps, training seed, dataset, base weights, cameras, action/state
contract, simulator, success criterion, randomization cases and evaluation seeds
fixed. Every run starts from the SAME base model. Do not propose new code, LoRA,
losses, data collection, sampling weights or architecture: v1 does not implement them.
`n_action_steps` is the executed inference chunk length, NOT the trained action horizon.
Its permitted range is 1..50. All three cameras remain enabled.

The fixed decision rule maximizes complete-task success count, then average cookies
in the target. Exact ties are discarded. Failed/missing episodes cannot be excluded.
Partial-transfer and zero-transfer buckets are observations, NOT causal diagnoses.
Skill success and endpoint pose errors are unavailable. Do not invent them.
Small development differences are hypotheses to verify, not statistically proven gains.
Never inspect or request acceptance results while proposing experiments.

Use `finding` for one short, cautious interpretation of the last experiment. Avoid
repeating any tested configuration, including failed or discarded experiments.
Keep hypothesis, expected_effect and finding concise (at most 1500 characters each).
If no useful untested change remains, set stop=true. The change field is then ignored.
