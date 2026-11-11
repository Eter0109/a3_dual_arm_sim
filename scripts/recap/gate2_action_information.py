"""Gate 2: does the action add information beyond the state, now that counterfactuals exist?

Why this script exists
----------------------
Gates 1 verified that `branch_rollouts_new.py` produces true counterfactuals: 78 states, 702
branches, every branch of a state leaving from a bit-identical simulator state (Gate B p95 =
0.0e+00).  The question this script answers is the one that decides DSRL:

    with one state and several actions, does the action predict the outcome at all,
    and does its effect depend on the state?

The last clause is the part that matters and the part the earlier offline work got wrong.  On
the 400-episode set, `Q(s, a)` reached 97.4 % pairwise accuracy yet its gradient with respect to
the action was **identical to five decimal places at five different states** whose Q spanned 1.18
to 5.19 -- it had learned `Q(s, a) = f(s) + c . a`, a global action preference.  A critic like
that cannot steer: every state would be pushed in the same direction.

So there are two separate results here and both are needed:

* **Does the action help?**  A cross-validated ablation: predict the change in Cookie count from
  the state alone, from the action alone, and from both.  If "both" does not beat "state alone",
  the action carries no usable information and DSRL is closed.
* **Does the action's effect depend on the state?**  A state x action interaction fitted and
  compared against the additive model.  If the additive model is as good, Q is `f(s) + g(a)` --
  the same degeneracy as before, and equally useless for steering.

Why linear models and not a network
-----------------------------------
702 rows over 78 states.  A network of any useful size fits this by memorisation, and a
leave-one-state-out split then reports "it generalises" while measuring nothing.  Ridge
regression with the regularisation strength chosen inside the cross-validation loop is the
honest tool at this sample size, and its interaction term is directly interpretable: the
state-dependent part of the action gradient is the coefficient matrix itself.

The design decisions that keep this from lying
----------------------------------------------
* **Grouped by state.**  Every branch of a state shares that state, so a random row split leaks
  the state into the test fold and inflates every score.  `LeaveOneGroupOut` over states is the
  only split that answers "a state the model has not seen".
* **The regularisation strength is chosen inside the fold.**  Tuning it on the full set and then
  cross-validating with that value is the most common way a small-data result becomes a fiction.
* **A permutation null.**  The whole thing is re-run with the action column shuffled *within the
  whole set* (which destroys the state-action pairing but preserves both marginals) and with the
  target shuffled within a state.  The first says how much a model could learn from the action's
  marginal alone; the second is the null for "there is nothing to predict".
* **Report the variance the target has.**  A target with almost no within-state variance makes
  every score look good for free, so `within_state_variance / total_variance` is printed first.

What the numbers mean
---------------------
`delta` is `count_end - count_start` over the branch horizon (120 control steps at the default
`--horizon`).  At most branch points the control shows no movement at all (opportunity rate
0.389), so a large fraction of `delta` is exactly 0.  That is a property of the data, not a
defect: it says the horizon is short relative to the placement cycle, and it caps how much any
model can explain.  The script prints `P(delta != 0)` and the achievable ceiling alongside.

Usage
-----
    python scripts/recap/gate2_action_information.py

    # with the shard files named explicitly
    python scripts/recap/gate2_action_information.py --glob 'branches/run_1108_2036_s*.npz'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def load(patterns: list[str]) -> dict:
    """Stack every shard into one table, keeping the state id as the grouping key."""
    files: list[str] = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise SystemExit(f"no shard matched {patterns}")

    point_offset = 0
    states, noises, deltas, groups, sigmas = [], [], [], [], []
    for path in files:
        data = np.load(path)
        points = data["points"]
        point_states = data["point_states"]
        branch_point = data["branch_point"]
        keep = points[:, 5] > 0.5          # `valid` flag; invalid points were never branched

        # Group id must be unique across shards, so offset by the running count.
        local_to_global = {
            int(local): point_offset + int(local) for local in np.flatnonzero(keep)
        }
        for row in range(branch_point.shape[0]):
            local = int(branch_point[row])
            if local not in local_to_global:
                continue
            global_id = local_to_global[local]
            states.append(point_states[local])
            noises.append(data["branch_noise"][row])
            deltas.append(int(data["branch_count_end"][row])
                          - int(data["branch_count_start"][row]))
            groups.append(global_id)
            sigmas.append(float(data["branch_sigma"][row]))
        point_offset += int(points.shape[0])

    return {
        "state": np.asarray(states, dtype=np.float64),
        "action": np.asarray(noises, dtype=np.float64),
        "delta": np.asarray(deltas, dtype=np.float64),
        "group": np.asarray(groups, dtype=np.int64),
        "sigma": np.asarray(sigmas, dtype=np.float64),
        "files": files,
    }


# ----------------------------------------------------------------------------------------------
# ridge, with the penalty chosen inside the training fold only
# ----------------------------------------------------------------------------------------------
def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    y_mean = y.mean()
    # Solve (Z'Z + lam I) w = Z'(y - y_mean) without forming the normal equations twice.
    gram = Z.T @ Z + lam * np.eye(Z.shape[1])
    weights = np.linalg.solve(gram, Z.T @ (y - y_mean))
    return weights, y_mean, mean, scale


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    weights, y_mean, mean, scale = model
    return ((X - mean) / scale) @ weights + y_mean


def cv_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    lams: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> dict:
    """Leave-one-group-out CV, with `lam` selected on the training folds of each split.

    Selecting `lam` once on the whole set and then reporting the CV score of that `lam` is the
    standard way a small-data result turns out to be fiction; each fold has to pick its own.
    """
    predictions = np.empty_like(y)
    chosen = []
    unique_groups = np.unique(groups)
    for group in unique_groups:
        test = groups == group
        train = ~test
        if train.sum() < 8 or test.sum() == 0:
            predictions[test] = y[train].mean() if train.any() else y.mean()
            continue
        # Inner split: leave-one-out over the training groups to pick the penalty.
        inner_groups = np.unique(groups[train])
        best_lambda, best_error = lams[len(lams) // 2], np.inf
        for lam in lams:
            errors = []
            for inner in inner_groups:
                inner_test = train & (groups == inner)
                inner_train = train & ~inner_test
                if inner_train.sum() < 8:
                    continue
                model = ridge_fit(X[inner_train], y[inner_train], lam)
                errors.append(float(np.mean((ridge_predict(model, X[inner_test])
                                             - y[inner_test]) ** 2)))
            if errors and np.mean(errors) < best_error:
                best_lambda, best_error = lam, float(np.mean(errors))
        chosen.append(best_lambda)
        model = ridge_fit(X[train], y[train], best_lambda)
        predictions[test] = ridge_predict(model, X[test])

    residual = y - predictions
    total = y - y.mean()
    return {
        "r2": float(1.0 - np.sum(residual ** 2) / max(np.sum(total ** 2), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_lambda": float(np.mean(chosen)) if chosen else float("nan"),
        "predictions": predictions,
    }


def build_features(state: np.ndarray, action: np.ndarray, kind: str) -> np.ndarray:
    """Feature blocks for the ablation.  Interaction is a full outer product, ridge-regularised."""
    if kind == "intercept":
        return np.zeros((state.shape[0], 1))
    if kind == "state":
        return state
    if kind == "action":
        return action
    if kind == "state+action":
        return np.hstack([state, action])
    if kind == "state+action+interaction":
        # Every product of one state coordinate and one action coordinate.  256 columns for a
        # 16x16 pair, which is why the penalty has to be selected per fold.
        interaction = np.einsum("ni,nj->nij", state, action).reshape(state.shape[0], -1)
        return np.hstack([state, action, interaction])
    raise ValueError(kind)


def gradient_analysis(state: np.ndarray, action: np.ndarray, y: np.ndarray) -> dict:
    """Does the fitted action gradient point the same way at every state?

    This is the measurement that failed on the offline critic: `dQ/da` was constant to five
    decimal places across states whose Q spanned 1.18 to 5.19.  Here it is computed from the
    interaction model, whose `d(delta)/da = w_a + W_sa^T s` genuinely varies with `s` if the data
    supports it.

    Two numbers are reported:
      * the mean cosine between gradients at *nearby* states, which is where a steering actor
        would need the gradient to be locally consistent, and
      * the mean cosine over *all* pairs, which is the null: if both are the same, the gradient
        does not move with the state.
    """
    feature_dim = state.shape[1]
    interaction = np.einsum("ni,nj->nij", state, action).reshape(state.shape[0], -1)
    X = np.hstack([state, action, interaction])
    weights, _, _, _ = ridge_fit(X, y, 100.0)

    # Layout is [state(16) | action(16) | interaction(256)], so the interaction block starts at
    # 2 * feature_dim, not at feature_dim (which is the action block and is only 16 wide).
    W = weights[2 * feature_dim:].reshape(feature_dim, feature_dim)
    w_a = weights[feature_dim:2 * feature_dim]
    gradients = w_a[None, :] + state @ W

    norm = np.linalg.norm(gradients, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    unit = gradients / norm
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, np.nan)

    # Nearby states: the closest other state in joint space.
    distances = np.linalg.norm(state[:, None, :] - state[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.argmin(axis=1)
    near_cos = np.array([similarity[i, nearest[i]] for i in range(state.shape[0])])

    return {
        "gradient_norm_median": float(np.median(np.linalg.norm(gradients, axis=1))),
        "gradient_norm_cv": float(np.std(np.linalg.norm(gradients, axis=1))
                                  / max(np.mean(np.linalg.norm(gradients, axis=1)), 1e-12)),
        "near_cosine_mean": float(np.nanmean(near_cos)),
        "all_pairs_cosine_mean": float(np.nanmean(similarity)),
        # The constant-gradient hypothesis, measured rather than assumed: if `dQ/da` were a
        # constant vector, the across-state standard deviation of each gradient component would
        # be zero relative to the component's own magnitude.
        "gradient_component_cv": float(np.mean(np.std(gradients, axis=0)
                                               / np.maximum(np.abs(np.mean(gradients, axis=0)), 1e-9))),
    }


def within_state_ranking(state, action, y, groups, lams=(1.0, 10.0, 100.0, 1000.0)) -> dict:
    """The measurement DSRL actually needs: at one state, does the model rank the actions right?

    The leave-one-state-out regression above answers a different question -- "predict this new
    state's outcome from its joints" -- and cannot work here, because what distinguishes two
    states is *which Cookie sits in which slot*, a visual fact that the 16 joint angles do not
    carry.  Its negative R^2 is therefore expected and is not evidence about the action.

    What a sampler-steering actor needs is narrower and harder: with the state held fixed, among
    several candidate actions, can anything predict which one produces the larger change in the
    Cookie count?  That is a ranking question within a state, and it is measured here by

      1. centring the target within each state, so the state's own contribution is removed and
         only within-state variation is left to explain;
      2. fitting on all other states and predicting on the held-out state -- the action
         coefficients have to transfer, the state cannot be memorised;
      3. scoring pairwise accuracy over every pair of branches of the held-out state.

    A model that only knows "which state this is" scores 0.5 here by construction, so 0.5 is a
    real null rather than an absence of data.
    """
    # Centre within state: y_c = y - mean(y | state).
    y_centred = np.empty_like(y)
    for group in np.unique(groups):
        mask = groups == group
        y_centred[mask] = y[mask] - y[mask].mean()

    unique_groups = np.unique(groups)
    predictions = np.empty_like(y)
    for group in unique_groups:
        test = groups == group
        train = ~test
        if train.sum() < 8:
            predictions[test] = 0.0
            continue
        # Pick the penalty on a leave-one-group-out loop inside the training states, scoring
        # *ranking* rather than squared error, because that is the target quantity.
        best_lam, best_score = lams[len(lams) // 2], -np.inf
        for lam in lams:
            inner_predictions = np.empty(train.sum())
            inner_index = np.flatnonzero(train)
            for inner in np.unique(groups[train]):
                inner_test = groups[inner_index] == inner
                inner_train = ~inner_test
                if inner_train.sum() < 8:
                    inner_predictions[inner_test] = 0.0
                    continue
                model = ridge_fit(action[inner_index][inner_train],
                                  y_centred[inner_index][inner_train], lam)
                inner_predictions[inner_test] = ridge_predict(
                    model, action[inner_index][inner_test]
                )
            score = _pairwise_accuracy(y_centred[inner_index], inner_predictions,
                                      groups[inner_index])
            if score > best_score:
                best_lam, best_score = lam, score
        model = ridge_fit(action[train], y_centred[train], best_lam)
        predictions[test] = ridge_predict(model, action[test])

    return {
        "pairwise_accuracy_action_only": _pairwise_accuracy(y_centred, predictions, groups),
        "pairwise_accuracy_chance": 0.5,
        "centred_target_variance": float(np.var(y_centred)),
    }


def _pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of within-state branch pairs the prediction orders correctly."""
    correct, total = 0, 0
    for group in np.unique(groups):
        mask = np.flatnonzero(groups == group)
        if len(mask) < 2:
            continue
        truth = y_true[mask]
        pred = y_pred[mask]
        for i in range(len(mask)):
            for j in range(i + 1, len(mask)):
                if truth[i] == truth[j]:
                    continue
                total += 1
                if (pred[i] > pred[j]) == (truth[i] > truth[j]):
                    correct += 1
    return float(correct / total) if total else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--glob", nargs="+",
        default=["/root/autodl-tmp/datasets/a3_recap_all/branches/run_1108_2036_s*.npz"],
        help="shard npz files written by branch_rollouts_new.py",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("/root/autodl-tmp/datasets/a3_recap_all/gate2.json"))
    parser.add_argument("--permutations", type=int, default=20)
    args = parser.parse_args(argv)

    data = load(args.glob)
    state, action, y = data["state"], data["action"], data["delta"]
    groups = data["group"]
    print(f"[gate2] {len(data['files'])} shards, {len(y)} branches, "
          f"{len(np.unique(groups))} states", flush=True)

    # --- 1. what is there to explain? ----------------------------------------------------
    within = []
    for group in np.unique(groups):
        mask = groups == group
        within.append(float(np.var(y[mask])))
    within_var = float(np.mean(within))
    total_var = float(np.var(y))
    moved = float(np.mean(y != 0))
    print()
    print("=" * 78)
    print("1. THE TARGET -- how much is there to predict at all")
    print(f"  delta = count_end - count_start over the branch horizon")
    print(f"  mean {y.mean():+.3f}  sd {y.std():.3f}  range {y.min():+.0f} .. {y.max():+.0f}")
    print(f"  P(delta != 0)                    {moved:.3f}")
    print(f"  within-state variance            {within_var:.4f}")
    print(f"  total variance                   {total_var:.4f}")
    print(f"  fraction explainable by state    {within_var / max(total_var, 1e-12):.3f}")
    if within_var < 1e-9:
        print("  No within-state variance at all: every branch of every state scored the same.")
        print("  Nothing downstream can be measured; stop here.")
        return 1

    # --- 2. the ablation ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("2. THE ABLATION -- does the action add anything to the state?")
    print("   leave-one-STATE-out CV; the penalty is chosen inside each training fold")
    print()
    print(f"   {'features':30s} {'R^2':>8s} {'RMSE':>8s} {'MAE':>8s} {'lambda':>9s}")
    results = {}
    for kind in ("intercept", "state", "action", "state+action",
                 "state+action+interaction"):
        X = build_features(state, action, kind)
        score = cv_score(X, y, groups)
        results[kind] = {key: value for key, value in score.items() if key != "predictions"}
        results[kind]["features"] = int(X.shape[1])
        print(f"   {kind:30s} {score['r2']:8.3f} {score['rmse']:8.3f} {score['mae']:8.3f} "
              f"{score['mean_lambda']:9.1f}  ({X.shape[1]} cols)")

    gain_action = results["state+action"]["r2"] - results["state"]["r2"]
    gain_interaction = (results["state+action+interaction"]["r2"]
                        - results["state+action"]["r2"])
    print()
    print(f"   action above state          {gain_action:+.3f} R^2")
    print(f"   interaction above additive  {gain_interaction:+.3f} R^2")

    # --- 3. the null ----------------------------------------------------------------------
    # Two different shuffles, because they rule out different mistakes.
    rng = np.random.default_rng(0)
    action_null, target_null = [], []
    for _ in range(args.permutations):
        shuffled_action = action[rng.permutation(len(action))]
        shuffled_target = y.copy()
        for group in np.unique(groups):
            mask = np.flatnonzero(groups == group)
            shuffled_target[mask] = y[rng.permutation(mask)]
        action_null.append(
            cv_score(build_features(state, shuffled_action, "state+action"),
                     y, groups)["r2"]
            - cv_score(build_features(state, shuffled_action, "state"), y, groups)["r2"]
        )
        target_null.append(
            cv_score(build_features(state, action, "state+action"),
                     shuffled_target, groups)["r2"]
        )

    print()
    print("=" * 78)
    print(f"3. THE NULL -- {args.permutations} permutations")
    print(f"   action shuffled (state-action pairing destroyed):")
    print(f"     action gain {np.mean(action_null):+.3f} +/- {np.std(action_null):.3f} R^2")
    print(f"     observed    {gain_action:+.3f}  ->  "
          f"{(gain_action - np.mean(action_null)) / max(np.std(action_null), 1e-9):+.1f} sd")
    print(f"   target shuffled within each state (nothing left to learn):")
    print(f"     best R^2    {np.mean(target_null):+.3f} +/- {np.std(target_null):.3f}")

    # --- 4. the ranking -- the question DSRL actually asks ----------------------------
    # The regression above is a cross-state extrapolation and is expected to fail: two states
    # differ in *which Cookie sits in which slot*, which the joint angles do not encode.  What
    # an actor needs is the within-state ordering, measured here.
    ranking = within_state_ranking(state, action, y, groups)
    print()
    print("=" * 78)
    print("4. THE RANKING -- at a fixed state, can the action order the outcomes?")
    print(f"   pairwise accuracy, action only        {ranking['pairwise_accuracy_action_only']:.3f}")
    print(f"   chance                                {ranking['pairwise_accuracy_chance']:.3f}")
    print(f"   within-state centred variance         {ranking['centred_target_variance']:.4f}")

    # --- 5. the gradient -------------------------------------------------------------
    gradient = gradient_analysis(state, action, y)
    print()
    print("=" * 78)
    print("5. THE GRADIENT -- does d(delta)/d(action) point the same way everywhere?")
    print(f"   |gradient| median                {gradient['gradient_norm_median']:.4f}")
    print(f"   |gradient| across-state CV       {gradient['gradient_norm_cv']:.3f}")
    print(f"   cosine, nearest state            {gradient['near_cosine_mean']:+.3f}")
    print(f"   cosine, all pairs                {gradient['all_pairs_cosine_mean']:+.3f}")
    print(f"   per-component CV                 {gradient['gradient_component_cv']:.3f}")
    print()
    print("   For reference, the offline critic measured a per-component CV of 0.000 --")
    print("   identical gradients everywhere, i.e. Q(s,a) = f(s) + c.a.")

    # --- 5. the verdict ---------------------------------------------------------------
    print()
    print("=" * 78)
    print("VERDICT")
    z = (gain_action - np.mean(action_null)) / max(np.std(action_null), 1e-9)
    interaction_z = gain_interaction
    pairwise = ranking["pairwise_accuracy_action_only"]
    if not np.isfinite(pairwise) or pairwise < 0.55:
        print(f"  At a fixed state the action does not order the outcomes "
              f"(pairwise {pairwise:.3f} against 0.50 chance).")
        print("  DSRL cannot be steered by a critic trained on this signal. Close the direction.")
        verdict = "action_does_not_rank"
    elif interaction_z <= 0.0:
        print(f"  The action adds information ({z:+.1f} sd) but the effect is ADDITIVE: the")
        print("  interaction term does not improve on state+action.  A critic of the form")
        print("  f(s) + g(a) can rank the actions at a state, so DSRL's actor would receive a")
        print("  gradient -- but it would push every state in the same direction, which is the")
        print("  degeneracy the offline critic had. Usable only if the actor is allowed to")
        print("  condition on the state; not usable as a global preference.")
        verdict = "additive_only"
    else:
        print(f"  The action adds information ({z:+.1f} sd) AND the effect depends on the")
        print(f"  state (interaction +{interaction_z:.3f} R^2).  This is the first data in this")
        print("  project that supports a state-conditioned action value, which is what DSRL")
        print("  needs. Next: train the critic on these 702 branches and re-run the constant-")
        print("  gradient test before spending anything on an actor.")
        verdict = "state_dependent"

    payload = {
        "branches": int(len(y)),
        "states": int(len(np.unique(groups))),
        "target": {
            "mean": float(y.mean()), "sd": float(y.std()), "moved_fraction": moved,
            "within_state_variance": within_var, "total_variance": total_var,
        },
        "ablation": results,
        "gain_action": float(gain_action),
        "gain_interaction": float(gain_interaction),
        "action_null_mean": float(np.mean(action_null)),
        "action_null_sd": float(np.std(action_null)),
        "target_null_mean": float(np.mean(target_null)),
        "action_gain_z": float(z),
        "ranking": ranking,
        "gradient": gradient,
        "verdict": verdict,
        "shards": data["files"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n[gate2] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
