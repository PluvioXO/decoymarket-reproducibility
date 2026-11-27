"""Checkpoint and evaluate the two exploratory effort experiments."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json

from decoymarket.analysis import paired_delta_ci
from scripts.auto_effort import assess, auto_choice, fit
from scripts.collect_live import write_json


def collect(directory, manifest, cases, run, workers):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Checkpoint settings changed in {directory}")
    else:
        if any(directory.iterdir()):
            raise ValueError(f"Checkpoints in {directory} have no manifest")
        write_json(manifest_path, manifest)

    paths = {key: directory / f"{key}.json" for key in cases}
    missing = [key for key, path in paths.items() if not path.exists()]
    done = len(paths) - len(missing)
    print(f"{directory}: {done}/{len(paths)} completed", flush=True)

    def save(key):
        write_json(paths[key], run(cases[key]))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(save, key) for key in missing]
        for count, future in enumerate(as_completed(futures), done + 1):
            future.result()
            if count == 1 or count % 50 == 0 or count == len(paths):
                print(f"{directory}: {count}/{len(paths)} completed", flush=True)
    return [json.loads(path.read_text()) for path in paths.values()]


def fixed_action(_observation, action):
    return action


def calibrated_policies(calibration, actions, token_price):
    estimates = fit(calibration, actions)
    fixed = {action: partial(fixed_action, action=action) for action in actions}
    best = min(actions, key=lambda action: assess(
        calibration, fixed[action], 1.0, token_price, actions,
    )[0]["loss_plus_inference"])

    def budget_only(observation):
        return auto_choice(estimates["all"], observation.budget, token_price, actions)

    def auto(observation):
        state = estimates.get(observation.state, estimates["all"])
        return auto_choice(state, observation.budget, token_price, actions)

    policies = {**fixed, "best_fixed": fixed[best], "budget_only": budget_only, "auto": auto}
    return policies, {"best_fixed_action": best, "calibration_estimates": estimates}


def evaluate_policies(tests, policies, actions, token_price, contrasts):
    scores, costs = {}, {}
    for name, policy in policies.items():
        scores[name], costs[name] = assess(tests, policy, 1.0, token_price, actions)
    return {
        "test_tasks": len(tests),
        "policies": scores,
        "net_savings_with_paired_95_ci": {
            f"{policy}_vs_{baseline}": paired_delta_ci(costs[baseline], costs[policy])
            for policy, baseline in contrasts
        },
    }
