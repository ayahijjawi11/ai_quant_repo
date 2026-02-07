import pandas as pd

from quantum_optimizer import (
    solve_quantum_inspired,
    classical_greedy,
    compute_score_row,
)

AI_FILE = "../data/ai_output_for_quantum.csv"
LEVELS_ALLOWED = [1.0, 0.5, 0.0]


def enforce_fair_partial_policy(df: pd.DataFrame, supply_mw: float) -> pd.DataFrame:
    """
    Fair Partial policy:
    - decision_x not used
    """
    out = df.copy()

    if "base_score" not in out.columns:
        out["base_score"] = compute_score_row(out)

    out = out.sort_values(
        by=["priority_level", "base_score"],
        ascending=[False, False]
    ).copy()

    used = 0.0
    levels = []

    for _, row in out.iterrows():
        d = float(row["predicted_demand_mw"])
        remaining = supply_mw - used

        lvl = 0.0
        for L in LEVELS_ALLOWED:
            if L * d <= remaining + 1e-9:
                lvl = L
                break

        levels.append(lvl)
        used += lvl * d

    out["allocation_level"] = levels
    out["allocated_mw"] = out["allocation_level"] * out["predicted_demand_mw"].astype(float)
    out["unmet_mw"] = out["predicted_demand_mw"].astype(float) - out["allocated_mw"]

    out["score"] = out["base_score"]
    out["total_score"] = (out["score"] * out["allocation_level"]).sum()

    assert out["allocated_mw"].sum() <= supply_mw + 1e-6
    return out


def run_one_hour(
    hour: int = 12,
    supply_ratio: float = 0.35,
    n_sample: int = 12,
    seed: int = 42
):
    df = pd.read_csv(AI_FILE)

    df_hour = df[df["hour"] == hour].copy()
    if df_hour.empty:
        raise ValueError(f"No rows for hour={hour}")

    if len(df_hour) > n_sample:
        df_hour = df_hour.sample(n_sample, random_state=seed).copy()

    total_demand = df_hour["predicted_demand_mw"].astype(float).sum()
    supply_mw = supply_ratio * total_demand

    # ---------- Quantum-inspired selection ----------
    out_q_sel, best_score = solve_quantum_inspired(
        df_hour,
        supply_mw,
        enforce_fairness=True,
        enforce_pump_region=False
    )

    out_q_sel = out_q_sel.copy()
    out_q_sel["base_score"] = compute_score_row(out_q_sel)

    out_q_sorted = out_q_sel.sort_values(
        by=["decision_x", "priority_level", "base_score"],
        ascending=[False, False, False]
    ).copy()

    out_q = enforce_fair_partial_policy(out_q_sorted, supply_mw)

    # ---------- Classical baseline ----------
    out_c_sel = classical_greedy(df_hour, supply_mw)
    out_c = enforce_fair_partial_policy(out_c_sel, supply_mw)

    # ---------- Console summary ----------
    q_used = out_q["allocated_mw"].sum()
    c_used = out_c["allocated_mw"].sum()

    q_served = int((out_q["allocated_mw"] > 0).sum())
    c_served = int((out_c["allocated_mw"] > 0).sum())

    q_total_score = float(out_q["total_score"].iloc[0])
    c_total_score = float(out_c["total_score"].iloc[0])

    print(f"Quantum chosen bitstring: {bitstring}\n")
    print("=== Comparison: Quantum vs Classical (Fair Partial) ===")
    print(f"Supply limit: {supply_mw:.2f} MW")
    print(f"Quantum used: {q_used:.2f} MW | facilities served: {q_served} | total score: {q_total_score:.2f}")
    print(f"Classic used: {c_used:.2f} MW | facilities served: {c_served} | total score: {c_total_score:.2f}\n")

    cols = [
        "region", "zone", "facility_type", "predicted_demand_mw",
        "priority_level", "outage_risk",
        "allocation_level", "allocated_mw", "unmet_mw", "score"
    ]

    out_show = out_q.sort_values(
        by=["priority_level", "score"],
        ascending=[False, False]
    ).copy()

    print(out_show[cols])

    out_q.to_csv(f"quantum_allocation_hour_{hour}.csv", index=False)
    out_c.to_csv(f"classical_greedy_hour_{hour}.csv", index=False)

    return out_q, out_c


def run_pipeline(hour: int):
    """
    API-safe wrapper.
    This is what api.py imports and calls.
    """
    return run_one_hour(
        hour=hour,
        supply_ratio=0.35,
        n_sample=10,
        seed=42
    )


if __name__ == "__main__":
    run_one_hour(hour=12, supply_ratio=0.35, n_sample=10, seed=42)
