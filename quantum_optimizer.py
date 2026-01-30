import numpy as np
import pandas as pd

# weights 
FACILITY_WEIGHT = {
    "Emergency Center": 3.0,
    "Water Pump": 2.0,
    "Residential": 1.0,
}

CRITICAL_TYPES = {"Emergency Center", "Water Pump"}


def compute_score_row(df: pd.DataFrame) -> pd.Series:
    """
    base_score = priority * weight(type) * demand * risk_factor
    risk_factor: 
    """
    w = df["facility_type"].map(FACILITY_WEIGHT).fillna(1.0).astype(float)
    demand = df["predicted_demand_mw"].astype(float)
    risk = df["outage_risk"].astype(float)

    # risk_factor 
    risk_factor = 0.5 + 0.5 * (1.0 - risk)

    return df["priority_level"].astype(float) * w * demand * risk_factor


def _fairness_ok(df: pd.DataFrame, mask: int) -> bool:
    """
    fairness_ok:

    """
    has_gaza_possible = (
        (df["region"] == "Gaza")
        & (df["priority_level"] == 3)
        & (df["facility_type"].isin(CRITICAL_TYPES))
    ).any()

    has_wb_possible = (
        (df["region"] == "West Bank")
        & (df["priority_level"] == 3)
        & (df["facility_type"].isin(CRITICAL_TYPES))
    ).any()


    if not (has_gaza_possible and has_wb_possible):
        return True

    has_gaza_critical = False
    has_wb_critical = False

    n = len(df)
    for i in range(n):
        if (mask >> i) & 1:
            region = str(df.loc[i, "region"])
            ftype = str(df.loc[i, "facility_type"])
            pr = float(df.loc[i, "priority_level"])

            if pr == 3 and ftype in CRITICAL_TYPES:
                if region == "Gaza":
                    has_gaza_critical = True
                if region == "West Bank":
                    has_wb_critical = True

    return has_gaza_critical and has_wb_critical


def _pump_by_region_constraint_ok(df: pd.DataFrame, mask: int) -> bool:
    """
    pump_by_region_constraint:
  
    """
    has_gaza_pump = (
        (df["region"] == "Gaza")
        & (df["priority_level"] == 3)
        & (df["facility_type"] == "Water Pump")
    ).any()

    has_wb_pump = (
        (df["region"] == "West Bank")
        & (df["priority_level"] == 3)
        & (df["facility_type"] == "Water Pump")
    ).any()

    if not (has_gaza_pump and has_wb_pump):
        return True

    got_gaza = False
    got_wb = False
    n = len(df)
    for i in range(n):
        if (mask >> i) & 1:
            if float(df.loc[i, "priority_level"]) == 3 and str(df.loc[i, "facility_type"]) == "Water Pump":
                if str(df.loc[i, "region"]) == "Gaza":
                    got_gaza = True
                if str(df.loc[i, "region"]) == "West Bank":
                    got_wb = True

    return got_gaza and got_wb


def solve_quantum_inspired(df_hour: pd.DataFrame, supply_mw: float,
                          enforce_fairness: bool = True,
                          enforce_pump_region: bool = False):
    """
    Quantum-inspired exact brute force :
    - decision_x in {0,1}
    - maximize base_score sum
    - hard constraint: sum(demand*x) <= supply_mw
    - fairness_ok: 
    - optional pump_by_region_constraint
    """
    df = df_hour.copy().reset_index(drop=True)
    df["base_score"] = compute_score_row(df)

    demands = df["predicted_demand_mw"].astype(float).to_numpy()
    scores = df["base_score"].astype(float).to_numpy()
    n = len(df)

    best_score = -1e18
    best_mask = 0

    for mask in range(1 << n):
        total_d = 0.0
        total_s = 0.0
        feasible = True

        for i in range(n):
            if (mask >> i) & 1:
                total_d += demands[i]
                if total_d > supply_mw + 1e-9:
                    feasible = False
                    break
                total_s += scores[i]

        if not feasible:
            continue

        if enforce_fairness and (not _fairness_ok(df, mask)):
            continue

        if enforce_pump_region and (not _pump_by_region_constraint_ok(df, mask)):
            continue

        if total_s > best_score:
            best_score = total_s
            best_mask = mask

    x = np.array([(best_mask >> i) & 1 for i in range(n)], dtype=int)
    out = df.copy()
    out["decision_x"] = x
    return out, float(best_score)


def classical_greedy(df_hour: pd.DataFrame, supply_mw: float) -> pd.DataFrame:
    """
    Baseline: Classical greedy selection (0/1) by priority then base_score
    """
    df = df_hour.copy().reset_index(drop=True)
    df["base_score"] = compute_score_row(df)

    df = df.sort_values(by=["priority_level", "base_score"], ascending=[False, False]).copy()

    used = 0.0
    x = []
    for _, row in df.iterrows():
        d = float(row["predicted_demand_mw"])
        if used + d <= supply_mw + 1e-9:
            x.append(1)
            used += d
        else:
            x.append(0)

    df["decision_x"] = x
    return df
