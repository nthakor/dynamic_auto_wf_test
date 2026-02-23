"""
Dummy dataset generator for the Rule Waterfall App.
Generates a realistic dataset mimicking credit/application decisioning.
Supports up to 200 rules and millions of records.

Includes categorical columns so the app can demonstrate categorical filtering.
"""

import numpy as np
import pandas as pd
import os


def generate_dummy_dataset(
    n_records: int = 100_000,
    n_rules: int = 30,
    seed: int = 42,
    output_path: str = "data/applications.parquet",
) -> pd.DataFrame:
    """
    Generate a synthetic dataset of loan/application records with binary rule flags.

    Columns:
      - app_id        : unique application identifier
      - application_date : application date (for year/quarter filtering)
      - is_default    : 1 if the account is 'bad' (defaulted, etc.), 0 if good
      - product_type  : categorical — loan product category
      - channel       : categorical — origination channel
      - state         : categorical — US state (10 values)
      - risk_tier     : categorical — internal risk classification
      - rule_XXX      : binary 0/1 — 1 if rule XXX declined the application
    """
    np.random.seed(seed)

    print(f"Generating {n_records:,} records with {n_rules} rules...")

    # ── Date column ──────────────────────────────────────────────────────────
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-31")
    days = (end - start).days
    dates = start + pd.to_timedelta(
        np.random.randint(0, days, size=n_records), unit="D"
    )

    # ── Bad / target flag (overall bad rate ~15%) ────────────────────────────
    is_default = np.random.binomial(1, 0.15, size=n_records).astype(np.int8)

    # ── Categorical columns ──────────────────────────────────────────────────
    product_types = ["Personal Loan", "Credit Card", "Auto Loan", "Mortgage", "HELOC"]
    channels = ["Online", "Branch", "Broker", "Phone", "Partner"]
    states = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "PA", "AZ"]
    risk_tiers = ["Prime", "Near-Prime", "Sub-Prime", "Deep-Sub-Prime"]

    product_arr = np.random.choice(product_types, size=n_records)
    channel_arr = np.random.choice(channels, size=n_records)
    state_arr = np.random.choice(states, size=n_records)
    risk_arr = np.random.choice(risk_tiers, size=n_records, p=[0.40, 0.30, 0.22, 0.08])

    # ── Rule flags ───────────────────────────────────────────────────────────
    base_rates = np.random.uniform(0.02, 0.25, size=n_rules)
    lift = np.random.uniform(1.5, 4.0, size=n_rules)

    rule_cols = {}
    for i in range(n_rules):
        p_good = base_rates[i]
        p_bad = min(base_rates[i] * lift[i], 0.98)
        prob = np.where(is_default == 1, p_bad, p_good)
        rule_cols[f"rule_{i + 1:03d}"] = np.random.binomial(1, prob).astype(np.int8)

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "app_id": np.arange(1, n_records + 1, dtype=np.int32),
            "application_date": dates,
            "is_default": is_default,
            "product_type": product_arr,
            "channel": channel_arr,
            "state": state_arr,
            "risk_tier": risk_arr,
            **rule_cols,
        }
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Dataset saved to '{output_path}'  ({size_mb:.1f} MB in memory)")
    return df


def generate_aggregated_dataset(
    n_records: int = 100_000,
    n_rules: int = 30,
    seed: int = 42,
    output_path: str = "data/applications_agg.parquet",
) -> pd.DataFrame:
    """
    Generate a pre-aggregated dataset for 'aggregated' data mode.

    Process:
      1. Generate raw row-level data (reuses generate_dummy_dataset logic)
      2. Group by all rule-flag combinations + categorical columns + quarter
      3. Aggregate: count_apps = row count, bad_count = sum(is_default)
      4. Drop the binary bad flag (so auto-detection picks aggregated mode)

    Output columns:
      - application_date : quarter start date (for date filters)
      - product_type, channel, state, risk_tier : categorical filters
      - rule_XXX  : binary rule flags (shared by all rows in the group)
      - count_apps: total applications in this aggregated segment
      - bad_count : bad applications in this aggregated segment
    """
    np.random.seed(seed)
    print(
        f"Generating aggregated dataset ({n_records:,} base records, {n_rules} rules)..."
    )

    # Build raw row-level data in memory (don't save it)
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-31")
    days = (end - start).days
    dates = start + pd.to_timedelta(
        np.random.randint(0, days, size=n_records), unit="D"
    )

    is_default = np.random.binomial(1, 0.15, size=n_records).astype(np.int8)

    product_types = ["Personal Loan", "Credit Card", "Auto Loan", "Mortgage", "HELOC"]
    channels = ["Online", "Branch", "Broker", "Phone", "Partner"]
    states = ["CA", "TX", "FL", "NY", "IL", "OH", "GA", "NC", "PA", "AZ"]
    risk_tiers = ["Prime", "Near-Prime", "Sub-Prime", "Deep-Sub-Prime"]

    product_arr = np.random.choice(product_types, size=n_records)
    channel_arr = np.random.choice(channels, size=n_records)
    state_arr = np.random.choice(states, size=n_records)
    risk_arr = np.random.choice(risk_tiers, size=n_records, p=[0.40, 0.30, 0.22, 0.08])

    base_rates = np.random.uniform(0.02, 0.25, size=n_rules)
    lift = np.random.uniform(1.5, 4.0, size=n_rules)

    rule_dict = {}
    for i in range(n_rules):
        p_good = base_rates[i]
        p_bad = min(base_rates[i] * lift[i], 0.98)
        prob = np.where(is_default == 1, p_bad, p_good)
        rule_dict[f"rule_{i + 1:03d}"] = np.random.binomial(1, prob).astype(np.int8)

    df_raw = pd.DataFrame(
        {
            "application_date": dates,
            "is_default": is_default,
            "product_type": product_arr,
            "channel": channel_arr,
            "state": state_arr,
            "risk_tier": risk_arr,
            **rule_dict,
        }
    )

    rule_cols = sorted(rule_dict.keys())
    cat_cols = ["product_type", "channel", "state", "risk_tier"]
    group_keys = rule_cols + cat_cols

    # Collapse to quarterly periods to keep date filtering meaningful
    df_raw["_qtr"] = df_raw["application_date"].dt.to_period("Q").dt.to_timestamp()

    agg_df = (
        df_raw.groupby(group_keys + ["_qtr"], observed=True, sort=False)
        .agg(count_apps=("is_default", "count"), bad_count=("is_default", "sum"))
        .reset_index()
        .rename(columns={"_qtr": "application_date"})
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    agg_df.to_parquet(output_path, index=False, engine="pyarrow")
    size_mb = agg_df.memory_usage(deep=True).sum() / 1e6
    total = agg_df["count_apps"].sum()
    bad_rt = agg_df["bad_count"].sum() / total
    print(
        f"Aggregated dataset saved to '{output_path}'  "
        f"({len(agg_df):,} rows, {size_mb:.1f} MB)\n"
        f"Total apps: {total:,}  |  Bad rate: {bad_rt:.2%}"
    )
    return agg_df


if __name__ == "__main__":
    # Row-level dataset
    df = generate_dummy_dataset(n_records=100_000, n_rules=30)
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    print(f"Bad rate: {df['is_default'].mean():.2%}")
    print(f"Rule hit rates (sample):\n{df[rule_cols[:5]].mean()}")

    # Aggregated dataset
    print("\n" + "=" * 60)
    generate_aggregated_dataset(n_records=100_000, n_rules=30)
