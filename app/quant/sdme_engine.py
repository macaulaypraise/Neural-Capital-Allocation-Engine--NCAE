import pandas as pd
import numpy as np
from typing import Dict, List

def compute_sdme_allocations(market_data: List[Dict], max_assets: int = 4) -> Dict[str, float]:
    """
    The SDME Quant Engine.
    Processes cross-sectional market data to find Sentiment-Divergence.

    Expected input format:
    [
        {"ticker": "FET", "price_momentum": 65.4, "social_heat": 88.2, "volume_24h": 12000000},
        {"ticker": "TAO", "price_momentum": 45.1, "social_heat": 92.5, "volume_24h": 8500000},
        ...
    ]
    """
    df = pd.DataFrame(market_data)

    # 1. Hard Liquidity Guardrail
    # Drop any asset trading under $5M in 24h to prevent simulated slippage
    df = df[df['volume_24h'] >= 5_000_000].copy()

    # Fallback to stablecoin if the market is entirely illiquid
    if df.empty:
        return {"USDT": 1.0}

    # 2. Cross-Sectional Z-Scores
    # Measures how much an asset deviates from the current market baseline
    df['M_zscore'] = (df['price_momentum'] - df['price_momentum'].mean()) / df['price_momentum'].std()
    df['S_zscore'] = (df['social_heat'] - df['social_heat'].mean()) / df['social_heat'].std()

    # Fill NaNs with 0 (neutral) to handle zero-variance edge cases
    df.fillna(0, inplace=True)

    # 3. Divergence Alpha Calculation (The Secret Weapon)
    # High Alpha = Social narrative is building heavily, but price hasn't caught up yet.
    df['alpha'] = df['S_zscore'] - df['M_zscore']

    # 4. Extract Top Assets
    top_assets = df.sort_values(by='alpha', ascending=False).head(max_assets).copy()

    # 5. Softmax Weighting
    # Softmax guarantees all weights are positive and mathematically sum to 1.0.
    exp_alpha = np.exp(top_assets['alpha'])
    top_assets['weight'] = exp_alpha / exp_alpha.sum()

    # Convert to the exact dictionary format required by our Pydantic schema
    allocations = top_assets.set_index('ticker')['weight'].to_dict()

    # 6. Floating Point Normalization
    # Fixes rounding errors so Pydantic's exact 1.0 summation validator doesn't crash
    clean_allocations = {k: round(v, 4) for k, v in allocations.items()}
    difference = 1.0 - sum(clean_allocations.values())

    if difference != 0:
        first_key = list(clean_allocations.keys())[0]
        clean_allocations[first_key] = round(clean_allocations[first_key] + difference, 4)

    return clean_allocations

# --- Helper for Execution Rules ---

def generate_execution_rules() -> Dict[str, Dict[str, str]]:
    """
    Returns the deterministic entry and exit logic for the Pydantic spec.
    """
    return {
        "entry_logic": {
            "condition_1": "S_zscore > 1.0",
            "condition_2": "M_zscore < 0.5",
            "action": "allocate_capital"
        },
        "exit_logic": {
            "condition_1": "S_zscore < -1.0",
            "condition_2": "M_zscore > 1.5",
            "action": "liquidate_to_base"
        }
    }
