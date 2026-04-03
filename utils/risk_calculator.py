import math

def calculate_trade_risk(account_size, risk_pct, entry_price, stop_loss, target_price):
    """
    Calculates position size, total risk, and reward-to-risk ratio.
    Assumes a 'Long' position (buying low, selling high).
    """
    # Safeguards against invalid math
    if entry_price <= stop_loss:
        return None, "Stop loss must be strictly lower than the entry price for a long trade."
    if entry_price <= 0 or stop_loss <= 0 or target_price <= 0:
        return None, "Prices must be greater than zero."

    # 1. Calculate Dollar Risk (How much money are we willing to lose?)
    dollar_risk = account_size * (risk_pct / 100)

    # 2. Calculate Risk Per Share (Entry - Stop Loss)
    risk_per_share = entry_price - stop_loss

    # 3. Calculate Max Shares (Dollar Risk / Risk Per Share)
    # We use math.floor to round down. You can't buy half a share, and rounding up exceeds our risk!
    max_shares = math.floor(dollar_risk / risk_per_share)

    if max_shares <= 0:
        return None, "Calculated share size is 0. Your stop loss is too wide for this account size/risk %."

    # 4. Calculate Total Capital Required (Position Size)
    position_size = max_shares * entry_price

    # 5. Calculate Reward metrics
    expected_profit = max_shares * (target_price - entry_price)
    reward_to_risk = (target_price - entry_price) / risk_per_share

    # Return a dictionary of the results
    results = {
        "shares": max_shares,
        "dollar_risk": dollar_risk,
        "position_size": position_size,
        "expected_profit": expected_profit,
        "rr_ratio": reward_to_risk
    }
    
    return results, None