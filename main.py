"""
main.py
=======
Example script: pricing a European Put option via the Black-Scholes PDE
using both the Explicit and Implicit finite difference schemes, then
comparing against the closed-form analytical Black-Scholes price.

Standard parameters
-------------------
    r     = 0.05   (5%  risk-free rate)
    sigma = 0.20   (20% volatility)
    K     = 100    (strike price)
    T     = 1.0    (1 year to expiry)
    xl    = 0.01   (lower spot bound)
    xu    = 200.0  (upper spot bound = 2 × K)
    imax  = 1000   (time steps)
    jmax  = 100    (spatial steps)
"""

import math
import time

import numpy as np
from scipy.stats import norm

from pde import BlackScholesPDE
from solver import ExplicitSolver, ImplicitSolver


# ---------------------------------------------------------------------------
# Analytical Black-Scholes price (benchmark)
# ---------------------------------------------------------------------------

def bs_put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Closed-form Black-Scholes price for a European Put option.

    Parameters
    ----------
    S     : float  Current spot price.
    K     : float  Strike price.
    r     : float  Risk-free rate (annualised).
    sigma : float  Volatility (annualised).
    T     : float  Time to expiry (years).

    Returns
    -------
    float  Put option price.
    """
    if T <= 0.0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Standard parameters
    # ------------------------------------------------------------------
    r     = 0.05    # Risk-free interest rate
    sigma = 0.20    # Volatility
    K     = 100.0   # Strike price
    T     = 1.0     # Time to expiry (years)
    xl    = 0.01    # Lower bound for spot price
    xu    = 200.0   # Upper bound for spot price (2 × K)
    imax  = 1000    # Number of time steps
    jmax  = 100     # Number of spatial steps

    # Spot price at which we query the option value (at-the-money)
    S0 = 100.0

    print("=" * 60)
    print("  European Put Option — Finite Difference Pricing")
    print("=" * 60)
    print(f"  Parameters:")
    print(f"    r      = {r:.2%}  (risk-free rate)")
    print(f"    sigma  = {sigma:.2%}  (volatility)")
    print(f"    K      = {K:.2f}  (strike price)")
    print(f"    T      = {T:.2f}  (time to expiry, years)")
    print(f"    xl     = {xl:.4f}  (lower spot bound)")
    print(f"    xu     = {xu:.2f}  (upper spot bound)")
    print(f"    imax   = {imax}  (time steps)")
    print(f"    jmax   = {jmax}   (spatial steps)")
    print(f"  Query point: S = {S0:.2f} (at-the-money)")
    print()

    # ------------------------------------------------------------------
    # Instantiate the PDE
    # ------------------------------------------------------------------
    pde = BlackScholesPDE(r=r, sigma=sigma, K=K, T=T, xl=xl, xu=xu)

    # ------------------------------------------------------------------
    # Explicit Scheme
    # ------------------------------------------------------------------
    print("Running Explicit Scheme … ", end="", flush=True)
    t_start = time.perf_counter()
    explicit_solver = ExplicitSolver(pde, imax=imax, jmax=jmax)
    explicit_solver.solve()
    t_explicit = time.perf_counter() - t_start
    price_explicit = explicit_solver.price_at(t=0.0, x=S0)
    print(f"done in {t_explicit:.3f}s")
    print(f"  Explicit price at S={S0}: {price_explicit:.6f}")

    # ------------------------------------------------------------------
    # Implicit Scheme
    # ------------------------------------------------------------------
    print("Running Implicit Scheme … ", end="", flush=True)
    t_start = time.perf_counter()
    implicit_solver = ImplicitSolver(pde, imax=imax, jmax=jmax)
    implicit_solver.solve()
    t_implicit = time.perf_counter() - t_start
    price_implicit = implicit_solver.price_at(t=0.0, x=S0)
    print(f"done in {t_implicit:.3f}s")
    print(f"  Implicit price at S={S0}: {price_implicit:.6f}")

    # ------------------------------------------------------------------
    # Analytical benchmark
    # ------------------------------------------------------------------
    price_analytical = bs_put_price(S=S0, K=K, r=r, sigma=sigma, T=T)
    print()
    print(f"  Analytical B-S price    : {price_analytical:.6f}")
    print()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"  {'Method':<20} {'Price':>10} {'Error':>10} {'Time (s)':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for label, price, elapsed in [
        ("Explicit",   price_explicit, t_explicit),
        ("Implicit",   price_implicit, t_implicit),
        ("Analytical", price_analytical, float("nan")),
    ]:
        err = abs(price - price_analytical)
        t_str = f"{elapsed:.3f}" if not math.isnan(elapsed) else "—"
        print(f"  {label:<20} {price:>10.6f} {err:>10.6f} {t_str:>10}")
    print("-" * 60)

    # ------------------------------------------------------------------
    # Full price curve at t=0 (for spot in [xl, xu])
    # ------------------------------------------------------------------
    print()
    print("  Price curve at t=0 (selected spot prices):")
    print(f"  {'Spot (S)':>10} {'Explicit':>12} {'Implicit':>12} {'Analytical':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    spots = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    for S in spots:
        pe = explicit_solver.price_at(0.0, S)
        pi = implicit_solver.price_at(0.0, S)
        pa = bs_put_price(S, K, r, sigma, T)
        print(f"  {S:>10.1f} {pe:>12.6f} {pi:>12.6f} {pa:>12.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
