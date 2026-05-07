import pytest
import math
import numpy as np

from pde import BlackScholesPDE
from solver import ExplicitSolver, ImplicitSolver
from main import bs_put_price

def test_bs_put_price():
    # Known values or approximations
    S = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = 1.0
    price = bs_put_price(S, K, r, sigma, T)
    # The analytical B-S price is approximately 5.5735
    assert pytest.approx(price, rel=1e-3) == 5.5735

def test_pde_coefficients():
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100, T=1.0)
    
    # Check a(t, z) = -0.5 * sigma^2 * z^2
    assert pde.a(0, 100) == -0.5 * 0.2**2 * 100**2
    
    # Check b(t, z) = -r * z
    assert pde.b(0, 100) == -0.05 * 100
    
    # Check c(t, z) = r
    assert pde.c(0, 100) == 0.05
    
    # Check d(t, z) = 0
    assert pde.d(0, 100) == 0.0
    
    # Check f(z) = max(K - z, 0)
    assert pde.f(90) == 10
    assert pde.f(110) == 0
    
    # Check fl(t) = e^{-r(T-t)} * K
    assert pde.fl(0) == math.exp(-0.05 * 1.0) * 100
    
    # Check fu(t) = 0
    assert pde.fu(0) == 0.0

def test_explicit_solver():
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100, T=1.0, xl=0.01, xu=200)
    solver = ExplicitSolver(pde, imax=1000, jmax=100)
    solver.solve()
    price = solver.price_at(0.0, 100.0)
    analytical = bs_put_price(100.0, 100.0, 0.05, 0.2, 1.0)
    # The explicit finite difference solution should be close to the analytical
    assert pytest.approx(price, rel=1e-2) == analytical

def test_implicit_solver():
    pde = BlackScholesPDE(r=0.05, sigma=0.2, K=100, T=1.0, xl=0.01, xu=200)
    solver = ImplicitSolver(pde, imax=1000, jmax=100)
    solver.solve()
    price = solver.price_at(0.0, 100.0)
    analytical = bs_put_price(100.0, 100.0, 0.05, 0.2, 1.0)
    # The implicit finite difference solution should be close to the analytical
    assert pytest.approx(price, rel=1e-2) == analytical
