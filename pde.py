"""
pde.py
======
Abstract base class for Parabolic PDEs and a concrete Black-Scholes
implementation for a European Put option.

The general parabolic PDE considered here is (cf. aot-tutorial.tex §2):

    ∂v/∂t = a(t,x) ∂²v/∂x² + b(t,x) ∂v/∂x + c(t,x) v + d(t,x)

for (t,x) ∈ [0,T] × [x_l, x_u], subject to:
    - Terminal condition : v(T,  x)   = f(x)
    - Lower boundary     : v(t,  x_l) = fl(t)
    - Upper boundary     : v(t,  x_u) = fu(t)
"""

from abc import ABC, abstractmethod
import math


class ParabolicPDE(ABC):
    """
    Abstract base class representing a parabolic PDE of the form

        ∂v/∂t = a(t,x) ∂²v/∂x² + b(t,x) ∂v/∂x + c(t,x) v + d(t,x)

    Subclasses must implement the four coefficient methods and the three
    boundary/terminal condition methods, and expose the domain properties
    ``xl``, ``xu``, and ``T``.
    """

    # ------------------------------------------------------------------
    # Domain properties (must be set by concrete subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def xl(self) -> float:
        """Lower spatial boundary x_l."""

    @property
    @abstractmethod
    def xu(self) -> float:
        """Upper spatial boundary x_u."""

    @property
    @abstractmethod
    def T(self) -> float:
        """Terminal time T."""

    # ------------------------------------------------------------------
    # PDE coefficients
    # ------------------------------------------------------------------

    @abstractmethod
    def a(self, t: float, x: float) -> float:
        """Diffusion coefficient a(t,x)."""

    @abstractmethod
    def b(self, t: float, x: float) -> float:
        """Drift coefficient b(t,x)."""

    @abstractmethod
    def c(self, t: float, x: float) -> float:
        """Reaction coefficient c(t,x)."""

    @abstractmethod
    def d(self, t: float, x: float) -> float:
        """Source term d(t,x)."""

    # ------------------------------------------------------------------
    # Boundary and terminal conditions
    # ------------------------------------------------------------------

    @abstractmethod
    def f(self, x: float) -> float:
        """Terminal condition v(T, x) = f(x)."""

    @abstractmethod
    def fl(self, t: float) -> float:
        """Lower boundary condition v(t, x_l) = fl(t)."""

    @abstractmethod
    def fu(self, t: float) -> float:
        """Upper boundary condition v(t, x_u) = fu(t)."""


# ---------------------------------------------------------------------------
# Black-Scholes PDE for a European Put Option
# ---------------------------------------------------------------------------

class BlackScholesPDE(ParabolicPDE):
    """
    The Black-Scholes equation written in the canonical parabolic form
    (cf. aot-tutorial.tex §3):

        ∂u/∂t + (σ²/2) z² ∂²u/∂z² + r z ∂u/∂z − r u = 0

    Rearranged to match the sign convention ∂v/∂t = … :

        a(t,z) = −σ²z²/2
        b(t,z) = −rz
        c(t,z) =  r
        d(t,z) =  0

    Boundary conditions for a European Put with strike K and expiry T:
        f(z)   = max(K − z, 0)          (terminal payoff)
        fl(t)  = e^{−r(T−t)} · K        (lower: deep in-the-money)
        fu(t)  = 0                       (upper: deep out-of-the-money)

    Parameters
    ----------
    r     : float  Risk-free interest rate (annualised, e.g. 0.05).
    sigma : float  Volatility (annualised, e.g. 0.20).
    K     : float  Strike price (e.g. 100.0).
    T     : float  Time to expiry in years (e.g. 1.0).
    xl    : float  Lower bound for the spot price grid (e.g. 0.01).
    xu    : float  Upper bound for the spot price grid (e.g. 200.0).
    """

    def __init__(
        self,
        r: float = 0.05,
        sigma: float = 0.20,
        K: float = 100.0,
        T: float = 1.0,
        xl: float = 0.01,
        xu: float = 200.0,
    ) -> None:
        self.r = r
        self.sigma = sigma
        self.K = K
        self._T = T
        self._xl = xl
        self._xu = xu

    # ------------------------------------------------------------------
    # Domain properties
    # ------------------------------------------------------------------

    @property
    def xl(self) -> float:
        return self._xl

    @property
    def xu(self) -> float:
        return self._xu

    @property
    def T(self) -> float:
        return self._T

    # ------------------------------------------------------------------
    # PDE coefficients
    # ------------------------------------------------------------------

    def a(self, t: float, x: float) -> float:
        """a(t,z) = −σ²z²/2  (note the minus sign from rearrangement)."""
        return -0.5 * self.sigma ** 2 * x ** 2

    def b(self, t: float, x: float) -> float:
        """b(t,z) = −rz."""
        return -self.r * x

    def c(self, t: float, x: float) -> float:
        """c(t,z) = r."""
        return self.r

    def d(self, t: float, x: float) -> float:
        """d(t,z) = 0 (no source term)."""
        return 0.0

    # ------------------------------------------------------------------
    # Boundary and terminal conditions
    # ------------------------------------------------------------------

    def f(self, x: float) -> float:
        """Terminal payoff of a European Put: max(K − z, 0)."""
        return max(self.K - x, 0.0)

    def fl(self, t: float) -> float:
        """Lower boundary (spot → 0): discounted strike e^{−r(T−t)} K."""
        return math.exp(-self.r * (self._T - t)) * self.K

    def fu(self, t: float) -> float:
        """Upper boundary (spot → ∞): option is worthless, so 0."""
        return 0.0
