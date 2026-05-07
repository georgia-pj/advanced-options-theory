"""
solver.py
=========
Abstract base class for finite difference solvers and two concrete
implementations — an Explicit Scheme and an Implicit Scheme — for
solving parabolic PDEs of the form defined in ``pde.ParabolicPDE``.

Mathematical reference: aot-tutorial.tex §4 (Explicit) and §5 (Implicit).

Grid convention
---------------
    i = 0, 1, …, imax   (time axis, t_i = i * dt,  t=0 earliest, t=T latest)
    j = 0, 1, …, jmax   (space axis, x_j = xl + j * dx)

The solution array ``v[i, j]`` stores the option value at (t_i, x_j).
Backward-in-time integration: start from the terminal condition at i=imax
and march down to i=0.
"""

from abc import ABC, abstractmethod
import warnings

import numpy as np
from scipy.linalg import solve_banded

from pde import ParabolicPDE


class FiniteDifferenceSolver(ABC):
    """
    Abstract base class for finite difference solvers.

    Parameters
    ----------
    pde  : ParabolicPDE  The PDE to solve (provides coefficients and BCs).
    imax : int           Number of time steps.
    jmax : int           Number of spatial steps.
    """

    def __init__(self, pde: ParabolicPDE, imax: int = 1000, jmax: int = 100) -> None:
        self.pde = pde
        self.imax = imax
        self.jmax = jmax

        # Grid spacing
        self.dt = pde.T / imax
        self.dx = (pde.xu - pde.xl) / jmax

        # Grid nodes
        self.t_grid = np.linspace(0.0, pde.T, imax + 1)          # length imax+1
        self.x_grid = np.linspace(pde.xl, pde.xu, jmax + 1)      # length jmax+1

        # Solution array (full grid, initialised to NaN)
        self.v = np.full((imax + 1, jmax + 1), np.nan)

        # Apply terminal condition at i = imax
        for j in range(jmax + 1):
            self.v[imax, j] = pde.f(self.x_grid[j])

        # Apply lower and upper boundary conditions at all time levels
        for i in range(imax + 1):
            self.v[i, 0]    = pde.fl(self.t_grid[i])
            self.v[i, jmax] = pde.fu(self.t_grid[i])

    # ------------------------------------------------------------------
    # Convenience coefficient accessors
    # ------------------------------------------------------------------

    def _a(self, i: int, j: int) -> float:
        return self.pde.a(self.t_grid[i], self.x_grid[j])

    def _b(self, i: int, j: int) -> float:
        return self.pde.b(self.t_grid[i], self.x_grid[j])

    def _c(self, i: int, j: int) -> float:
        return self.pde.c(self.t_grid[i], self.x_grid[j])

    def _d(self, i: int, j: int) -> float:
        return self.pde.d(self.t_grid[i], self.x_grid[j])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Perform the backward-in-time sweep and return the full solution
        array ``v`` of shape ``(imax+1, jmax+1)``.
        """

    # ------------------------------------------------------------------
    # Convenience result accessor
    # ------------------------------------------------------------------

    def price_at(self, t: float, x: float) -> float:
        """
        Interpolate the solution to return the option value at (t, x).

        Parameters
        ----------
        t : float  Time (0 ≤ t ≤ T).  Use t=0 for the current price.
        x : float  Spot price.

        Returns
        -------
        float  Interpolated option price.
        """
        i_float = t / self.dt
        j_float = (x - self.pde.xl) / self.dx
        i0 = int(np.clip(i_float, 0, self.imax - 1))
        j0 = int(np.clip(j_float, 0, self.jmax - 1))
        # Bilinear interpolation
        wi = i_float - i0
        wj = j_float - j0
        return (
            (1 - wi) * (1 - wj) * self.v[i0,     j0    ]
          + (1 - wi) *      wj  * self.v[i0,     j0 + 1]
          +      wi  * (1 - wj) * self.v[i0 + 1, j0    ]
          +      wi  *      wj  * self.v[i0 + 1, j0 + 1]
        )


# ---------------------------------------------------------------------------
# Explicit Scheme  (aot-tutorial.tex §4)
# ---------------------------------------------------------------------------

class ExplicitSolver(FiniteDifferenceSolver):
    """
    Explicit finite difference scheme.

    At each interior time level i (from imax down to 1) and for each
    interior spatial node j (1 to jmax−1), the update rule is:

        v[i-1, j] = A_{i,j} v[i, j-1]
                  + B_{i,j} v[i, j  ]
                  + C_{i,j} v[i, j+1]
                  + D_{i,j}

    Coefficients (cf. tutorial §4):
        A_{i,j} = (dt/dx) * (b_{i,j}/2 − a_{i,j}/dx)
        B_{i,j} = 1 − dt * c_{i,j} + 2*dt*a_{i,j}/dx²
        C_{i,j} = −(dt/dx) * (b_{i,j}/2 + a_{i,j}/dx)
        D_{i,j} = −dt * d_{i,j}

    .. warning::
        The explicit scheme is only conditionally stable.  A CFL-like
        condition is checked and a warning is raised if it is violated.
    """

    def _coefficients(self, i: int, j: int):
        """Return (A, B, C, D) for the explicit update at (i, j)."""
        a = self._a(i, j)
        b = self._b(i, j)
        c = self._c(i, j)
        d = self._d(i, j)
        dt, dx = self.dt, self.dx

        A = (dt / dx) * (b / 2.0 - a / dx)
        B = 1.0 - dt * c + 2.0 * dt * a / dx ** 2
        C = -(dt / dx) * (b / 2.0 + a / dx)
        D = -dt * d
        return A, B, C, D

    def _check_stability(self) -> None:
        """
        Heuristic stability check: warn if the maximum absolute value of
        the B coefficient drops below zero at any interior node for i=imax.
        """
        for j in range(1, self.jmax):
            _, B, _, _ = self._coefficients(self.imax, j)
            if B < 0.0:
                warnings.warn(
                    f"Explicit scheme may be unstable (B={B:.4f} < 0 at j={j}). "
                    "Consider increasing imax or decreasing jmax.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return  # warn once

    def solve(self) -> np.ndarray:
        """Run the explicit backward sweep and return the solution grid."""
        self._check_stability()

        for i in range(self.imax, 0, -1):
            for j in range(1, self.jmax):
                A, B, C, D = self._coefficients(i, j)
                self.v[i - 1, j] = (
                    A * self.v[i, j - 1]
                    + B * self.v[i, j    ]
                    + C * self.v[i, j + 1]
                    + D
                )
            # Boundary conditions are already set in __init__

        return self.v


# ---------------------------------------------------------------------------
# Implicit Scheme  (aot-tutorial.tex §5)
# ---------------------------------------------------------------------------

class ImplicitSolver(FiniteDifferenceSolver):
    """
    Implicit finite difference scheme.

    At each time step the update is written as the linear system:

        B_i * v_{i-1} = A_i * v_i + w_i

    where:
      - ``v_i`` is the column vector of interior values at time level i.
      - ``A_i`` is the identity matrix  (A_{i,j}=0, B_{i,j}=1, C_{i,j}=0).
      - ``B_i`` is a tridiagonal matrix with diagonals E, F, G.
      - ``w_i`` absorbs the boundary corrections and D terms.

    Scalars (cf. tutorial §5):
        E_{i,j} = −(dt/dx) * (b_{i-1,j}/2 − a_{i-1,j}/dx)
        F_{i,j} =  1 + dt * c_{i-1,j} − 2*dt*a_{i-1,j}/dx²
        G_{i,j} =  (dt/dx) * (b_{i-1,j}/2 + a_{i-1,j}/dx)
        D_{i,j} = −dt * d_{i-1,j}

    The tridiagonal system is solved efficiently using
    ``scipy.linalg.solve_banded``.
    """

    def _implicit_coefficients(self, i: int, j: int):
        """
        Return (E, F, G, D) for the implicit scheme at time level i-1
        and spatial node j.  Note: coefficients are evaluated at (i-1, j).
        """
        # coefficients evaluated one step earlier
        a = self.pde.a(self.t_grid[i - 1], self.x_grid[j])
        b = self.pde.b(self.t_grid[i - 1], self.x_grid[j])
        c = self.pde.c(self.t_grid[i - 1], self.x_grid[j])
        d = self.pde.d(self.t_grid[i - 1], self.x_grid[j])
        dt, dx = self.dt, self.dx

        E = -(dt / dx) * (b / 2.0 - a / dx)
        F =  1.0 + dt * c - 2.0 * dt * a / dx ** 2
        G =  (dt / dx) * (b / 2.0 + a / dx)
        D = -dt * d
        return E, F, G, D

    def solve(self) -> np.ndarray:
        """Run the implicit backward sweep and return the solution grid."""
        n = self.jmax - 1  # number of interior nodes

        for i in range(self.imax, 0, -1):
            # ----------------------------------------------------------
            # Build the banded matrix B_i  (scipy format: (3, n))
            # Row 0: super-diagonal G  (upper)
            # Row 1: main diagonal  F
            # Row 2: sub-diagonal   E  (lower)
            # ----------------------------------------------------------
            ab = np.zeros((3, n))
            rhs = np.zeros(n)

            for idx, j in enumerate(range(1, self.jmax)):
                E, F, G, D = self._implicit_coefficients(i, j)

                # Banded storage (scipy convention)
                ab[1, idx] = F           # main diagonal
                if idx > 0:
                    ab[2, idx - 1] = E   # sub-diagonal (stored in row 2, col idx-1)
                if idx < n - 1:
                    ab[0, idx + 1] = G   # super-diagonal (stored in row 0, col idx+1)

                # RHS: A_i * v_i  (identity, so just v[i, j]) + D + boundary terms
                rhs[idx] = self.v[i, j] + D

            # Incorporate boundary corrections into the RHS
            # j=1: subtract E * v[i-1, 0]  (lower boundary at i-1)
            E0, _, _, _ = self._implicit_coefficients(i, 1)
            rhs[0] -= E0 * self.v[i - 1, 0]

            # j=jmax-1: subtract G * v[i-1, jmax]  (upper boundary at i-1)
            _, _, Gn, _ = self._implicit_coefficients(i, self.jmax - 1)
            rhs[-1] -= Gn * self.v[i - 1, self.jmax]

            # ----------------------------------------------------------
            # Solve B_i * v_{i-1} = rhs
            # ----------------------------------------------------------
            interior = solve_banded((1, 1), ab, rhs)
            self.v[i - 1, 1:self.jmax] = interior

        return self.v
