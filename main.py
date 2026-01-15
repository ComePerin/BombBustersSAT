from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Solver


def scale10(x) -> int:
    """
    Convert a numeric value (int/float/str) to an integer scaled by 10.

    Examples:
      7      -> 70
      2.1    -> 21
      "11.5" -> 115

    Notes:
      - This helper assumes at most one decimal digit in the input string.
      - Using an integer representation avoids floating-point issues.
    """
    if isinstance(x, int):
        return x * 10
    if isinstance(x, float):
        return int(round(x * 10))
    if isinstance(x, str):
        s = x.strip()
        if "." in s:
            a, b = s.split(".")
            b = (b + "0")[:1]  # keep one decimal digit
            return int(a) * 10 + int(b)
        return int(s) * 10
    raise TypeError(f"Unsupported type for value: {type(x)}")


def unscale10(v10: int) -> str:
    """
    Convert an integer scaled by 10 back to a compact string.

    Examples:
      70  -> "7"
      21  -> "2.1"
      115 -> "11.5"
    """
    a, b = divmod(v10, 10)
    return f"{a}" if b == 0 else f"{a}.{b}"


@dataclass
class InstanceData:
    """
    Container for a single game instance.

    Players are indexed from 1 to p.
    Card positions t are indexed from 1 to N_i for player i.
    """
    p: int
    N: int  # max blue value (here: blue values are {1,...,N})
    yellow_values: List[int]  # scaled by 10 (e.g., 21 for 2.1)
    red_values: List[int]     # scaled by 10 (e.g., 115 for 11.5)
    Ni: Dict[int, int]        # Ni[i] = number of cards for player i
    observed_values: Dict[Tuple[int, int], int]  # (i,t) -> value*10
    observed_yellow_positions: Set[Tuple[int, int]]  # positions known to be yellow (color only)


class BombBustersSAT:
    """
    SAT encoding for Bomb Busters based on a one-hot value representation.

    This class:
      - Builds a CNF encoding of the current knowledge about the game state
      - Solves and queries satisfiability under assumptions
      - Can export, for each player, a matrix of feasible values per position
    """

    def __init__(self, data: InstanceData, solver_name: str = "glucose3"):
        self.data = data
        self.vpool = IDPool()
        self.cnf = CNF()
        self.solver_name = solver_name

        # Universe of card positions P
        self.P: List[Tuple[int, int]] = [
            (i, t)
            for i in range(1, data.p + 1)
            for t in range(1, data.Ni[i] + 1)
        ]

        # In this instance, blue values are {1,...,N} (no 0)
        self.blue_vals: List[int] = [10 * v for v in range(1, data.N + 1)]
        self.yellow_vals: List[int] = list(data.yellow_values)
        self.red_vals: List[int] = list(data.red_values)

        # Global set of possible values (scaled by 10), deduplicated and sorted
        self.V: List[int] = sorted(set(self.blue_vals + self.yellow_vals + self.red_vals))

        # Build CNF
        self._add_exactly_one_color()
        self._add_exactly_one_value()
        self._add_color_value_compatibility()
        self._add_order_constraints()
        self._add_multiplicity_constraints()
        self._add_observations()

        # Initialize solver with CNF clauses
        self.solver = Solver(name=self.solver_name, bootstrap_with=self.cnf.clauses)

    # ---------- SAT variables ----------
    def var_b(self, i: int, t: int) -> int:
        """Boolean variable: True iff card (i,t) is blue."""
        return self.vpool.id(("b", i, t))

    def var_j(self, i: int, t: int) -> int:
        """Boolean variable: True iff card (i,t) is yellow."""
        return self.vpool.id(("j", i, t))

    def var_r(self, i: int, t: int) -> int:
        """Boolean variable: True iff card (i,t) is red."""
        return self.vpool.id(("r", i, t))

    def var_x(self, i: int, t: int, v: int) -> int:
        """Boolean variable: True iff card (i,t) has value v (scaled by 10)."""
        return self.vpool.id(("x", i, t, v))

    # ---------- CNF construction ----------
    def _add_exactly_one_color(self) -> None:
        """
        For each card position (i,t), enforce exactly one color among {blue, yellow, red}.
        """
        for (i, t) in self.P:
            lits = [self.var_b(i, t), self.var_j(i, t), self.var_r(i, t)]
            enc = CardEnc.equals(lits=lits, bound=1, vpool=self.vpool, encoding=1)
            self.cnf.extend(enc.clauses)

    def _add_exactly_one_value(self) -> None:
        """
        For each card position (i,t), enforce exactly one value among V (one-hot).
        """
        for (i, t) in self.P:
            lits = [self.var_x(i, t, v) for v in self.V]
            enc = CardEnc.equals(lits=lits, bound=1, vpool=self.vpool, encoding=1)
            self.cnf.extend(enc.clauses)

    def _add_color_value_compatibility(self) -> None:
        """
        Color -> allowed values constraints:
          - blue  -> value in blue_vals
          - yellow -> value in yellow_vals
          - red   -> value in red_vals
        """
        for (i, t) in self.P:
            b = self.var_b(i, t)
            j = self.var_j(i, t)
            r = self.var_r(i, t)

            self.cnf.append([-b] + [self.var_x(i, t, v) for v in self.blue_vals])
            self.cnf.append([-j] + [self.var_x(i, t, v) for v in self.yellow_vals])
            self.cnf.append([-r] + [self.var_x(i, t, v) for v in self.red_vals])

    def _add_order_constraints(self) -> None:
        """
        For each player i: X_i(t) <= X_i(t+1).
        CNF encoding: for all v > v', forbid X_i(t)=v and X_i(t+1)=v'.
        """
        for i in range(1, self.data.p + 1):
            for t in range(1, self.data.Ni[i]):
                for v in self.V:
                    for vp in self.V:
                        if v > vp:
                            self.cnf.append([
                                -self.var_x(i, t, v),
                                -self.var_x(i, t + 1, vp)
                            ])

    def _add_multiplicity_constraints(self) -> None:
        """
        Global multiplicity constraints:
          - each blue value appears exactly 4 times
          - each yellow value appears exactly as often as in yellow_values (duplicates allowed)
          - each red value appears exactly as often as in red_values (duplicates allowed)
        """
        # Blue: each v occurs exactly 4 times over all positions
        for v in self.blue_vals:
            lits = [self.var_x(i, t, v) for (i, t) in self.P]
            enc = CardEnc.equals(lits=lits, bound=4, vpool=self.vpool, encoding=1)
            self.cnf.extend(enc.clauses)

        # Yellow and red: multiplicities from provided lists
        from collections import Counter

        cy = Counter(self.yellow_vals)
        for v, k in cy.items():
            lits = [self.var_x(i, t, v) for (i, t) in self.P]
            enc = CardEnc.equals(lits=lits, bound=k, vpool=self.vpool, encoding=1)
            self.cnf.extend(enc.clauses)

        cr = Counter(self.red_vals)
        for v, k in cr.items():
            lits = [self.var_x(i, t, v) for (i, t) in self.P]
            enc = CardEnc.equals(lits=lits, bound=k, vpool=self.vpool, encoding=1)
            self.cnf.extend(enc.clauses)

    def _add_observations(self) -> None:
        """
        Add observations as unit clauses:
          - observed values: x_{i,t,value} is True
          - observed yellow positions (color only): j_{i,t} is True
        """
        for (i, t), v in self.data.observed_values.items():
            self.cnf.append([self.var_x(i, t, v)])

        for (i, t) in self.data.observed_yellow_positions:
            self.cnf.append([self.var_j(i, t)])

    # ---------- Solver API ----------
    def is_sat(self, assumptions: Optional[List[int]] = None) -> bool:
        """Check satisfiability under optional assumptions."""
        return self.solver.solve(assumptions=assumptions or [])

    def possible_values(self, i: int, t: int) -> List[int]:
        """
        Return all values v in V such that X_i(t) = v is consistent with the current constraints.
        """
        poss: List[int] = []
        for v in self.V:
            if self.is_sat([self.var_x(i, t, v)]):
                poss.append(v)
        return poss

    def possible_colors(self, i: int, t: int) -> List[str]:
        """
        Return the set of possible colors for position (i,t).
        """
        out: List[str] = []
        if self.is_sat([self.var_b(i, t)]):
            out.append("bleu")
        if self.is_sat([self.var_j(i, t)]):
            out.append("jaune")
        if self.is_sat([self.var_r(i, t)]):
            out.append("rouge")
        return out

    def add_observed_value(self, i: int, t: int, value10: int) -> None:
        """
        Incrementally add an observed value: X_i(t) = value10.
        """
        self.solver.add_clause([self.var_x(i, t, value10)])

    def add_observed_yellow(self, i: int, t: int) -> None:
        """
        Incrementally add an observed yellow color (color only).
        """
        self.solver.add_clause([self.var_j(i, t)])

    def close(self) -> None:
        """Release the underlying solver."""
        self.solver.delete()

    # ---------- Feasibility matrices & export ----------
    def feasible_value_matrix(self, i: int) -> List[List[bool]]:
        """
        Build a feasibility matrix for player i.

        Rows correspond to values in self.V (in increasing order).
        Columns correspond to positions t = 1..N_i.

        Entry [row][col] is True iff X_i(t) can take that value.
        """
        n_rows = len(self.V)
        n_cols = self.data.Ni[i]
        mat = [[False] * n_cols for _ in range(n_rows)]

        for col, t in enumerate(range(1, n_cols + 1)):
            for row, v in enumerate(self.V):
                mat[row][col] = self.is_sat([self.var_x(i, t, v)])
        return mat

    def export_feasibility_heatmap(
        self,
        i: int,
        out_path: str | Path,
        *,
        title: Optional[str] = None,
        show_value_labels: bool = True,
    ) -> Path:
        """
        Export a heatmap image for the feasibility matrix of player i.

        Green cells correspond to satisfiable assignments X_i(t)=v,
        red cells correspond to unsatisfiable assignments.

        Parameters:
          - i: player index (1..p)
          - out_path: output file path (e.g., "out/player1.png")
          - title: optional plot title
          - show_value_labels: whether to display y-axis labels for values
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mat = np.array(self.feasible_value_matrix(i), dtype=int)

        # 0 -> red, 1 -> green
        cmap = ListedColormap(["#d9534f", "#5cb85c"])
        norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * self.data.Ni[i]), max(6, 0.25 * len(self.V))))

        ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

        # Axes: columns = positions (1..N_i), rows = values in V
        ax.set_xlabel("Position t")
        ax.set_ylabel("Valeur v")

        ax.set_xticks(list(range(self.data.Ni[i])))
        ax.set_xticklabels([str(t) for t in range(1, self.data.Ni[i] + 1)])

        if show_value_labels:
            ax.set_yticks(list(range(len(self.V))))
            ax.set_yticklabels([unscale10(v) for v in self.V])
        else:
            ax.set_yticks([])

        ax.set_title(title or f"Joueur {i} — Valeurs possibles par position")

        # Grid for readability
        ax.set_xticks([x - 0.5 for x in range(1, self.data.Ni[i])], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(self.V))], minor=True)
        ax.grid(which="minor", linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def export_all_players_heatmaps(self, out_dir: str | Path) -> List[Path]:
        """
        Export one feasibility heatmap per player into out_dir.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: List[Path] = []
        for i in range(1, self.data.p + 1):
            paths.append(self.export_feasibility_heatmap(i, out_dir / f"player_{i}_feasible.png"))
        return paths


# --- Step 1 (example instance used for early testing) ---
# Global parameters:
#   - Number of players: p = 4
#   - Observer: player 0
#   - Blue max value: N = 12   (blue values are {1,...,12} in this instance)
#
# Non-blue cards (globally known multiset of values):
#   - Yellow values: [2.1, 7.1]
#   - Red values:    [11.5]
#
# Full hands (ground truth, for validation only; NOT available to the assistant):
#   - Player 1: [1, 1, 2, 2, 3, 4, 6, 7, 9, 9, 10, 11.5, 12]
#   - Player 2: [1, 2, 3, 3, 4, 5, 5, 6, 7.1, 8, 8, 10]
#   - Player 3: [2.1, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 12]
#   - Player 4: [1, 2, 3, 4, 4, 6, 7, 7, 9, 10, 11, 11, 12]
#
# Public partial observations at step 1 (asterisk = unknown value at that position):
#   - Player 1: [1, 1, 2, 2, 3, *, *, 7, *, *, *, *, 12]
#   - Player 2: [1, 2, 3, 3, *, *, *, *, *, 8, 8, 10]
#   - Player 3: [*, *, *, *, 7, *, *, *, *, *, *, 12, *]
#   - Player 4: [1, 2, 3, *, *, *, 7, 7, *, 10, 11, *, *]
# --------------------------------------------------------

def build_step1_instance() -> InstanceData:
    p = 4
    N = 12
    yellow_values = [scale10("2.1"), scale10("7.1")]
    red_values = [scale10("11.5")]

    Ni = {
        1: 13,
        2: 12,
        3: 13,
        4: 13,
    }

    # Partial observations: known values are set, unknowns are None.
    obs_p1 = [1, 1, 2, 2, 3, None, None, 7, None, None, None, None, 12]
    obs_p2 = [1, 2, 3, 3, None, None, None, None, None, 8, 8, 10]
    obs_p3 = [None, None, None, None, 7, None, None, None, None, None, None, 12, None]
    obs_p4 = [1, 2, 3, None, None, None, 7, 7, None, 10, 11, None, None]

    observed_values: Dict[Tuple[int, int], int] = {}
    for i, arr in [(1, obs_p1), (2, obs_p2), (3, obs_p3), (4, obs_p4)]:
        for t, val in enumerate(arr, start=1):
            if val is not None:
                observed_values[(i, t)] = scale10(val)

    # No explicit yellow token positions in this step.
    observed_yellow_positions: Set[Tuple[int, int]] = set()

    return InstanceData(
        p=p,
        N=N,
        yellow_values=yellow_values,
        red_values=red_values,
        Ni=Ni,
        observed_values=observed_values,
        observed_yellow_positions=observed_yellow_positions
    )


def print_possible_values_report(model: BombBustersSAT) -> None:
    """
    Print, for each player and each position, the set of feasible values.
    """
    for i in range(1, model.data.p + 1):
        print(f"\n=== Joueur {i} ===")
        for t in range(1, model.data.Ni[i] + 1):
            vals = model.possible_values(i, t)
            pretty = ", ".join(unscale10(v) for v in vals)
            print(f"  pos {t:>2}: {{{pretty}}}")

import time

if __name__ == "__main__":
    data = build_step1_instance()
    model = BombBustersSAT(data)

    print("SAT ?", model.is_sat())

    # 1) Console report: possible values per (player, position)
    print_possible_values_report(model)

    # 2) Export feasibility heatmaps as PNG files
    out_paths = model.export_all_players_heatmaps("out_heatmaps")
    print("\nHeatmaps exported:")
    for pth in out_paths:
        print(" ", pth)

    model.close()
