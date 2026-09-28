"""Console tables (compare_ALL_INDIA.m)."""
from .optimizers.common import OptResult


def comparison_table(g: OptResult, h: OptResult, m: OptResult) -> str:
    L = []; a = L.append
    a("=" * 69); a("   COMPARISON TABLE: GA-PSO vs HOA vs MPA - India (single run each)"); a("=" * 69)
    a(f"{'Metric':<24} {'GA-PSO':<16} {'HOA':<16} {'MPA':<16}"); a("-" * 72)
    a(f"{'Best Fitness (F)':<24} {g.gbest_fit:<16.6f} {h.gbest_fit:<16.6f} {m.gbest_fit:<16.6f}")
    for lab, bi, ci in (("RES 1 (Bus, kW)", 0, 2), ("RES 2 (Bus, kW)", 1, 3), ("CS 1 (Bus, kW)", 4, 6), ("CS 2 (Bus, kW)", 5, 7)):
        cells = [f"{int(r.gbest[bi])}, {r.gbest[ci] * 1000:.0f} kW" for r in (g, h, m)]
        a(f"{lab:<24} {cells[0]:<16} {cells[1]:<16} {cells[2]:<16}")
    if len(g.gbest) > 8:                                  # DSTATCOM problem
        for lab, bi, ci in (("DSTATCOM 1 (Bus, kVAr)", 8, 10), ("DSTATCOM 2 (Bus, kVAr)", 9, 11)):
            cells = [f"{int(r.gbest[bi])}, {r.gbest[ci] * 1000:.0f} kVAr" for r in (g, h, m)]
            a(f"{lab:<24} {cells[0]:<16} {cells[1]:<16} {cells[2]:<16}")
    a(f"{'F1 - Loss (MWh/day)':<24} {g.F1:<16.4f} {h.F1:<16.4f} {m.F1:<16.4f}")
    a(f"{'F2 - Voltage (pu)':<24} {g.F2:<16.4f} {h.F2:<16.4f} {m.F2:<16.4f}")
    a(f"{'F3 - Energy Cost (Rs)':<24} {g.F3:<16.0f} {h.F3:<16.0f} {m.F3:<16.0f}")
    a(f"{'F4 - Battery Wear (Rs)':<24} {g.F4:<16.2f} {h.F4:<16.2f} {m.F4:<16.2f}")
    a(f"{'NOFE (total)':<24} {g.NOFE:<16d} {h.NOFE:<16d} {m.NOFE:<16d}")
    a(f"{'Time (s)':<24} {g.time:<16.2f} {h.time:<16.2f} {m.time:<16.2f}"); a("-" * 72)
    win = min((g, h, m), key=lambda r: r.gbest_fit).algo
    a(f"Best single-run fitness: {win} (lower is better)")
    a(f"NOFE ratio GA-PSO/HOA = {g.NOFE / h.NOFE:.2f}x, GA-PSO/MPA = {g.NOFE / m.NOFE:.2f}x")
    a("NOTE: unequal budgets above make this comparison suggestive, not conclusive --\n"
      "run `python -m evbtp fair-compare` for the real, equal-budget, multi-seed comparison.")
    a("=" * 69)
    return "\n".join(L)
