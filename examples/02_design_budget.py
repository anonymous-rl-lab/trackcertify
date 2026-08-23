"""Pre-experiment design: T*, optimal allocation, coupling tax."""
import track_certify as tc

inst = tc.solvable_instance(r=0.9, c1=1.0, c2=1.0)
rep = tc.characteristic_time(inst.model, inst.truth, inst.amplitudes)
tax = tc.coupling_tax(inst.model, inst.truth, inst.amplitudes)
print("T* (LP)        :", round(rep.tstar, 4), "| closed form:",
      round(inst.tstar_closed_form, 4))
print("optimal split  :", tuple(round(w, 3) for w in rep.w_star))
print("active kinds   :", rep.active_kinds)
print("coupling tax   :", round(tax.tax, 3))
print("budget @ 1e-3  :", round(rep.budget(1e-3), 1), "samples (first order)")
