import numpy as np

# ============================================================
# DATA
# ============================================================
oil  = np.array([270,362,162, 45, 91,233,372,305,234,122, 25,210,450,325, 52])
temp = np.array([ 40, 27, 40, 73, 65, 65, 10,  9, 24, 65, 66, 41, 22, 40, 60])
ins  = np.array([  4,  4, 10,  6,  7, 40,  6, 10, 10,  4, 10,  6,  4,  4, 10])
n = len(oil)

print("="*70)
print("PART (a): FULL DATA – Linear and Quadratic Models")
print("="*70)

# ---------- helper: OLS via normal equations ----------
def ols(X, y):
    """Returns beta, y_hat, R2, R2_adj"""
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    SS_res = np.sum((y - y_hat)**2)
    SS_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - SS_res / SS_tot
    n = len(y)
    p = X.shape[1]          # includes intercept
    R2_adj = 1 - (SS_res / (n - p)) / (SS_tot / (n - 1))
    return beta, y_hat, R2, R2_adj, SS_res, SS_tot

# ---- Linear model: Oil = b0 + b1*Temp + b2*Ins ----
X_lin = np.column_stack([np.ones(n), temp, ins])
beta_lin, yhat_lin, R2_lin, R2adj_lin, SSres_lin, SStot_lin = ols(X_lin, oil)

print("\n--- Linear Model ---")
print(f"  Oil = {beta_lin[0]:.4f} + ({beta_lin[1]:.4f})*Temp + ({beta_lin[2]:.4f})*Ins")
print(f"  R²     = {R2_lin:.6f}")
print(f"  R²_adj = {R2adj_lin:.6f}")
print(f"  SS_res = {SSres_lin:.2f}   SS_tot = {SStot_lin:.2f}")

# ---- Quadratic model: Oil = b0 + b1*T + b2*I + b3*T² + b4*I² + b5*T*I ----
X_quad = np.column_stack([np.ones(n), temp, ins, temp**2, ins**2, temp*ins])
beta_q, yhat_q, R2_q, R2adj_q, SSres_q, SStot_q = ols(X_quad, oil)

print("\n--- Quadratic Model ---")
print(f"  Oil = {beta_q[0]:.4f} + ({beta_q[1]:.4f})*T + ({beta_q[2]:.4f})*I")
print(f"        + ({beta_q[3]:.6f})*T² + ({beta_q[4]:.6f})*I² + ({beta_q[5]:.6f})*T*I")
print(f"  R²     = {R2_q:.6f}")
print(f"  R²_adj = {R2adj_q:.6f}")
print(f"  SS_res = {SSres_q:.2f}   SS_tot = {SStot_q:.2f}")

print(f"\n  ΔR² (quad – linear) = {R2_q - R2_lin:.6f}")
if R2_q - R2_lin < 0.05:
    print("  => Gain in R² is small; prefer the simpler linear model (fewer parameters).")
else:
    print("  => Quadratic model gives a meaningful improvement.")

# ============================================================
# PART (b): OUTLIER DETECTION & REMOVAL
# ============================================================
print("\n" + "="*70)
print("PART (b): OUTLIER DETECTION")
print("="*70)

# --- Residual analysis on linear model ---
residuals_lin = oil - yhat_lin
std_res = np.std(residuals_lin, ddof=X_lin.shape[1])
print("\n  Observation-by-observation residuals (linear model):")
print(f"  {'i':>3}  {'Oil':>5}  {'Temp':>4}  {'Ins':>3}  {'ŷ':>8}  {'resid':>8}  {'std_resid':>9}")
outlier_indices = []
for i in range(n):
    sr = residuals_lin[i] / std_res
    flag = " <-- OUTLIER" if abs(sr) > 2 else ""
    print(f"  {i+1:3d}  {oil[i]:5.0f}  {temp[i]:4.0f}  {ins[i]:3.0f}  {yhat_lin[i]:8.2f}  {residuals_lin[i]:8.2f}  {sr:9.3f}{flag}")
    if abs(sr) > 2:
        outlier_indices.append(i)

# Also flag point 6 (Insulation=40) based on domain knowledge
print(f"\n  Domain check: Insulation values = {sorted(ins)}")
print("  Observation 6 has Insulation = 40 inches, far outside the 4–10 range of all others.")
if 5 not in outlier_indices:
    outlier_indices.append(5)
    print("  Adding observation 6 as outlier based on domain knowledge.")

outlier_indices = sorted(set(outlier_indices))
print(f"\n  Outlier observation(s): {[i+1 for i in outlier_indices]}")

# Remove outliers
mask = np.ones(n, dtype=bool)
mask[outlier_indices] = False
oil_c  = oil[mask]
temp_c = temp[mask]
ins_c  = ins[mask]
n_c = len(oil_c)
print(f"  Cleaned data: {n_c} observations (removed {n - n_c})")

# ============================================================
# REFIT ON CLEANED DATA
# ============================================================
print("\n" + "="*70)
print("PART (b continued): MODELS ON CLEANED DATA")
print("="*70)

# Linear
X_lin_c = np.column_stack([np.ones(n_c), temp_c, ins_c])
beta_linc, yhat_linc, R2_linc, R2adj_linc, SSres_linc, SStot_linc = ols(X_lin_c, oil_c)

print("\n--- Linear Model (cleaned) ---")
print(f"  Oil = {beta_linc[0]:.4f} + ({beta_linc[1]:.4f})*Temp + ({beta_linc[2]:.4f})*Ins")
print(f"  R²     = {R2_linc:.6f}")
print(f"  R²_adj = {R2adj_linc:.6f}")

# Quadratic
X_quad_c = np.column_stack([np.ones(n_c), temp_c, ins_c, temp_c**2, ins_c**2, temp_c*ins_c])
beta_qc, yhat_qc, R2_qc, R2adj_qc, SSres_qc, SStot_qc = ols(X_quad_c, oil_c)

print("\n--- Quadratic Model (cleaned) ---")
print(f"  Oil = {beta_qc[0]:.4f} + ({beta_qc[1]:.4f})*T + ({beta_qc[2]:.4f})*I")
print(f"        + ({beta_qc[3]:.6f})*T² + ({beta_qc[4]:.6f})*I² + ({beta_qc[5]:.6f})*T*I")
print(f"  R²     = {R2_qc:.6f}")
print(f"  R²_adj = {R2adj_qc:.6f}")

print(f"\n  ΔR² (quad – linear, cleaned) = {R2_qc - R2_linc:.6f}")

# --- Comparison table ---
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(f"  {'Model':<28} {'R²':>9} {'R²_adj':>9} {'Params':>6}")
print(f"  {'Linear (full)':<28} {R2_lin:9.6f} {R2adj_lin:9.6f} {3:6d}")
print(f"  {'Quadratic (full)':<28} {R2_q:9.6f} {R2adj_q:9.6f} {6:6d}")
print(f"  {'Linear (cleaned)':<28} {R2_linc:9.6f} {R2adj_linc:9.6f} {3:6d}")
print(f"  {'Quadratic (cleaned)':<28} {R2_qc:9.6f} {R2adj_qc:9.6f} {6:6d}")

# Decide best model for cleaned data
if R2_qc - R2_linc < 0.05:
    print("\n  => For cleaned data the quadratic gain is small; LINEAR model is preferred (parsimony).")
    best_beta = beta_linc
    best_type = "linear"
else:
    print("\n  => For cleaned data the quadratic model is substantially better.")
    best_beta = beta_qc
    best_type = "quadratic"

# ============================================================
# PART (c): PREDICTION  T = 15 °F,  Insulation = 5 in
# ============================================================
T_new, I_new = 15, 5
print("\n" + "="*70)
print(f"PART (c): PREDICTION for T = {T_new} °F, Insulation = {I_new} in")
print("="*70)

# Predict with BOTH cleaned models
pred_lin = beta_linc[0] + beta_linc[1]*T_new + beta_linc[2]*I_new
print(f"\n  Linear (cleaned):    Oil = {beta_linc[0]:.4f} + ({beta_linc[1]:.4f})*{T_new} + ({beta_linc[2]:.4f})*{I_new}")
print(f"                       Oil = {pred_lin:.2f} gallons")

pred_quad = (beta_qc[0] + beta_qc[1]*T_new + beta_qc[2]*I_new
             + beta_qc[3]*T_new**2 + beta_qc[4]*I_new**2 + beta_qc[5]*T_new*I_new)
print(f"\n  Quadratic (cleaned): Oil = {beta_qc[0]:.4f} + ({beta_qc[1]:.4f})*{T_new} + ({beta_qc[2]:.4f})*{I_new}")
print(f"                             + ({beta_qc[3]:.6f})*{T_new}² + ({beta_qc[4]:.6f})*{I_new}² + ({beta_qc[5]:.6f})*{T_new}*{I_new}")
print(f"                       Oil = {pred_quad:.2f} gallons")

print(f"\n  Recommended prediction (best cleaned model): ", end="")
if best_type == "linear":
    print(f"{pred_lin:.2f} gallons (linear)")
else:
    print(f"{pred_quad:.2f} gallons (quadratic)")