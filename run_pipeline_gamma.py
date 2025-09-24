#!/usr/bin/env python3
"""
Pipeline para:
 - contar 'ok' vs 'review'
 - ajustar una gamma por galaxia (minimizando RMSE entre Vbar y combinación de componentes)
 - exportar CSVs y guardar plots individuales
Requiere: pandas, numpy, matplotlib, scipy, sklearn
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- CONFIG ---
ROOT = "."                    # carpeta con summary.csv y subcarpeta profiles/
PROFILES_DIR = os.path.join(ROOT, "profiles")
SUMMARY_NAMES = ["summary.csv", "analyze_structures.csv", "summary_checked.csv"]
OUT_DIR = os.path.join(ROOT, "gamma_pipeline_out")
os.makedirs(OUT_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- LOAD summary (prioridad) ---
summary = None
for n in SUMMARY_NAMES:
    p = os.path.join(ROOT, n)
    if os.path.exists(p):
        summary = pd.read_csv(p)
        print("Leyendo:", p)
        break
if summary is None:
    print("No se encontró summary.csv ni analyze_structures.csv en", ROOT)
    # intentar construir summary mínimo desde profiles
    summary = pd.DataFrame()

# --- Estadísticas de data_quality_flag (si existe) ---
if "data_quality_flag" in summary.columns:
    counts = summary["data_quality_flag"].value_counts(dropna=False)
    print("Cuenta data_quality_flag:\n", counts)
else:
    print("summary no tiene columna data_quality_flag; se listarán perfiles en profiles/")

# --- localizar perfiles ---
profile_files = sorted(glob.glob(os.path.join(PROFILES_DIR, "*components_profile.csv")))
if len(profile_files)==0:
    print("No se encontraron archivos '*components_profile.csv' en", PROFILES_DIR)
else:
    print(f"Encontrados {len(profile_files)} profiles.")

# --- función auxiliar: ajustar gamma por galaxia ---
def fit_gamma(df_profile, vbar_col="Vbar_kms", components_cols=None):
    # df_profile: contém columnas r_kpc, Vgas_kms, Vdisk_kms, Vbul_kms, Vbar_kms, ...
    # model: Vbar ~ sqrt(Vgas^2 + Vdisk^2 + (gamma * Vbul)^2)  (variant A)
    # alternative: Vbar ~ gamma*Vbul + Vdisk + Vgas  (variant B linear)
    r = df_profile.get("r_kpc").values if "r_kpc" in df_profile.columns else np.arange(df_profile.shape[0])
    vbar = df_profile[vbar_col].values
    gas = df_profile.get("Vgas_kms", 0.0).values
    disk = df_profile.get("Vdisk_kms", 0.0).values
    bul  = df_profile.get("Vbul_kms", 0.0).values

    # EPS para evitar ceros
    eps = 1e-8

    # Variant A: gamma acts on bulge in quadrature: model = sqrt(gas^2 + disk^2 + (gamma*bul)^2)
    def residA(g):
        model = np.sqrt(gas**2 + disk**2 + (g[0]*bul)**2 + eps)
        return (model - vbar)

    # initial guess
    try:
        g0 = [1.0]
        resA = least_squares(residA, g0, bounds=(0.0, 10.0))
        gammaA = float(resA.x[0])
        rmseA = np.sqrt(np.mean(resA.fun**2))
    except Exception as e:
        gammaA = np.nan
        rmseA = np.nan

    # Variant B: linear combination: model = gas + disk + gamma*bul
    def residB(g):
        model = gas + disk + g[0]*bul
        return (model - vbar)

    try:
        resB = least_squares(residB, [1.0], bounds=(-10, 10))
        gammaB = float(resB.x[0])
        rmseB = np.sqrt(np.mean(resB.fun**2))
    except:
        gammaB = np.nan
        rmseB = np.nan

    return dict(gamma_quad=gammaA, rmse_quad=rmseA, gamma_lin=gammaB, rmse_lin=rmseB)

# --- loop sobre perfiles, ajuste gamma y guardado de plots ---
rows = []
for pf in profile_files:
    name = os.path.basename(pf).replace("_components_profile.csv","")
    try:
        dfp = pd.read_csv(pf)
    except Exception as e:
        print("No pude leer", pf, ":", e)
        continue
    # verificar columnas
    for col in ("Vbar_kms","Vgas_kms","Vdisk_kms","Vbul_kms"):
        if col not in dfp.columns:
            # intentar nombres alternativos
            pass
    res = fit_gamma(dfp, vbar_col="Vbar_kms")
    # guardar plot comparativo
    try:
        plt.figure(figsize=(6,4))
        plt.plot(dfp.get("r_kpc", np.arange(len(dfp))), dfp["Vbar_kms"], label="Vbar (observado)")
        plt.plot(dfp.get("r_kpc", np.arange(len(dfp))), np.sqrt(dfp.get("Vgas_kms",0)**2 + dfp.get("Vdisk_kms",0)**2 + (res["gamma_quad"]*dfp.get("Vbul_kms",0))**2), label=f"Modelo quad γ={res['gamma_quad']:.3g}")
        plt.plot(dfp.get("r_kpc", np.arange(len(dfp))), dfp.get("Vdisk_kms",0), linestyle="--", label="Vdisk")
        plt.plot(dfp.get("r_kpc", np.arange(len(dfp))), dfp.get("Vgas_kms",0), linestyle=":", label="Vgas")
        if "Vbul_kms" in dfp.columns:
            plt.plot(dfp.get("r_kpc", np.arange(len(dfp))), res["gamma_lin"]*dfp.get("Vbul_kms",0), linestyle="-.", label=f"gamma_lin*Vbul ({res['gamma_lin']:.3g})")
        plt.xlabel("r [kpc]"); plt.ylabel("V [km/s]")
        plt.title(name)
        plt.legend(fontsize="small")
        plt.tight_layout()
        png = os.path.join(PLOTS_DIR, f"{name}_curve.png")
        plt.savefig(png, dpi=150)
        plt.close()
    except Exception as e:
        print("Error plot", name, e)

    row = {"galaxy": name,
           "file": pf,
           "gamma_quad": res["gamma_quad"], "rmse_quad": res["rmse_quad"],
           "gamma_lin": res["gamma_lin"], "rmse_lin": res["rmse_lin"],
           "n_points": dfp.shape[0]}
    rows.append(row)

df_gamma = pd.DataFrame(rows)
df_gamma.to_csv(os.path.join(OUT_DIR, "gamma_per_galaxy.csv"), index=False)
print("Guardado gamma_per_galaxy.csv con", len(df_gamma), "entradas")

# --- combinar con summary (si existe) y exportar resumen final ---
if not summary.empty:
    # normalizar nombre de galaxia en summary para juntar (si necesario)
    s = summary.copy()
    s['galaxy_key'] = s['galaxy'].astype(str).str.replace(".csv","",regex=False)
    df_gamma['galaxy_key'] = df_gamma['galaxy']
    merged = s.merge(df_gamma.drop(columns=["file"]), on="galaxy_key", how="left")
    out_summary = os.path.join(OUT_DIR, "summary_checked.csv")
    merged.to_csv(out_summary, index=False)
    print("Guardado", out_summary)
else:
    print("No se fusionó con summary (no disponible).")

print("Plots en:", PLOTS_DIR)
print("FIN")
