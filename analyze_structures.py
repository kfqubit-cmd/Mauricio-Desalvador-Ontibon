#!/usr/bin/env python3
"""
analyze_sparc_structures.py

Procesa archivos *_rotmod.dat en la carpeta 'sparc_database' y produce:
 - output/summaries/summary_structures.csv
 - output/profiles/{galaxy}_components_profile.csv

Versión: 1.0
"""

import os
import glob
import numpy as np
import pandas as pd

# -----------------------
# Constantes físicas
# -----------------------
G_SI = 6.67430e-11              # m^3 kg^-1 s^-2
kpc_to_m = 3.085677581491367e19
kms_to_ms = 1000.0
Msun_kg = 1.98847e30
c = 299792458.0
tiny = 1e-60

# -----------------------
# Parámetros de configuración
# -----------------------
SPARC_DIR = "sparc_database"   # carpeta de entrada (ajusta si es necesario)
OUT_DIR = "output"
PROFILES_DIR = os.path.join(OUT_DIR, "profiles")
SUMMARIES_DIR = os.path.join(OUT_DIR, "summaries")

BULGE_CONTRIB_THRESHOLD = 0.10  # consideramos bulbo significativo si Vbul^2/Vbar^2 >= 0.10
MIN_POINTS_GOOD = 6             # mínimo de puntos para considerarla 'ok'

# -----------------------
# Funciones auxiliares
# -----------------------
def read_rotmod(path):
    """
    Lee con tolerancia un archivo *_rotmod.dat.
    Devuelve DataFrame con columnas: r_kpc, Vobs_kms, eVobs_kms (si existe),
    Vgas_kms, Vdisk_kms, Vbul_kms (las que no existan se crean como 0).
    """
    # intentos de lectura flexibles
    try:
        df = pd.read_csv(path, comment='#', delim_whitespace=True, header=None)
    except Exception:
        try:
            df = pd.read_csv(path, comment='#', sep=',', header=None)
        except Exception:
            data = np.loadtxt(path, comments=('#','%'))
            df = pd.DataFrame(data)

    ncol = df.shape[1]
    if ncol >= 6:
        colnames = ['r_kpc','Vobs_kms','eVobs_kms','Vgas_kms','Vdisk_kms','Vbul_kms'] + [f'c{i}' for i in range(6,ncol)]
    elif ncol == 5:
        colnames = ['r_kpc','Vobs_kms','eVobs_kms','Vgas_kms','Vdisk_kms']
    elif ncol == 4:
        colnames = ['r_kpc','Vobs_kms','eVobs_kms','Vgas_kms']
    elif ncol == 3:
        colnames = ['r_kpc','Vobs_kms','eVobs_kms']
    elif ncol == 2:
        colnames = ['r_kpc','Vobs_kms']
    else:
        colnames = [f'c{i}' for i in range(ncol)]

    df.columns = colnames
    df = df.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    df = df.dropna(how='all')
    # requerir al menos r y Vobs
    if not {'r_kpc','Vobs_kms'}.issubset(df.columns):
        return None
    df = df.dropna(subset=['r_kpc','Vobs_kms']).reset_index(drop=True)

    # asegurar columnas presentes
    for col in ['Vgas_kms','Vdisk_kms','Vbul_kms','eVobs_kms']:
        if col not in df.columns:
            df[col] = 0.0

    # Vbar (componente bariónica aprox)
    df['Vbar_kms'] = np.sqrt(np.clip(df['Vgas_kms']**2 + df['Vdisk_kms']**2 + df['Vbul_kms']**2, 0, None))

    return df

def mass_enclosed_from_v_component(v_kms, r_kpc):
    """
    M(<r) = v^2 r / G  (approx).
    Devuelve array en Msun (misma longitud que v_kms/r_kpc).
    """
    v = np.asarray(v_kms) * kms_to_ms
    r = np.asarray(r_kpc) * kpc_to_m
    M_kg = v**2 * r / (G_SI + tiny)
    return M_kg / Msun_kg

def compute_shear_ln(v_kms, r_kpc):
    """
    Estimación de torsión: slope = d ln v / d ln r (ajuste lineal en log-log).
    """
    v = np.asarray(v_kms)
    r = np.asarray(r_kpc)
    mask = (v > 0) & (r > 0)
    if np.sum(mask) < 3:
        return np.nan
    ln_v = np.log(v[mask])
    ln_r = np.log(r[mask])
    A = np.vstack([ln_r, np.ones_like(ln_r)]).T
    slope, intercept = np.linalg.lstsq(A, ln_v, rcond=None)[0]
    return float(slope)

# -----------------------
# Preparar salidas
# -----------------------
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(SUMMARIES_DIR, exist_ok=True)

# detectar archivos *_rotmod.dat en sparc_database
pattern = os.path.join(SPARC_DIR, "*_rotmod.dat")
rot_files = sorted(glob.glob(pattern))
if len(rot_files) == 0:
    print(f"No se encontraron archivos con patrón {pattern}. Asegura que la carpeta existe y contiene *_rotmod.dat")
    raise SystemExit(1)

summary_list = []

# -----------------------
# Procesar cada archivo
# -----------------------
for path in rot_files:
    try:
        df = read_rotmod(path)
        if df is None or len(df) == 0:
            print(f"Skipping {path}: formato inesperado o vacío.")
            continue

        gal_name = os.path.basename(path).replace('.dat','')
        Npoints = len(df)
        R_max_kpc = float(df['r_kpc'].max())

        # flags y bulbo
        has_bulge = (np.nanmax(np.abs(df['Vbul_kms'])) > 1e-3)
        gas_negative_flag = np.any(df['Vgas_kms'] < 0)
        ratio_bulge = np.where(df['Vbar_kms']>0, (df['Vbul_kms']**2) / (df['Vbar_kms']**2 + tiny), 0.0)
        bulge_indices = np.where(ratio_bulge >= BULGE_CONTRIB_THRESHOLD)[0]
        bulge_size_kpc = float(df['r_kpc'].iloc[bulge_indices[0]]) if len(bulge_indices)>0 else 0.0

        # masas por componente (perfil)
        Mgas = mass_enclosed_from_v_component(df['Vgas_kms'].values, df['r_kpc'].values)
        Mdisk = mass_enclosed_from_v_component(df['Vdisk_kms'].values, df['r_kpc'].values)
        Mbul = mass_enclosed_from_v_component(df['Vbul_kms'].values, df['r_kpc'].values)
        Mbar = mass_enclosed_from_v_component(df['Vbar_kms'].values, df['r_kpc'].values)

        # valores al Rmax
        Mgas_R = float(Mgas[-1])
        Mdisk_R = float(Mdisk[-1])
        Mbul_R = float(Mbul[-1])
        Mbar_R = float(Mbar[-1])

        # relaciones y razones
        B_T = (Mbul_R / (Mbar_R + tiny)) if Mbar_R>0 else np.nan
        disk_vs_gas_ratio = (np.sum(df['Vdisk_kms']**2) + tiny) / (np.sum(df['Vgas_kms']**2) + tiny)

        # potencial newtoniano adimensional en Rmax
        Mtot_kg = Mbar_R * Msun_kg
        Rm = R_max_kpc * kpc_to_m
        Phi_N = (G_SI * Mtot_kg) / (Rm * c**2 + tiny)

        # shear/torsion y v_flat
        shear_mean = compute_shear_ln(df['Vobs_kms'].values, df['r_kpc'].values)
        k = min(5, Npoints)
        v_flat = float(np.mean(df['Vobs_kms'].values[-k:]))

        data_quality_flag = 'ok' if Npoints >= MIN_POINTS_GOOD and not gas_negative_flag else 'review'

        # guardar perfil por componente
        prof_df = pd.DataFrame({
            'r_kpc': df['r_kpc'].values,
            'Vgas_kms': df['Vgas_kms'].values,
            'Vdisk_kms': df['Vdisk_kms'].values,
            'Vbul_kms': df['Vbul_kms'].values,
            'Vbar_kms': df['Vbar_kms'].values,
            'Mgas_Msun': Mgas,
            'Mdisk_Msun': Mdisk,
            'Mbul_Msun': Mbul,
            'Mbar_Msun': Mbar
        })
        prof_df.to_csv(os.path.join(PROFILES_DIR, f"{gal_name}_components_profile.csv"), index=False)

        # agregar al resumen
        summary_list.append({
            'galaxy': gal_name,
            'Npoints': int(Npoints),
            'data_quality_flag': data_quality_flag,
            'has_bulge': bool(has_bulge),
            'bulge_size_kpc': float(bulge_size_kpc),
            'gas_negative_flag': bool(gas_negative_flag),
            'R_max_kpc': float(R_max_kpc),
            'Mtot_Msun': float(Mbar_R),
            'Mgas_Msun': float(Mgas_R),
            'Mdisk_Msun': float(Mdisk_R),
            'Mbul_Msun': float(Mbul_R),
            'bulge_to_total_at_Rmax': float(B_T),
            'disk_vs_gas_ratio': float(disk_vs_gas_ratio),
            'Phi_N_Rmax': float(Phi_N),
            'shear_mean': float(shear_mean) if not np.isnan(shear_mean) else None,
            'v_flat_kms': float(v_flat)
        })

    except Exception as e:
        print(f"Error procesando {path}: {e}")
        continue

# -----------------------
# Guardar resumen global
# -----------------------
summary_df = pd.DataFrame(summary_list)
summary_path = os.path.join(SUMMARIES_DIR, "summary_structures.csv")
summary_df.to_csv(summary_path, index=False)

print("Procesamiento completado.")
print(f"Archivos procesados: {len(summary_df)}")
print(f"Resumen guardado en: {summary_path}")
print(f"Perfiles guardados en: {PROFILES_DIR}")
