import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Carpetas
profiles_dir = "profiles/"
plots_dir = "plots/"
os.makedirs(plots_dir, exist_ok=True)

summary = []

for file in glob.glob(os.path.join(profiles_dir, "*_components_profile.csv")):
    galaxy = os.path.basename(file).replace("_components_profile.csv", "")
    try:
        df = pd.read_csv(file)

        # --- Métricas básicas ---
        V_flat = df["Vbar_kms"].tail(5).mean()   # velocidad en la parte plana
        M_total = df["Mbar_Msun"].iloc[-1]       # masa total acumulada

        # Composición
        Mgas = df["Mgas_Msun"].iloc[-1]
        Mdisk = df["Mdisk_Msun"].iloc[-1]
        Mbul = df["Mbul_Msun"].iloc[-1]

        bulge_to_total = Mbul / M_total if M_total > 0 else 0
        disk_vs_gas_ratio = Mdisk / Mgas if Mgas > 0 else float("inf")

        summary.append({
            "galaxy": galaxy,
            "V_flat_kms": V_flat,
            "M_total_Msun": M_total,
            "Mgas_Msun": Mgas,
            "Mdisk_Msun": Mdisk,
            "Mbul_Msun": Mbul,
            "bulge_to_total": bulge_to_total,
            "disk_vs_gas_ratio": disk_vs_gas_ratio
        })

        # --- Gráficos ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

        # Curva de rotación
        ax1.plot(df["r_kpc"], df["Vbar_kms"], "k-", label="Total")
        ax1.plot(df["r_kpc"], df["Vgas_kms"], "b--", label="Gas")
        ax1.plot(df["r_kpc"], df["Vdisk_kms"], "r--", label="Disco")
        ax1.plot(df["r_kpc"], df["Vbul_kms"], "g--", label="Bulbo")
        ax1.set_xlabel("Radio [kpc]")
        ax1.set_ylabel("v [km/s]")
        ax1.set_title(f"{galaxy} - Curva de rotación")
        ax1.legend()

        # Masa acumulada
        ax2.plot(df["r_kpc"], df["Mbar_Msun"], "k-", label="Total")
        ax2.plot(df["r_kpc"], df["Mgas_Msun"], "b--", label="Gas")
        ax2.plot(df["r_kpc"], df["Mdisk_Msun"], "r--", label="Disco")
        ax2.plot(df["r_kpc"], df["Mbul_Msun"], "g--", label="Bulbo")
        ax2.set_xlabel("Radio [kpc]")
        ax2.set_ylabel("M [M☉]")
        ax2.set_title("Masa acumulada")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{galaxy}.png"))
        plt.close()

    except Exception as e:
        print(f"No pude procesar {galaxy}: {e}")

# Guardar el resumen general
summary_df = pd.DataFrame(summary)
summary_df.to_csv("summary.csv", index=False)

print(f"Procesadas {len(summary)} galaxias. Resultados en summary.csv y carpeta plots/")
