"""Módulo de visualizações padronizadas em alta resolução (DPI=180) para o portfólio PETR4.

Gera os 9 painéis analíticos salvos em `reports/figures/`:
1. `01_historico_choques_volatilidade.png`
2. `02_correlacao_espuria_vs_retornos.png`
3. `03_capm_beta_rolante_correlacoes.png`
4. `04_teste_chow_quebras_estruturais.png`
5. `05_estacionariedade_acf_pacf.png`
6. `06_arima_diagnostico_residuos.png`
7. `07_var_irf_fevd.png`
8. `08_benchmark_walk_forward_leakage.png`
9. `09_cenarios_macroeconomicos.png`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

from src.data_pipeline import HISTORICAL_EVENTS, assign_regime

# Estilo global padronizado
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfd",
        "axes.edgecolor": "#d0d7de",
        "axes.grid": True,
        "grid.color": "#e6ebf1",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.9,
    }
)

SOURCE_FOOTNOTE = (
    "Fonte: Yahoo Finance (PETR4.SA, ^BVSP, BZ=F), IPEADATA (Brent 2000–2007) e "
    "Banco Central do Brasil (PTAX, Selic) | Elaboração própria."
)


def _save_fig(fig: plt.Figure, output_path: str | Path, close_fig: bool = False) -> plt.Figure:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    return fig


def _annotate_events(ax: plt.Axes, y_frac: float = 0.96, add_labels: bool = True) -> None:
    """Desenha linhas verticais dos 5 marcos históricos da Petrobras."""
    ymin, ymax = ax.get_ylim()
    y_pos = ymin + y_frac * (ymax - ymin)
    for idx, (dt_str, (label, color)) in enumerate(HISTORICAL_EVENTS.items()):
        ts = pd.Timestamp(dt_str)
        ax.axvline(ts, color=color, linestyle="--", linewidth=1.2, alpha=0.85)
        if add_labels:
            offset_frac = y_frac - (idx % 2) * 0.09
            y_txt = ymin + offset_frac * (ymax - ymin)
            ax.text(
                ts + pd.Timedelta(days=25),
                y_txt,
                label,
                color=color,
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
            )


def plot_historical_shocks_and_volatility(
    df_daily: pd.DataFrame,
    output_path: str | Path = "reports/figures/01_historico_choques_volatilidade.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Painel triplo: Preço PETR4 vs. Brent, Retorno Total Acumulado (Base 100) e Volatilidade."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.1, 0.9]})
    fig.suptitle(
        "PETR4 (2000–2026): Trajetória Histórica, Choques Estruturais e Regimes de Volatilidade",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Painel A: Preço de fechamento vs Brent USD
    ax1 = axes[0]
    l1 = ax1.plot(df_daily.index, df_daily["petr4_close"], color="#0b5394", linewidth=1.3, label="PETR4 Fechamento Tela (R$, esq.)")
    ax1.set_ylabel("PETR4 Preço (R$)", color="#0b5394", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0b5394")
    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    l2 = ax1_r.plot(df_daily.index, df_daily["brent_usd"], color="#d97706", linewidth=1.0, alpha=0.75, label="Petróleo Brent (US$/barril, dir.)")
    ax1_r.set_ylabel("Brent (US$/barril)", color="#d97706", fontweight="bold")
    ax1_r.tick_params(axis="y", labelcolor="#d97706")
    ax1.set_title("A. Preço Nominal Ajustado a Splits (BRL) vs. Petróleo Brent (USD/barril) com 5 Marcos Históricos")
    _annotate_events(ax1, y_frac=0.92, add_labels=True)
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Painel B: Retorno Total Acumulado Base 100 (escala log)
    ax2 = axes[1]
    base_petr = 100.0 * df_daily["petr4_adj_close"] / df_daily["petr4_adj_close"].iloc[0]
    base_ibov = 100.0 * df_daily["ibov_close"] / df_daily["ibov_close"].iloc[0]
    base_selic = 100.0 * np.exp(df_daily["selic_daily_log"].cumsum())

    ax2.plot(df_daily.index, base_petr, color="#1b7837", linewidth=1.5, label=f"PETR4 Retorno Total (Adj Close): {base_petr.iloc[-1]:,.0f} pts")
    ax2.plot(df_daily.index, base_ibov, color="#4d4d4d", linewidth=1.2, label=f"Ibovespa: {base_ibov.iloc[-1]:,.0f} pts")
    ax2.plot(df_daily.index, base_selic, color="#8e24aa", linewidth=1.2, linestyle="-.", label=f"Taxa Selic Acumulada: {base_selic.iloc[-1]:,.0f} pts")
    ax2.set_yscale("log")
    ax2.set_ylabel("Índice Base 100 (Escala Log)")
    ax2.set_title("B. Retorno Total Acumulado (Reinvestimento de Dividendos): PETR4 vs. Ibovespa vs. Selic (Jan/2000 = 100)")
    _annotate_events(ax2, add_labels=False)
    ax2.legend(loc="upper left")

    # Painel C: Volatilidade Anualizada Móvel (252 pregões)
    ax3 = axes[2]
    vol_pct = df_daily["vol_252d"] * 100.0
    ax3.plot(df_daily.index, vol_pct, color="#b22222", linewidth=1.3, label="Volatilidade Anualizada (Janela 252 pregões)")
    ax3.fill_between(df_daily.index, 0, vol_pct, color="#b22222", alpha=0.15)
    ax3.axhline(vol_pct.mean(), color="#333333", linestyle=":", linewidth=1.1, label=f"Média Histórica ({vol_pct.mean():.1f}% a.a.)")
    ax3.set_ylabel("Volatilidade (% a.a.)")
    ax3.set_title("C. Regimes de Volatilidade Realizada Anualizada — Picos na Lava Jato (2015–16) e COVID-19 (2020)")
    _annotate_events(ax3, add_labels=False)
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_spurious_vs_returns(
    df_daily: pd.DataFrame,
    output_path: str | Path = "reports/figures/02_correlacao_espuria_vs_retornos.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Demonstra correlação espúria em níveis vs. retornos e passeio aleatório vs. ruído branco."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle(
        "Fundamentos Estocásticos: Correlação Espúria (Níveis vs. Retornos) e Passeio Aleatório",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # A: Correlação em níveis (risco de regressão espúria)
    r_lvl = df_daily["petr4_adj_close"].corr(df_daily["ibov_close"])
    axes[0].scatter(df_daily["ibov_close"] / 1000.0, df_daily["petr4_adj_close"], alpha=0.25, s=12, color="#c62828")
    axes[0].set_title(f"A. Níveis (Não-Estacionários)\nCorrelação Bruta r = {r_lvl:.2f} (Inflada por Tendência)")
    axes[0].set_xlabel("Ibovespa (mil pontos)")
    axes[0].set_ylabel("PETR4 Adj Close (R$)")

    # B: Correlação nos log-retornos diários
    r_ret = df_daily["ret_petr4"].corr(df_daily["ret_ibov"])
    axes[1].scatter(df_daily["ret_ibov"] * 100, df_daily["ret_petr4"] * 100, alpha=0.25, s=12, color="#1565c0")
    m_slope, m_int = np.polyfit(df_daily["ret_ibov"] * 100, df_daily["ret_petr4"] * 100, 1)
    xs = np.linspace(df_daily["ret_ibov"].min() * 100, df_daily["ret_ibov"].max() * 100, 100)
    axes[1].plot(xs, m_slope * xs + m_int, color="#0d47a1", linewidth=1.8, label=f"Inclinação β={m_slope:.2f}")
    axes[1].set_title(f"B. Log-Retornos Diários (Estacionários)\nCorrelação Real r = {r_ret:.2f}")
    axes[1].set_xlabel("Retorno Diário Ibovespa (%)")
    axes[1].set_ylabel("Retorno Diário PETR4 (%)")
    axes[1].legend(loc="upper left")

    # C: ACF do Log-Preço Real vs Passeio Aleatório Simulado vs Ruído Branco
    rng = np.random.default_rng(42)
    mu = float(df_daily["ret_petr4"].mean())
    sigma = float(df_daily["ret_petr4"].std())
    wn = rng.normal(mu, sigma, size=len(df_daily))
    rw = np.cumsum(wn)

    lags = np.arange(1, 21)
    acf_real_lvl = acf(df_daily["log_petr4_adj"], nlags=20)[1:]
    acf_rw = acf(rw, nlags=20)[1:]
    acf_real_ret = acf(df_daily["ret_petr4"], nlags=20)[1:]
    acf_wn = acf(wn, nlags=20)[1:]

    axes[2].plot(lags, acf_real_lvl, "o-", color="#b22222", label="Log-Preço PETR4 (Nível)")
    axes[2].plot(lags, acf_rw, "--", color="#fb8c00", linewidth=2, label="Passeio Aleatório Simulado")
    axes[2].plot(lags, acf_real_ret, "s-", color="#1565c0", label="Log-Retorno PETR4 (1ª Diff)")
    axes[2].plot(lags, acf_wn, ":", color="#2e7d32", linewidth=2, label="Ruído Branco Simulado")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("C. Função de Autocorrelação (ACF)\nPETR4 em Nível ≈ Passeio Aleatório | Retorno ≈ Ruído Branco")
    axes[2].set_xlabel("Defasagem (pregões)")
    axes[2].set_ylabel("Autocorrelação")
    axes[2].legend(fontsize=8, loc="center right")

    fig.text(0.01, -0.03, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout()
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_capm_and_rolling_beta(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    capm_results: dict[str, Any],
    output_path: str | Path = "reports/figures/03_capm_beta_rolante_correlacoes.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Painel CAPM por regime, Beta Rolante (252d) e Correlações Móveis com Ibovespa e Brent."""
    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.35])
    fig.suptitle(
        "Risco Sistemático & CAPM Dinâmico: Sensibilidade de PETR4 ao Mercado e ao Petróleo",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # A: Dispersão CAPM Mensal por Regime
    ax_sc = fig.add_subplot(gs[:, 0])
    reg_m = assign_regime(df_monthly.index)
    palette = sns.color_palette("Set1", n_colors=len(reg_m.unique()))
    for color, regime in zip(palette, reg_m.dropna().unique()):
        mask = reg_m == regime
        ax_sc.scatter(
            df_monthly.loc[mask, "excess_ibov"] * 100,
            df_monthly.loc[mask, "excess_petr4"] * 100,
            alpha=0.72,
            s=36,
            color=color,
            label=regime,
        )

    capm_m = capm_results["capm_monthly"]
    alpha_m = float(capm_m.params["const"]) * 100
    beta_m = float(capm_m.params["excess_ibov"])
    x_grid = np.linspace(df_monthly["excess_ibov"].min() * 100, df_monthly["excess_ibov"].max() * 100, 100)
    ax_sc.plot(
        x_grid,
        alpha_m + beta_m * x_grid,
        color="black",
        linewidth=2.2,
        label=f"CAPM Global: β = {beta_m:.2f} | α = {alpha_m:.2f}%/mês | R² = {capm_m.rsquared:.2f}",
    )
    ax_sc.axhline(0, color="#888", linewidth=0.8)
    ax_sc.axvline(0, color="#888", linewidth=0.8)
    ax_sc.set_title("A. Regressão CAPM Mensal por Regime (Excesso sobre Selic)")
    ax_sc.set_xlabel("Prêmio de Mercado Ibovespa - Selic (% mês)")
    ax_sc.set_ylabel("Excesso de Retorno PETR4 - Selic (% mês)")
    ax_sc.legend(loc="upper left", fontsize=8.5)

    # B: Beta Rolante 252 dias
    ax_b = fig.add_subplot(gs[0, 1])
    rb = capm_results["rolling_beta_252d"].dropna()
    ax_b.plot(rb.index, rb.values, color="#0b5394", linewidth=1.5, label="Beta Rolante (252 pregões)")
    ax_b.axhline(1.0, color="#333333", linestyle=":", linewidth=1.2, label="Beta Neutro = 1.0")
    ax_b.fill_between(rb.index, 1.0, rb.values, where=(rb.values >= 1.0), color="#c62828", alpha=0.14, label="Risco Amplificado (β > 1)")
    ax_b.fill_between(rb.index, 1.0, rb.values, where=(rb.values < 1.0), color="#2e7d32", alpha=0.14, label="Menor Sensibilidade (β < 1)")
    ax_b.set_title("B. Evolução Dinâmica do Beta CAPM (Janela Móvel de 252 Pregões)")
    ax_b.set_ylabel("Beta (β)")
    _annotate_events(ax_b, y_frac=0.92, add_labels=False)
    ax_b.legend(loc="upper right", fontsize=8.5, ncol=2)

    # C: Correlações Móveis (Ibov vs Brent)
    ax_c = fig.add_subplot(gs[1, 1], sharex=ax_b)
    ci = capm_results["rolling_corr_ibov_252d"].dropna()
    cb = capm_results["rolling_corr_brent_252d"].dropna()
    ax_c.plot(ci.index, ci.values, color="#1565c0", linewidth=1.4, label=f"Correlação PETR4 vs. Ibovespa (Média: {ci.mean():.2f})")
    ax_c.plot(cb.index, cb.values, color="#d97706", linewidth=1.4, label=f"Correlação PETR4 vs. Brent USD (Média: {cb.mean():.2f})")
    ax_c.axhline(0, color="#666666", linewidth=0.8)
    ax_c.set_title("C. Correlação Móvel (252 Pregões): Desacoplamento Relativo Pós-2023")
    ax_c.set_ylabel("Correlação de Pearson")
    ax_c.set_ylim(-0.3, 1.0)
    _annotate_events(ax_c, y_frac=0.92, add_labels=False)
    ax_c.legend(loc="lower left", fontsize=8.5)
    ax_c.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_chow_structural_breaks(
    chow_table: pd.DataFrame,
    rolling_curves: pd.DataFrame,
    capm_results: dict[str, Any],
    output_path: str | Path = "reports/figures/04_teste_chow_quebras_estruturais.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Curvas da estatística F de Chow (local e global) e mudança nos Betas por regime."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [1.0, 1.0, 1.05]})
    fig.suptitle(
        "Teste de Quebra Estrutural de Chow: Impacto dos 5 Marcos Históricos na Dinâmica de PETR4",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # A: Curva Chow Local (Trajetória Log-Preço ±252d)
    ax1 = axes[0]
    c_trend = rolling_curves["f_stat_trend_local"].dropna()
    crit_trend = float(rolling_curves["f_crit_1pct_trend"].dropna().iloc[0])
    ax1.plot(c_trend.index, c_trend.values, color="#1565c0", linewidth=1.4, label="Estatística F de Chow (Trajetória Local ±252d)")
    ax1.axhline(crit_trend, color="#b22222", linestyle=":", linewidth=1.3, label=f"Valor Crítico 1% (F = {crit_trend:.2f})")
    ax1.set_ylim(0, c_trend.max() * 1.25)
    _annotate_events(ax1, y_frac=0.76, add_labels=True)
    ax1.set_title("A. Quebras na Trajetória de Preço (Log-Preço ~ Tendência Local): Picos nos Choques Históricos")
    ax1.set_ylabel("Estatística F")
    ax1.legend(loc="upper left", fontsize=9)

    # B: Curva Chow Global Multifatorial (Quandt-Andrews)
    ax2 = axes[1]
    c_multi = rolling_curves["f_stat_multifactor_global"].dropna()
    crit_multi = float(rolling_curves["f_crit_1pct_multifactor"].dropna().iloc[0])
    ax2.plot(c_multi.index, c_multi.values, color="#6a1b9a", linewidth=1.6, label="Estatística F de Chow (Multifatorial CAPM + Brent + Câmbio)")
    ax2.axhline(crit_multi, color="#b22222", linestyle=":", linewidth=1.3, label=f"Valor Crítico 1% (F = {crit_multi:.2f})")
    ax2.fill_between(c_multi.index, crit_multi, c_multi.values, where=(c_multi.values > crit_multi), color="#6a1b9a", alpha=0.14)
    _annotate_events(ax2, y_frac=0.88, add_labels=False)
    ax2.set_title("B. Estabilidade Estrutural do Modelo Multifatorial (Varredura Global 2001–2025)")
    ax2.set_ylabel("Estatística F")
    ax2.legend(loc="upper right")

    # C: Betas por Regime Histórico
    ax3 = axes[2]
    reg_df = capm_results["regime_table"]
    x = np.arange(len(reg_df))
    width = 0.28
    b1 = ax3.bar(x - width, reg_df["Beta_Multi_Ibov"], width, label="β Mercado (Ibovespa)", color="#0b5394")
    b2 = ax3.bar(x, reg_df["Beta_Multi_Brent"], width, label="β Petróleo (Brent USD)", color="#d97706")
    b3 = ax3.bar(x + width, reg_df["Beta_Multi_FX"], width, label="β Câmbio (USD/BRL)", color="#2e7d32")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(reg_df["Regime"], fontsize=9.5)
    ax3.set_ylabel("Elasticidade / Beta Estimado")
    ax3.set_title("C. Mudança de Regime Estrutural nos Coeficientes do Modelo Multifatorial")
    ax3.legend(loc="upper right")

    ymin_b, ymax_b = ax3.get_ylim()
    ax3.set_ylim(ymin_b - 0.18, ymax_b + 0.24)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.04 if h >= 0 else -0.04),
                f"{h:.2f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
            )

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_stationarity_acf_pacf(
    df_monthly: pd.DataFrame,
    stationarity_df: pd.DataFrame,
    output_path: str | Path = "reports/figures/05_estacionariedade_acf_pacf.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Painel 2x3 comparando Log-Nível (Não-Estacionário) vs. 1ª Diferença (Log-Retorno Estacionário)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(
        "Metodologia Box-Jenkins: Estacionariedade (ADF & KPSS) e Correlogramas ACF / PACF (Base Mensal)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    row_lvl = stationarity_df.loc[stationarity_df["Série"] == "Log PETR4 Adj (Mensal)"].iloc[0]
    row_ret = stationarity_df.loc[stationarity_df["Série"] == "Log-Retorno PETR4 (Mensal)"].iloc[0]

    # Linha 1: Log-Nível
    axes[0, 0].plot(df_monthly.index, df_monthly["log_petr4_adj"], color="#c62828", linewidth=1.4)
    axes[0, 0].set_title(
        f"Log-Preço PETR4 (Nível I(1))\nADF={row_lvl['ADF_Stat']:.2f} (p={row_lvl['ADF_p_valor']:.2f}) | KPSS={row_lvl['KPSS_Stat']:.2f} (p={row_lvl['KPSS_p_valor']:.2f})"
    )
    axes[0, 0].set_ylabel("ln(Adj Close)")
    plot_acf(df_monthly["log_petr4_adj"], lags=24, zero=False, ax=axes[0, 1], color="#c62828", vlines_kwargs={"colors": "#c62828"})
    axes[0, 1].set_title("ACF — Log-Preço (Decaimento Lento = Raiz Unitária)")
    plot_pacf(df_monthly["log_petr4_adj"], lags=24, zero=False, ax=axes[0, 2], method="ywm", color="#c62828", vlines_kwargs={"colors": "#c62828"})
    axes[0, 2].set_title("PACF — Log-Preço (Pico unitário no Lag 1)")

    # Linha 2: 1ª Diferença (Log-Retorno)
    axes[1, 0].plot(df_monthly.index, df_monthly["ret_petr4"] * 100, color="#1565c0", linewidth=1.0)
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_title(
        f"Log-Retorno Mensal PETR4 (Estacionário I(0))\nADF={row_ret['ADF_Stat']:.2f} (p={row_ret['ADF_p_valor']:.1e}) | KPSS={row_ret['KPSS_Stat']:.2f} (p={row_ret['KPSS_p_valor']:.2f})"
    )
    axes[1, 0].set_ylabel("Retorno Mensal (%)")
    plot_acf(df_monthly["ret_petr4"], lags=24, zero=False, ax=axes[1, 1], color="#1565c0", vlines_kwargs={"colors": "#1565c0"})
    axes[1, 1].set_title("ACF — Log-Retorno (Série Estacionária)")
    plot_pacf(df_monthly["ret_petr4"], lags=24, zero=False, ax=axes[1, 2], method="ywm", color="#1565c0", vlines_kwargs={"colors": "#1565c0"})
    axes[1, 2].set_title("PACF — Log-Retorno (Identificação de Ordem AR)")

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_arima_diagnostics(
    arima_res: dict[str, Any],
    output_path: str | Path = "reports/figures/06_arima_diagnostico_residuos.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Diagnóstico completo dos resíduos do melhor modelo ARIMA selecionado."""
    resid = arima_res["residuals"]
    std_resid = (resid - resid.mean()) / resid.std()
    order = arima_res["best_order"]
    lb_p10 = float(arima_res["ljung_box"].loc[10, "lb_pvalue"])
    lb_p20 = float(arima_res["ljung_box"].loc[20, "lb_pvalue"])
    jb_p = float(arima_res["jarque_bera_pval"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Diagnóstico Box-Jenkins — Melhor Modelo Selecionado: ARIMA{order}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1. Resíduos Padronizados
    axes[0, 0].plot(std_resid.index, std_resid.values, color="#0b5394", linewidth=1.0)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].axhline(2.5, color="#c62828", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0, 0].axhline(-2.5, color="#c62828", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[0, 0].set_title("A. Resíduos Padronizados ao Longo do Tempo")
    axes[0, 0].set_ylabel("Desvios-Padrão")

    # 2. Histograma + KDE vs Normal
    sns.histplot(std_resid.values, stat="density", bins=28, kde=True, color="#1565c0", ax=axes[0, 1], label="Densidade Empírica")
    xs = np.linspace(-4.5, 4.5, 200)
    axes[0, 1].plot(xs, stats.norm.pdf(xs), "r--", linewidth=1.8, label="Normal Padrão N(0,1)")
    axes[0, 1].set_title(f"B. Distribuição dos Resíduos (Jarque-Bera p = {jb_p:.2e})")
    axes[0, 1].legend()

    # 3. QQ-Plot
    (osm, osr), (slope, intercept, _) = stats.probplot(std_resid.values, dist="norm")
    axes[1, 0].scatter(osm, osr, s=18, alpha=0.7, color="#0b5394")
    axes[1, 0].plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.6)
    axes[1, 0].set_title("C. Gráfico Q-Q (Caudas Pesadas Típicas de Retornos Financeiros)")
    axes[1, 0].set_xlabel("Quantis Teóricos N(0,1)")
    axes[1, 0].set_ylabel("Quantis Amostrais")

    # 4. Correlograma dos resíduos
    plot_acf(resid, lags=20, zero=False, ax=axes[1, 1], color="#2e7d32", vlines_kwargs={"colors": "#2e7d32"})
    axes[1, 1].set_title(f"D. Correlograma Residual (Ljung-Box p[lag10]={lb_p10:.2f}, p[lag20]={lb_p20:.2f})")

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_var_irf_fevd(
    var_res: dict[str, Any],
    output_path: str | Path = "reports/figures/07_var_irf_fevd.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Funções Impulso-Resposta (IRF) sobre PETR4 e Decomposição da Variância (FEVD)."""
    cols = var_res["cols"]
    irf = var_res["irf"]
    fevd = var_res["fevd"]
    target_idx = cols.index("ret_petr4")

    orth_irfs = irf.orth_irfs  # (steps+1, k, k): [h, response, impulse]
    stderr = irf.stderr(orth=True)
    horizons = np.arange(orth_irfs.shape[0])

    pretty_names = {
        "ret_brent_usd": "Choque no Petróleo Brent (USD)",
        "ret_usd_brl": "Choque no Câmbio (USD/BRL)",
        "ret_ibov": "Choque no Ibovespa",
        "ret_petr4": "Choque Idiossincrático PETR4",
    }
    colors = {
        "ret_brent_usd": "#d97706",
        "ret_usd_brl": "#2e7d32",
        "ret_ibov": "#1565c0",
        "ret_petr4": "#6a1b9a",
    }

    fig = plt.figure(figsize=(15.5, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    fig.suptitle(
        f"Modelo Multivariado VAR({var_res['fitted_lag']}): Transmissão de Choques (IRF) e Decomposição da Variância (FEVD) de PETR4",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    impulse_cols = [c for c in cols]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (r_idx, c_idx), col in zip(positions, impulse_cols):
        ax = fig.add_subplot(gs[r_idx, c_idx])
        imp_idx = cols.index(col)
        resp = orth_irfs[:, target_idx, imp_idx] * 100.0
        err = 1.96 * stderr[:, target_idx, imp_idx] * 100.0
        ax.plot(horizons, resp, "o-", color=colors[col], linewidth=1.8, markersize=4)
        ax.fill_between(horizons, resp - err, resp + err, color=colors[col], alpha=0.18, label="IC 95%")
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(f"{pretty_names[col]} → PETR4")
        ax.set_xlabel("Meses após o choque")
        ax.set_ylabel("Resposta em PETR4 (p.p.)")
        ax.set_xticks(range(0, 13, 2))
        ax.legend(loc="upper right", fontsize=8)

    # Painel direito: FEVD de ret_petr4
    ax_fevd = fig.add_subplot(gs[:, 2])
    decomp_petr = fevd.decomp[target_idx] * 100.0  # (12, 4)
    h_fevd = np.arange(1, decomp_petr.shape[0] + 1)
    bottom = np.zeros(len(h_fevd))
    short_labels = {
        "ret_brent_usd": "Petróleo Brent (USD)",
        "ret_usd_brl": "Câmbio (USD/BRL)",
        "ret_ibov": "Ibovespa",
        "ret_petr4": "Choque Próprio PETR4",
    }
    for j, col in enumerate(cols):
        vals = decomp_petr[:, j]
        ax_fevd.bar(h_fevd, vals, bottom=bottom, color=colors[col], alpha=0.88, label=f"{short_labels[col]} ({vals[-1]:.1f}%)")
        bottom += vals

    ax_fevd.set_title("Decomposição da Variância (FEVD) de PETR4\n(Ordenação Cholesky: Brent → FX → Ibov → PETR4)")
    ax_fevd.set_xlabel("Horizonte (meses)")
    ax_fevd.set_ylabel("Participação na Variância do Erro (%)")
    ax_fevd.set_ylim(0, 105)
    ax_fevd.set_xticks(h_fevd)
    ax_fevd.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5)

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_benchmark_and_leakage(
    leakage_res: dict[str, Any],
    metrics_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    output_path: str | Path = "reports/figures/08_benchmark_walk_forward_leakage.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Painel demonstrando impacto do Data Leakage e ranking Walk-Forward dos 9 modelos."""
    fig = plt.figure(figsize=(16, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.15])
    fig.suptitle(
        "Validação Rigorosa: Armadilha de Data Leakage vs. Benchmark Walk-Forward Sem Vazamento (48 Meses)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # A: Data Leakage (R2 e Acurácia Direcional)
    ax_lk = fig.add_subplot(gs[0, 0])
    sum_df = leakage_res["summary_table"].copy()
    short_names = [
        "Ridge K-Fold\n(Vars t)",
        "Ridge Holdout\n(Vars t)",
        "RF K-Fold\n(Vars t)",
        "Ridge Causal\n(t-1 Realista)",
        "RF Causal\n(t-1 Realista)",
    ]
    y_pos = np.arange(len(short_names))
    bar_colors = ["#c62828", "#e53935", "#fb8c00", "#1565c0", "#2e7d32"]
    bars = ax_lk.barh(y_pos, sum_df["R2_Retorno"], color=bar_colors)
    ax_lk.axvline(0, color="black", linewidth=1.0)
    ax_lk.set_yticks(y_pos)
    ax_lk.set_yticklabels(short_names, fontsize=9)
    ax_lk.invert_yaxis()
    ax_lk.set_xlabel("R² Out-of-Sample no Retorno Mensal")
    ax_lk.set_title("A. Efeito do Data Leakage: R² Inflado (t) vs. Previsibilidade Real (t-1)")
    for bar, (_, r) in zip(bars, sum_df.iterrows()):
        w = bar.get_width()
        x_txt = max(w, 0.0) + 0.018
        ax_lk.text(
            x_txt,
            bar.get_y() + bar.get_height() / 2,
            f"R²={w:.2f} | Acerto={r['Acurácia_Direcional_%']:.0f}%",
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
        )
    ax_lk.set_xlim(min(-0.18, sum_df["R2_Retorno"].min() - 0.05), max(0.55, sum_df["R2_Retorno"].max() + 0.36))

    # B: Trajetória Walk-Forward no Preço Reconstruído
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_tr.plot(pred_df.index, pred_df["Real_Price"], color="black", linewidth=2.0, label="PETR4 Real (Adj Close)")
    for mdl_col, clr, ls in [
        ("SARIMAX (Exog Macro t-1)", "#1565c0", "--"),
        ("VAR(1) Multivariado", "#2e7d32", "-."),
        ("HistGradientBoosting Regressor", "#d97706", ":"),
    ]:
        p_rec = pred_df["Prev_Price"] * np.exp(pred_df[mdl_col])
        ax_tr.plot(pred_df.index, p_rec, color=clr, linestyle=ls, linewidth=1.4, label=mdl_col)
    ax_tr.set_title("B. Backtest Walk-Forward (Últimos 48 Meses): Preço Real vs. Previsão 1 Passo à Frente")
    ax_tr.set_ylabel("Preço Total Ajustado (R$)")
    ax_tr.legend(loc="upper left", fontsize=8.2)
    ax_tr.xaxis.set_major_locator(mdates.MonthLocator(interval=9))
    ax_tr.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_tr.tick_params(axis="x", rotation=20)

    # C: Ranking Comparativo RMSE Preço vs Acurácia Direcional
    ax_r1 = fig.add_subplot(gs[1, 0])
    m_sorted = metrics_df.sort_values("RMSE_Preço_R$", ascending=True)
    y_m = np.arange(len(m_sorted))
    cat_colors = {
        "Baseline": "#757575",
        "Econométrico Univariado": "#1e88e5",
        "Econométrico Exógeno": "#0d47a1",
        "Econométrico Multivariado": "#2e7d32",
        "Estado-Espaço": "#8e24aa",
        "Machine Learning": "#ef6c00",
        "Deep Learning (MLP)": "#d81b60",
    }
    colors_m = [cat_colors.get(c, "#444") for c in m_sorted["Categoria"]]
    b_rmse = ax_r1.barh(y_m, m_sorted["RMSE_Preço_R$"], color=colors_m)
    ax_r1.set_yticks(y_m)
    ax_r1.set_yticklabels(m_sorted["Modelo"], fontsize=8.8)
    ax_r1.invert_yaxis()
    ax_r1.set_xlabel("RMSE no Preço Reconstruído (R$)")
    ax_r1.set_title("C1. Ranking por Erro de Previsão (Menor RMSE = Melhor)")
    ax_r1.set_xlim(2.4, m_sorted["RMSE_Preço_R$"].max() + 0.18)
    for bar in b_rmse:
        w = bar.get_width()
        ax_r1.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"R$ {w:.3f}", va="center", fontsize=8.2)

    ax_r2 = fig.add_subplot(gs[1, 1])
    m_dir = metrics_df.dropna(subset=["Acurácia_Direcional_%"]).sort_values("Acurácia_Direcional_%", ascending=False)
    y_d = np.arange(len(m_dir))
    colors_d = [cat_colors.get(c, "#444") for c in m_dir["Categoria"]]
    b_dir = ax_r2.barh(y_d, m_dir["Acurácia_Direcional_%"], color=colors_d)
    ax_r2.axvline(50.0, color="#c62828", linestyle="--", linewidth=1.2, label="Cara ou Coroa (50%)")
    ax_r2.set_yticks(y_d)
    ax_r2.set_yticklabels(m_dir["Modelo"], fontsize=8.8)
    ax_r2.invert_yaxis()
    ax_r2.set_xlabel("Acurácia Direcional (%)")
    ax_r2.set_title("C2. Ranking por Acerto Direcional & Retorno Estratégia Long/Cash")
    ax_r2.set_xlim(0, 95)
    ax_r2.legend(loc="lower right", fontsize=8.5)
    for bar, (_, r) in zip(b_dir, m_dir.iterrows()):
        w = bar.get_width()
        ax_r2.text(
            w + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}% (Ret: +{r['Retorno_Estratégia_LongCash_%']:.0f}%)",
            va="center",
            fontsize=8.2,
        )

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return _save_fig(fig, output_path, close_fig=close_fig)


def plot_macro_scenarios(
    df_daily: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    output_path: str | Path = "reports/figures/09_cenarios_macroeconomicos.png",
    close_fig: bool = False,
) -> plt.Figure:
    """Projeção condicional de 12 meses para os 5 cenários macroeconômicos."""
    fig, ax = plt.subplots(figsize=(14.5, 7))
    # Série semanal recente (últimos 24 meses) + último fechamento diário exato
    cutoff_start = df_daily.index[-1] - pd.DateOffset(months=24)
    hist_weekly = df_daily.loc[df_daily.index >= cutoff_start, "petr4_adj_close"].resample("W-FRI").last().dropna()
    last_dt = df_daily.index[-1]
    last_price = float(df_daily["petr4_adj_close"].iloc[-1])
    if hist_weekly.index[-1] < last_dt:
        hist = pd.concat([hist_weekly, pd.Series([last_price], index=[last_dt])])
    else:
        hist = hist_weekly.copy()
        hist.iloc[-1] = last_price

    ax.plot(hist.index, hist.values, color="#1a237e", linewidth=2.2, label=f"Histórico PETR4 ({last_dt.date()}: R$ {last_price:.2f})")

    future_dates = pd.date_range(last_dt, last_dt + pd.DateOffset(months=12), periods=13)
    scen_colors = ["#b71c1c", "#e65100", "#1565c0", "#2e7d32", "#6a1b9a"]

    t_frac = np.linspace(0, 1, 13)
    for idx, row in scenarios_df.iterrows():
        target = float(row["Preço_Alvo_Implícito_R$"])
        ret_pct = float(row["Retorno_Esperado_PETR4_%"])
        # Trajetória exponencial suave a partir do preço real atual até o alvo em 12 meses
        path = last_price * np.exp(np.log(target / last_price) * t_frac)
        clr = scen_colors[idx % len(scen_colors)]
        ax.plot(
            future_dates,
            path,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=clr,
            label=f"{row['Cenário']} → R$ {target:.2f} ({ret_pct:+.1f}%)",
        )
        ax.annotate(
            f"R$ {target:.2f} ({ret_pct:+.1f}%)",
            xy=(future_dates[-1], target),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=clr,
        )

        # Banda de confiança 80% no Cenário Base
        if "Base" in row["Cenário"]:
            low_t = float(row["IC_80%_Inferior_R$"])
            high_t = float(row["IC_80%_Superior_R$"])
            low_path = last_price * np.exp(np.log(low_t / last_price) * np.sqrt(t_frac))
            high_path = last_price * np.exp(np.log(high_t / last_price) * np.sqrt(t_frac))
            ax.fill_between(future_dates, low_path, high_path, color=clr, alpha=0.12, label="IC 80% (Cenário Base)")

    ax.axvline(last_dt, color="#555555", linestyle="--", linewidth=1.2)
    ax.set_title(
        "Simulação de Cenários Macroeconômicos Condicionais (12 Meses) — Modelo Multifatorial PETR4",
        fontsize=13.5,
        fontweight="bold",
    )
    ax.set_ylabel("Preço Total Ajustado (R$)")
    ax.set_xlim(hist.index[0], future_dates[-1] + pd.Timedelta(days=95))
    ax.legend(loc="upper left", fontsize=8.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.text(0.01, 0.005, SOURCE_FOOTNOTE, fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    return _save_fig(fig, output_path, close_fig=close_fig)
