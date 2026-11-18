"""Módulo de econometria financeira e séries temporais para PETR4.

Inclui:
- Testes de estacionariedade combinados (ADF e KPSS).
- Teste F de Chow para quebras estruturais (implementação exata via OLS/statsmodels) e varredura rolante.
- Estimação de CAPM clássico, CAPM dinâmico (Beta rolante) e modelo multifatorial macrofinanceiro (HAC).
- Seleção e diagnóstico Box-Jenkins (ARIMA / SARIMAX com Ljung-Box e Jarque-Bera).
- Modelagem multivariada VAR, Causalidade de Granger, Impulso-Resposta (IRF) e Decomposição da Variância (FEVD).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from src.data_pipeline import HISTORICAL_EVENTS, REGIME_PERIODS, assign_regime


def stationarity_report(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Executa os testes ADF e KPSS para um dicionário de séries temporais."""
    rows: list[dict[str, Any]] = []
    for name, raw_series in series_dict.items():
        s = pd.Series(raw_series).dropna().astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_res = adfuller(s, autolag="AIC")
            kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="c", nlags="auto")

        adf_reject_5 = adf_res[1] < 0.05
        kpss_reject_5 = kpss_p < 0.05
        if adf_reject_5 and not kpss_reject_5:
            diag = "Estacionária I(0)"
        elif (not adf_reject_5) and kpss_reject_5:
            diag = "Não-Estacionária (Raiz Unitária)"
        elif adf_reject_5 and kpss_reject_5:
            diag = "Estacionária em tendência / Quebra estrutural"
        else:
            diag = "Inconclusivo (baixa potência)"

        rows.append(
            {
                "Série": name,
                "N": len(s),
                "ADF_Stat": float(adf_res[0]),
                "ADF_p_valor": float(adf_res[1]),
                "ADF_Lags": int(adf_res[2]),
                "ADF_Crit_1%": float(adf_res[4]["1%"]),
                "ADF_Crit_5%": float(adf_res[4]["5%"]),
                "ADF_Crit_10%": float(adf_res[4]["10%"]),
                "KPSS_Stat": float(kpss_stat),
                "KPSS_p_valor": float(kpss_p),
                "KPSS_Lags": int(kpss_lags),
                "Diagnóstico": diag,
            }
        )
    return pd.DataFrame(rows)


def chow_test_ols(
    y: pd.Series,
    X: pd.DataFrame,
    break_date: str | pd.Timestamp,
) -> dict[str, Any]:
    """Implementa o Teste F de Chow para quebra estrutural em regressão linear OLS.

    F = ((SSR_restrito - (SSR_1 + SSR_2)) / k) / ((SSR_1 + SSR_2) / (n_1 + n_2 - 2k))
    """
    df_joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    y_clean = df_joined["__y__"]
    X_clean = sm.add_constant(df_joined.drop(columns=["__y__"]), has_constant="add")

    break_ts = pd.Timestamp(break_date)
    pre_mask = y_clean.index < break_ts
    post_mask = y_clean.index >= break_ts

    n1 = int(pre_mask.sum())
    n2 = int(post_mask.sum())
    k = int(X_clean.shape[1])

    if n1 <= k or n2 <= k:
        raise ValueError(
            f"Amostra insuficiente para o Teste de Chow em {break_ts.date()}: n1={n1}, n2={n2}, k={k}."
        )

    fit_pooled = sm.OLS(y_clean, X_clean).fit()
    fit_pre = sm.OLS(y_clean.loc[pre_mask], X_clean.loc[pre_mask]).fit()
    fit_post = sm.OLS(y_clean.loc[post_mask], X_clean.loc[post_mask]).fit()

    ssr_pooled = float(fit_pooled.ssr)
    ssr_split = float(fit_pre.ssr + fit_post.ssr)
    df_num = k
    df_den = n1 + n2 - 2 * k

    num = max(0.0, (ssr_pooled - ssr_split) / df_num)
    den = ssr_split / df_den
    f_stat = float(num / den) if den > 0 else np.nan
    p_value = float(stats.f.sf(f_stat, df_num, df_den)) if np.isfinite(f_stat) else np.nan

    return {
        "break_date": break_ts,
        "n_total": n1 + n2,
        "n_pre": n1,
        "n_post": n2,
        "k": k,
        "df_num": df_num,
        "df_den": df_den,
        "f_stat": f_stat,
        "p_value": p_value,
        "f_crit_5pct": float(stats.f.ppf(0.95, df_num, df_den)),
        "f_crit_1pct": float(stats.f.ppf(0.99, df_num, df_den)),
        "reject_h0_5pct": bool(p_value < 0.05),
        "reject_h0_1pct": bool(p_value < 0.01),
        "ssr_pooled": ssr_pooled,
        "ssr_split": ssr_split,
        "r2_pooled": float(fit_pooled.rsquared),
        "r2_pre": float(fit_pre.rsquared),
        "r2_post": float(fit_post.rsquared),
        "params_pre": fit_pre.params.to_dict(),
        "params_post": fit_post.params.to_dict(),
    }


def rolling_chow_f_stat(
    y: pd.Series,
    X: pd.DataFrame,
    half_window: int = 252,
    step: int = 5,
    mode: str = "local",
) -> pd.DataFrame:
    """Calcula a curva da estatística F de Chow ao longo do tempo.

    - mode='local': janela móvel simétrica de [pos - half_window, pos + half_window].
    - mode='global': amostra completa dividida em cada data candidata (Estilo Quandt-Andrews).
    """
    df_joined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    y_clean = df_joined["__y__"]
    X_clean = sm.add_constant(df_joined.drop(columns=["__y__"]), has_constant="add")
    k = int(X_clean.shape[1])

    records: list[dict[str, Any]] = []
    n_obs = len(y_clean)
    fit_global = sm.OLS(y_clean, X_clean).fit() if mode == "global" else None
    ssr_global = float(fit_global.ssr) if fit_global is not None else np.nan

    for pos in range(half_window, n_obs - half_window, step):
        break_dt = y_clean.index[pos]
        if mode == "local":
            sub_y = y_clean.iloc[pos - half_window : pos + half_window]
            sub_X = X_clean.iloc[pos - half_window : pos + half_window]
            # Se X contém t_local constante/global, recalcula tempo relativo dentro da janela
            if "t_local" in sub_X.columns:
                sub_X = sub_X.copy()
                sub_X["t_local"] = np.arange(len(sub_X), dtype=float) / len(sub_X)
            pre_y, pre_X = sub_y.iloc[:half_window], sub_X.iloc[:half_window]
            post_y, post_X = sub_y.iloc[half_window:], sub_X.iloc[half_window:]
            fit_pool = sm.OLS(sub_y, sub_X).fit()
            ssr_p = float(fit_pool.ssr)
            df_den = 2 * half_window - 2 * k
        else:
            pre_y, pre_X = y_clean.iloc[:pos], X_clean.iloc[:pos]
            post_y, post_X = y_clean.iloc[pos:], X_clean.iloc[pos:]
            ssr_p = ssr_global
            df_den = n_obs - 2 * k

        fit_1 = sm.OLS(pre_y, pre_X).fit()
        fit_2 = sm.OLS(post_y, post_X).fit()
        ssr_s = float(fit_1.ssr + fit_2.ssr)
        f_stat = max(0.0, (ssr_p - ssr_s) / k) / (ssr_s / df_den)
        p_val = float(stats.f.sf(f_stat, k, df_den))
        f_crit_1 = float(stats.f.ppf(0.99, k, df_den))
        records.append(
            {
                "date": break_dt,
                "f_stat": float(f_stat),
                "p_value": p_val,
                "f_crit_1pct": f_crit_1,
            }
        )

    return pd.DataFrame(records).set_index("date")


def evaluate_historical_breaks(
    df_daily: pd.DataFrame,
    events: dict[str, tuple[str, str]] | None = None,
    window_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Testa quebras estruturais nos marcos históricos sob duas especificações complementares.

    - Spec A (Trajetória/Tendência Local ±252d): log_petr4_adj ~ 1 + t_local
    - Spec B (Multifatorial Amostra Completa): excess_petr4 ~ 1 + excess_ibov + ret_brent_usd + ret_usd_brl
    """
    if events is None:
        events = HISTORICAL_EVENTS

    work = df_daily.copy()
    work["t_local"] = np.arange(len(work), dtype=float) / max(1, len(work))

    rows: list[dict[str, Any]] = []
    for date_str, (label, _) in events.items():
        ev_ts = pd.Timestamp(date_str)
        pos = int(work.index.searchsorted(ev_ts))
        start_pos = max(0, pos - window_days)
        end_pos = min(len(work), pos + window_days)
        sub = work.iloc[start_pos:end_pos].copy()
        sub["t_local"] = np.arange(len(sub), dtype=float) / max(1, len(sub))

        # Spec A: Tendência Local (Trajetória do Log-Preço)
        res_a = chow_test_ols(
            sub["log_petr4_adj"],
            sub[["t_local"]],
            ev_ts,
        )
        rows.append(
            {
                "Evento": label,
                "Data_Marco": date_str,
                "Especificação": "A: Trajetória Local ±252d (Log-Preço)",
                "N_Pré": res_a["n_pre"],
                "N_Pós": res_a["n_post"],
                "F_Stat": res_a["f_stat"],
                "F_Crit_1%": res_a["f_crit_1pct"],
                "p_valor": res_a["p_value"],
                "Quebra_1%": res_a["reject_h0_1pct"],
                "Quebra_5%": res_a["reject_h0_5pct"],
                "Beta_Ibov_Pré": np.nan,
                "Beta_Ibov_Pós": np.nan,
                "Beta_Brent_Pré": np.nan,
                "Beta_Brent_Pós": np.nan,
            }
        )

        # Spec B: Multifatorial (CAPM + Brent + Câmbio) na amostra completa (pré vs pós-evento)
        res_b = chow_test_ols(
            work["excess_petr4"],
            work[["excess_ibov", "ret_brent_usd", "ret_usd_brl"]],
            ev_ts,
        )
        rows.append(
            {
                "Evento": label,
                "Data_Marco": date_str,
                "Especificação": "B: Multifatorial Estrutural (Amostra Completa)",
                "N_Pré": res_b["n_pre"],
                "N_Pós": res_b["n_post"],
                "F_Stat": res_b["f_stat"],
                "F_Crit_1%": res_b["f_crit_1pct"],
                "p_valor": res_b["p_value"],
                "Quebra_1%": res_b["reject_h0_1pct"],
                "Quebra_5%": res_b["reject_h0_5pct"],
                "Beta_Ibov_Pré": res_b["params_pre"].get("excess_ibov", np.nan),
                "Beta_Ibov_Pós": res_b["params_post"].get("excess_ibov", np.nan),
                "Beta_Brent_Pré": res_b["params_pre"].get("ret_brent_usd", np.nan),
                "Beta_Brent_Pós": res_b["params_post"].get("ret_brent_usd", np.nan),
            }
        )

    table = pd.DataFrame(rows)

    # Curvas ao longo da história (2001-2025)
    rolling_a = rolling_chow_f_stat(
        work["log_petr4_adj"],
        work[["t_local"]],
        half_window=window_days,
        step=5,
        mode="local",
    )
    rolling_b = rolling_chow_f_stat(
        work["excess_petr4"],
        work[["excess_ibov", "ret_brent_usd", "ret_usd_brl"]],
        half_window=window_days,
        step=5,
        mode="global",
    )
    rolling_curves = pd.DataFrame(
        {
            "f_stat_trend_local": rolling_a["f_stat"],
            "f_crit_1pct_trend": rolling_a["f_crit_1pct"],
            "f_stat_multifactor_global": rolling_b["f_stat"],
            "f_crit_1pct_multifactor": rolling_b["f_crit_1pct"],
        }
    )
    return table, rolling_curves


def estimate_capm_and_multifactor(
    df_monthly: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> dict[str, Any]:
    """Estima CAPM clássico, multifatorial macrofinanceiro, Betas rolantes e por regime."""
    # 1. CAPM Mensal (erros-padrão Newey-West HAC)
    X_capm_m = sm.add_constant(df_monthly[["excess_ibov"]])
    capm_m = sm.OLS(df_monthly["excess_petr4"], X_capm_m).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )

    # 2. Multifatorial Mensal (CAPM + Brent + Câmbio)
    X_multi_m = sm.add_constant(
        df_monthly[["excess_ibov", "ret_brent_usd", "ret_usd_brl"]]
    )
    multi_m = sm.OLS(df_monthly["excess_petr4"], X_multi_m).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )

    # 3. CAPM Diário e Multifatorial Diário
    X_capm_d = sm.add_constant(df_daily[["excess_ibov"]])
    capm_d = sm.OLS(df_daily["excess_petr4"], X_capm_d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}
    )
    X_multi_d = sm.add_constant(
        df_daily[["excess_ibov", "ret_brent_usd", "ret_usd_brl"]]
    )
    multi_d = sm.OLS(df_daily["excess_petr4"], X_multi_d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10}
    )

    # 4. Beta Rolante Diário (janela de 252 pregões) e Mensal (24 meses)
    cov_d = df_daily["excess_petr4"].rolling(252).cov(df_daily["excess_ibov"])
    var_d = df_daily["excess_ibov"].rolling(252).var()
    rolling_beta_252d = (cov_d / var_d).rename("beta_252d")

    cov_m = df_monthly["excess_petr4"].rolling(24).cov(df_monthly["excess_ibov"])
    var_m = df_monthly["excess_ibov"].rolling(24).var()
    rolling_beta_24m = (cov_m / var_m).rename("beta_24m")

    rolling_corr_ibov_252d = (
        df_daily["ret_petr4"].rolling(252).corr(df_daily["ret_ibov"]).rename("corr_ibov_252d")
    )
    rolling_corr_brent_252d = (
        df_daily["ret_petr4"].rolling(252).corr(df_daily["ret_brent_usd"]).rename("corr_brent_252d")
    )

    # 5. Tabela por Regime Histórico (base diária para robustez amostral em regimes curtos)
    df_reg = df_daily.copy()
    df_reg["Regime"] = assign_regime(df_reg.index)
    regime_rows: list[dict[str, Any]] = []
    for _, _, label in REGIME_PERIODS:
        sub = df_reg.loc[df_reg["Regime"] == label]
        if len(sub) < 30:
            continue
        fit_c = sm.OLS(sub["excess_petr4"], sm.add_constant(sub[["excess_ibov"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        fit_m = sm.OLS(
            sub["excess_petr4"],
            sm.add_constant(sub[["excess_ibov", "ret_brent_usd", "ret_usd_brl"]]),
        ).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

        ann_ret = float(np.expm1(sub["ret_petr4"].mean() * 252) * 100.0)
        ann_vol = float(sub["ret_petr4"].std() * np.sqrt(252) * 100.0)
        regime_rows.append(
            {
                "Regime": label,
                "Pregões": len(sub),
                "Retorno_Anual_%": ann_ret,
                "Volatilidade_Anual_%": ann_vol,
                "Beta_CAPM": float(fit_c.params["excess_ibov"]),
                "R2_CAPM": float(fit_c.rsquared),
                "Beta_Multi_Ibov": float(fit_m.params["excess_ibov"]),
                "Beta_Multi_Brent": float(fit_m.params["ret_brent_usd"]),
                "Beta_Multi_FX": float(fit_m.params["ret_usd_brl"]),
                "R2_Multi": float(fit_m.rsquared),
            }
        )

    return {
        "capm_monthly": capm_m,
        "multifactor_monthly": multi_m,
        "capm_daily": capm_d,
        "multifactor_daily": multi_d,
        "rolling_beta_252d": rolling_beta_252d,
        "rolling_beta_24m": rolling_beta_24m,
        "rolling_corr_ibov_252d": rolling_corr_ibov_252d,
        "rolling_corr_brent_252d": rolling_corr_brent_252d,
        "regime_table": pd.DataFrame(regime_rows),
    }


def select_and_fit_arima(
    series: pd.Series,
    max_p: int = 4,
    max_q: int = 4,
    d: int = 1,
    exog: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Busca em grade (p, d, q) por AIC/BIC e executa diagnóstico completo dos resíduos."""
    s = pd.Series(series).dropna().astype(float)
    s.index = pd.DatetimeIndex(s.index)
    exog_clean = exog.loc[s.index] if exog is not None else None

    grid_rows: list[dict[str, Any]] = []
    best_aic = np.inf
    best_model = None
    best_order = (0, d, 0)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = ARIMA(s, exog=exog_clean, order=(p, d, q))
                    res = mod.fit()
                grid_rows.append(
                    {
                        "order": f"({p},{d},{q})",
                        "p": p,
                        "d": d,
                        "q": q,
                        "AIC": float(res.aic),
                        "BIC": float(res.bic),
                        "LogLik": float(res.llf),
                    }
                )
                if res.aic < best_aic:
                    best_aic = float(res.aic)
                    best_model = res
                    best_order = (p, d, q)
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError("Nenhum modelo ARIMA convergiu na busca em grade.")

    grid_df = pd.DataFrame(grid_rows).sort_values("AIC").reset_index(drop=True)
    # Remove possível resíduo inicial igual ao valor inicial quando d >= 1
    resid = pd.Series(best_model.resid).iloc[d:].dropna()
    lb_df = acorr_ljungbox(resid, lags=[5, 10, 20], return_df=True)
    jb_stat, jb_pval = stats.jarque_bera(resid)

    return {
        "best_order": best_order,
        "best_model": best_model,
        "grid_table": grid_df,
        "residuals": resid,
        "ljung_box": lb_df,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pval": float(jb_pval),
    }


def fit_var_and_granger(
    df_monthly: pd.DataFrame,
    cols: list[str] | None = None,
    maxlags: int = 6,
) -> dict[str, Any]:
    """Ajusta modelo VAR nos retornos mensais e executa Causalidade de Granger, IRF e FEVD.

    A ordenação padrão segue causalidade econômica (do mais exógeno ao mais endógeno)
    para ortogonalização de Cholesky: Brent -> USD/BRL -> Ibovespa -> PETR4.
    """
    if cols is None:
        cols = ["ret_brent_usd", "ret_usd_brl", "ret_ibov", "ret_petr4"]

    data = df_monthly[cols].dropna().copy()
    # Garante frequência explícita mensal para statsmodels VAR
    data.index = pd.PeriodIndex(data.index, freq="M")

    var_model = VAR(data)
    order_sel = var_model.select_order(maxlags=maxlags)
    selected_lag = int(order_sel.selected_orders.get("aic", 1))
    fit_lag = max(1, selected_lag)
    results = var_model.fit(fit_lag)

    # Matriz de Causalidade de Granger (cada coluna causando cada outra coluna)
    granger_rows: list[dict[str, Any]] = []
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            test_res = results.test_causality(caused=caused, causing=[causing], kind="f")
            granger_rows.append(
                {
                    "Causadora": causing,
                    "Alvo": caused,
                    "Lag_VAR": fit_lag,
                    "F_Stat": float(test_res.test_statistic),
                    "p_valor": float(test_res.pvalue),
                    "Significativo_5%": bool(test_res.pvalue < 0.05),
                    "Significativo_10%": bool(test_res.pvalue < 0.10),
                }
            )

    irf = results.irf(12)
    fevd = results.fevd(12)

    return {
        "var_results": results,
        "order_selection_summary": order_sel.summary(),
        "selected_lag_aic": selected_lag,
        "fitted_lag": fit_lag,
        "granger_table": pd.DataFrame(granger_rows),
        "irf": irf,
        "fevd": fevd,
        "cols": cols,
    }
