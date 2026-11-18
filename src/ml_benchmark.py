"""Módulo de Machine Learning, validação Walk-Forward sem Data Leakage e cenários macrofinanceiros.

Inclui:
- Demonstração empírica do efeito de Data Leakage (features contemporâneas vs. estritamente causais).
- Engenharia de features temporais estritamente defasadas (t-1 ou anterior).
- Benchmark Walk-Forward (janela expansiva, 1 passo à frente) comparando:
  Baseline Naive/Drift, ARIMA, SARIMAX, VAR, Unobserved Components, Random Forest,
  HistGradientBoosting e MLP Neural Network.
- Simulação de cenários macroeconômicos condicionais de 12 meses via modelo multifatorial.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents


def build_causal_ml_features(
    df_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Constrói matriz de features estritamente causais (informação disponível até t-1).

    Retorna (X_causal, y_target, price_context) alinhados sem valores ausentes.
    """
    df = df_monthly.copy()
    feat = pd.DataFrame(index=df.index)

    for lag in range(1, 7):
        feat[f"ret_petr4_lag{lag}"] = df["ret_petr4"].shift(lag)

    for lag in range(1, 4):
        feat[f"ret_ibov_lag{lag}"] = df["ret_ibov"].shift(lag)
        feat[f"ret_brent_lag{lag}"] = df["ret_brent_usd"].shift(lag)
        feat[f"ret_fx_lag{lag}"] = df["ret_usd_brl"].shift(lag)

    feat["vol_ann_lag1"] = df["vol_monthly_ann"].shift(1)
    feat["selic_m_lag1"] = df["selic_monthly_log"].shift(1)
    feat["mom_3m_petr4_lag1"] = df["ret_petr4"].rolling(3).sum().shift(1)
    feat["mom_6m_petr4_lag1"] = df["ret_petr4"].rolling(6).sum().shift(1)
    feat["mom_3m_brent_lag1"] = df["ret_brent_usd"].rolling(3).sum().shift(1)
    feat["spread_ibov_lag1"] = (df["ret_petr4"] - df["ret_ibov"]).shift(1)
    feat["spread_brent_lag1"] = (df["ret_petr4"] - df["ret_brent_brl"]).shift(1)

    # Sazonalidade determinística do calendário (conhecida ex-ante, sem leakage)
    month_num = df.index.month
    feat["month_sin"] = np.sin(2 * np.pi * month_num / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * month_num / 12.0)

    target = df["ret_petr4"].rename("target_ret")
    price_ctx = pd.DataFrame(
        {
            "prev_price": df["petr4_adj_close"].shift(1),
            "actual_price": df["petr4_adj_close"],
            "actual_ret": df["ret_petr4"],
        },
        index=df.index,
    )

    valid_idx = feat.dropna().index.intersection(target.dropna().index).intersection(price_ctx.dropna().index)
    return feat.loc[valid_idx], target.loc[valid_idx], price_ctx.loc[valid_idx]


def demonstrate_data_leakage(
    df_monthly: pd.DataFrame,
    test_size: int = 48,
) -> dict[str, Any]:
    """Compara empiricamente modelos com Data Leakage vs. modelos estritamente causais.

    Ilustra duas armadilhas clássicas em séries financeiras:
    1) K-Fold aleatório com embaralhamento temporal (usa o futuro para prever o passado).
    2) Uso de variáveis contemporâneas (Ibov_t, Brent_t, FX_t) conhecidas apenas após o fechamento de t.
    """
    X_causal, y_target, price_ctx = build_causal_ml_features(df_monthly)
    df_aligned = df_monthly.loc[X_causal.index]
    X_leak_contemp = df_aligned[["ret_ibov", "ret_brent_usd", "ret_usd_brl"]]

    split_pos = len(X_causal) - test_size
    y_train, y_test = y_target.iloc[:split_pos], y_target.iloc[split_pos:]
    Xc_train, Xc_test = X_causal.iloc[:split_pos], X_causal.iloc[split_pos:]
    Xl_train, Xl_test = X_leak_contemp.iloc[:split_pos], X_leak_contemp.iloc[split_pos:]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Ridge com K-Fold Aleatório + Variáveis Contemporâneas (Leakage duplo)
    ridge_kf = Ridge(alpha=1.0)
    pred_ridge_kf = pd.Series(
        cross_val_predict(ridge_kf, X_leak_contemp, y_target, cv=kf),
        index=y_target.index,
    ).iloc[split_pos:]

    # 2. Ridge Holdout Temporal com Variáveis Contemporâneas em t (Leakage contemporâneo)
    ridge_contemp = Ridge(alpha=1.0).fit(Xl_train, y_train)
    pred_ridge_contemp = pd.Series(ridge_contemp.predict(Xl_test), index=y_test.index)

    # 3. Ridge Estritamente Causal (features t-1)
    ridge_causal = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=50.0))]).fit(
        Xc_train, y_train
    )
    pred_ridge_causal = pd.Series(ridge_causal.predict(Xc_test), index=y_test.index)

    # 4. Random Forest com K-Fold Aleatório + Contemporâneas vs. Causal t-1
    rf_kf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    pred_rf_kf = pd.Series(
        cross_val_predict(rf_kf, X_leak_contemp, y_target, cv=kf),
        index=y_target.index,
    ).iloc[split_pos:]

    rf_causal = RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42
    ).fit(Xc_train, y_train)
    pred_rf_causal = pd.Series(rf_causal.predict(Xc_test), index=y_test.index)

    def _summarize(name: str, pred_s: pd.Series) -> dict[str, Any]:
        prev_p = price_ctx.loc[y_test.index, "prev_price"]
        act_p = price_ctx.loc[y_test.index, "actual_price"]
        pred_p = prev_p * np.exp(pred_s)
        return {
            "Cenário": name,
            "R2_Retorno": float(r2_score(y_test, pred_s)),
            "RMSE_Retorno_%": float(np.sqrt(mean_squared_error(y_test, pred_s)) * 100.0),
            "MAE_Preço_R$": float(mean_absolute_error(act_p, pred_p)),
            "Acurácia_Direcional_%": float(np.mean(np.sign(pred_s) == np.sign(y_test)) * 100.0),
        }

    summary_df = pd.DataFrame(
        [
            _summarize("Com Leakage (Ridge): K-Fold Aleatório + Vars em t", pred_ridge_kf),
            _summarize("Com Leakage (Ridge): Holdout + Vars em t", pred_ridge_contemp),
            _summarize("Com Leakage (RF): K-Fold Aleatório + Vars em t", pred_rf_kf),
            _summarize("Sem Leakage (Ridge): Features Estritamente t-1", pred_ridge_causal),
            _summarize("Sem Leakage (RF): Features Estritamente t-1", pred_rf_causal),
        ]
    )

    return {
        "summary_table": summary_df,
        "y_test": y_test,
        "pred_kfold_test": pred_ridge_kf,
        "pred_contemp_test": pred_ridge_contemp,
        "pred_causal_test": pred_rf_causal,
        "price_ctx_test": price_ctx.loc[y_test.index],
    }


def _compute_metrics_row(
    model_name: str,
    category: str,
    actual_ret: pd.Series,
    pred_ret: pd.Series,
    prev_price: pd.Series,
    actual_price: pd.Series,
) -> dict[str, Any]:
    """Calcula métricas no espaço de retorno e no espaço de preço reconstruído."""
    pred_price = prev_price * np.exp(pred_ret)

    rmse_ret = float(np.sqrt(mean_squared_error(actual_ret, pred_ret)) * 100.0)
    mae_ret = float(mean_absolute_error(actual_ret, pred_ret) * 100.0)
    rmse_price = float(np.sqrt(mean_squared_error(actual_price, pred_price)))
    mae_price = float(mean_absolute_error(actual_price, pred_price))
    mape_price = float(mean_absolute_percentage_error(actual_price, pred_price) * 100.0)

    if np.allclose(pred_ret.values, 0.0):
        dir_acc = np.nan
    else:
        dir_acc = float(np.mean(np.sign(pred_ret) == np.sign(actual_ret)) * 100.0)

    # Retorno acumulado simples de uma regra Long/Cash (comprado se retorno previsto > 0)
    signal = (pred_ret > 0).astype(float)
    strat_log_ret = float((signal * actual_ret).sum())
    strat_cum_pct = float(np.expm1(strat_log_ret) * 100.0)

    return {
        "Modelo": model_name,
        "Categoria": category,
        "RMSE_Retorno_%": rmse_ret,
        "MAE_Retorno_%": mae_ret,
        "RMSE_Preço_R$": rmse_price,
        "MAE_Preço_R$": mae_price,
        "MAPE_Preço_%": mape_price,
        "Acurácia_Direcional_%": dir_acc,
        "Retorno_Estratégia_LongCash_%": strat_cum_pct,
    }


def run_walk_forward_benchmark(
    df_monthly: pd.DataFrame,
    test_size: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa backtesting Walk-Forward (janela expansiva, 1 passo à frente) sem Data Leakage."""
    X_causal, y_target, price_ctx = build_causal_ml_features(df_monthly)
    test_dates = y_target.index[-test_size:]

    # Séries alinhadas no calendário completo de df_monthly
    full_ret = df_monthly["ret_petr4"]
    full_log_price = df_monthly["log_petr4_adj"]
    exog_lag1 = df_monthly[["ret_ibov", "ret_brent_usd", "ret_usd_brl"]].shift(1)
    var_cols = ["ret_petr4", "ret_ibov", "ret_brent_usd", "ret_usd_brl"]
    full_var = df_monthly[var_cols].copy()

    preds: dict[str, list[float]] = {
        "Naive (Random Walk Sem Drift)": [],
        "Drift Histórico (Média Expansiva)": [],
        "ARIMA(1,0,1) (Walk-Forward)": [],
        "SARIMAX (Exog Macro t-1)": [],
        "VAR(1) Multivariado": [],
        "Estado-Espaço (UnobservedComponents)": [],
        "Random Forest Regressor": [],
        "HistGradientBoosting Regressor": [],
        "Rede Neural MLP": [],
    }

    for dt in test_dates:
        # Corte estritamente causal: apenas dados anteriores a dt (< dt)
        hist_ret = full_ret.loc[:dt].iloc[:-1].dropna()
        hist_log_p = full_log_price.loc[:dt].iloc[:-1].dropna()

        # 1. Naive Sem Drift
        preds["Naive (Random Walk Sem Drift)"].append(0.0)

        # 2. Drift Histórico
        preds["Drift Histórico (Média Expansiva)"].append(float(hist_ret.mean()))

        # 3. ARIMA(1,0,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod_ar = ARIMA(hist_ret.values, order=(1, 0, 1)).fit()
                f_ar = float(mod_ar.forecast(steps=1)[0])
        except Exception:
            f_ar = float(hist_ret.mean())
        preds["ARIMA(1,0,1) (Walk-Forward)"].append(f_ar)

        # 4. SARIMAX com exógenas defasadas em t-1 (já alinhadas ao tempo t!)
        try:
            joined_sx = pd.concat([full_ret, exog_lag1], axis=1).loc[:dt].dropna()
            train_sx = joined_sx.iloc[:-1]
            curr_exog = joined_sx.iloc[[-1]][["ret_ibov", "ret_brent_usd", "ret_usd_brl"]].values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod_sx = ARIMA(
                    train_sx["ret_petr4"].values,
                    exog=train_sx[["ret_ibov", "ret_brent_usd", "ret_usd_brl"]].values,
                    order=(1, 0, 0),
                ).fit()
                f_sx = float(mod_sx.forecast(steps=1, exog=curr_exog)[0])
        except Exception:
            f_sx = float(hist_ret.mean())
        preds["SARIMAX (Exog Macro t-1)"].append(f_sx)

        # 5. VAR(1)
        try:
            hist_var = full_var.loc[:dt].iloc[:-1].dropna()
            var_fit = VAR(hist_var.values).fit(1)
            f_var = float(var_fit.forecast(hist_var.values[-1:], steps=1)[0, 0])
        except Exception:
            f_var = float(hist_ret.mean())
        preds["VAR(1) Multivariado"].append(f_var)

        # 6. Estado-Espaço Estrutural (UnobservedComponents — equivalente em estado-espaço ao Prophet)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                uc_mod = UnobservedComponents(
                    hist_log_p.values,
                    level="local linear trend",
                    seasonal=12,
                ).fit(disp=False, maxiter=100)
                next_log_p = float(uc_mod.forecast(steps=1)[0])
                f_uc = next_log_p - float(hist_log_p.iloc[-1])
        except Exception:
            f_uc = float(hist_ret.mean())
        preds["Estado-Espaço (UnobservedComponents)"].append(f_uc)

        # 7-9. Modelos de Machine Learning (treinados apenas em X_causal < dt)
        X_tr = X_causal.loc[:dt].iloc[:-1]
        y_tr = y_target.loc[:dt].iloc[:-1]
        X_te = X_causal.loc[[dt]]

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        preds["Random Forest Regressor"].append(float(rf.predict(X_te)[0]))

        hgb = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.04,
            max_iter=120,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )
        hgb.fit(X_tr, y_tr)
        preds["HistGradientBoosting Regressor"].append(float(hgb.predict(X_te)[0]))

        mlp = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(24, 12),
                        alpha=1.0,
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_tr, y_tr)
        preds["Rede Neural MLP"].append(float(mlp.predict(X_te)[0]))

    pred_df = pd.DataFrame(preds, index=test_dates)
    pred_df["Real_Retorno"] = y_target.loc[test_dates]
    pred_df["Prev_Price"] = price_ctx.loc[test_dates, "prev_price"]
    pred_df["Real_Price"] = price_ctx.loc[test_dates, "actual_price"]

    categories = {
        "Naive (Random Walk Sem Drift)": "Baseline",
        "Drift Histórico (Média Expansiva)": "Baseline",
        "ARIMA(1,0,1) (Walk-Forward)": "Econométrico Univariado",
        "SARIMAX (Exog Macro t-1)": "Econométrico Exógeno",
        "VAR(1) Multivariado": "Econométrico Multivariado",
        "Estado-Espaço (UnobservedComponents)": "Estado-Espaço",
        "Random Forest Regressor": "Machine Learning",
        "HistGradientBoosting Regressor": "Machine Learning",
        "Rede Neural MLP": "Deep Learning (MLP)",
    }

    metrics_rows = [
        _compute_metrics_row(
            model_name=col,
            category=categories[col],
            actual_ret=pred_df["Real_Retorno"],
            pred_ret=pred_df[col],
            prev_price=pred_df["Prev_Price"],
            actual_price=pred_df["Real_Price"],
        )
        for col in categories
    ]

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values("RMSE_Preço_R$", ascending=True)
        .reset_index(drop=True)
    )
    return metrics_df, pred_df


def simulate_macro_scenarios(
    multifactor_model: Any,
    current_price: float,
    selic_annual_pct: float,
) -> pd.DataFrame:
    """Simula cenários macroeconômicos de 12 meses usando as elasticidades do modelo multifatorial."""
    params = multifactor_model.params
    alpha_m = float(params.get("const", 0.0))
    b_ibov = float(params.get("excess_ibov", 1.0))
    b_brent = float(params.get("ret_brent_usd", 0.0))
    b_fx = float(params.get("ret_usd_brl", 0.0))

    # Desvio padrão residual anualizado para bandas de incerteza de 12 meses
    resid_std_ann = float(np.std(multifactor_model.resid, ddof=len(params)) * np.sqrt(12))

    scenarios = [
        {
            "Cenário": "1. Estresse Global / Recessão",
            "Ibovespa_12m_%": -15.0,
            "Brent_USD_12m_%": -25.0,
            "USD_BRL_12m_%": 15.0,
            "Selic_Final_%_aa": selic_annual_pct + 1.5,
        },
        {
            "Cenário": "2. Cautela / Desaceleração",
            "Ibovespa_12m_%": 2.0,
            "Brent_USD_12m_%": -10.0,
            "USD_BRL_12m_%": 5.0,
            "Selic_Final_%_aa": selic_annual_pct,
        },
        {
            "Cenário": "3. Base / Consenso Macro",
            "Ibovespa_12m_%": 12.0,
            "Brent_USD_12m_%": 0.0,
            "USD_BRL_12m_%": 0.0,
            "Selic_Final_%_aa": max(2.0, selic_annual_pct - 1.0),
        },
        {
            "Cenário": "4. Otimista / Boom Commodities",
            "Ibovespa_12m_%": 22.0,
            "Brent_USD_12m_%": 25.0,
            "USD_BRL_12m_%": -8.0,
            "Selic_Final_%_aa": max(2.0, selic_annual_pct - 2.0),
        },
        {
            "Cenário": "5. Choque de Oferta (Brent +40%)",
            "Ibovespa_12m_%": 10.0,
            "Brent_USD_12m_%": 40.0,
            "USD_BRL_12m_%": -5.0,
            "Selic_Final_%_aa": selic_annual_pct + 0.5,
        },
    ]

    rows: list[dict[str, Any]] = []
    for sc in scenarios:
        rf_avg_pct = 0.5 * (selic_annual_pct + sc["Selic_Final_%_aa"])
        rf_log = np.log1p(rf_avg_pct / 100.0)
        ibov_log = np.log1p(sc["Ibovespa_12m_%"] / 100.0)
        brent_log = np.log1p(sc["Brent_USD_12m_%"] / 100.0)
        fx_log = np.log1p(sc["USD_BRL_12m_%"] / 100.0)

        excess_ibov_12m = ibov_log - rf_log
        expected_excess_petr4 = (
            12.0 * alpha_m
            + b_ibov * excess_ibov_12m
            + b_brent * brent_log
            + b_fx * fx_log
        )
        expected_log_ret = rf_log + expected_excess_petr4
        expected_ret_pct = float(np.expm1(expected_log_ret) * 100.0)
        target_price = float(current_price * np.exp(expected_log_ret))
        ic_low_80 = float(current_price * np.exp(expected_log_ret - 1.2816 * resid_std_ann))
        ic_high_80 = float(current_price * np.exp(expected_log_ret + 1.2816 * resid_std_ann))

        rows.append(
            {
                **sc,
                "Retorno_Esperado_PETR4_%": expected_ret_pct,
                "Preço_Alvo_Implícito_R$": target_price,
                "IC_80%_Inferior_R$": ic_low_80,
                "IC_80%_Superior_R$": ic_high_80,
            }
        )

    return pd.DataFrame(rows)
