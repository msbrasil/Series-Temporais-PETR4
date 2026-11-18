"""Pipeline de coleta, higienização e engenharia de features para PETR4 e variáveis macrofinanceiras.

Fontes utilizadas:
- Yahoo Finance (`yfinance`): PETR4.SA (Close nominal ajustado a splits e Adj Close ajustado a dividendos),
  ^BVSP (Ibovespa) e BZ=F (Petróleo Brent a partir de jul/2007).
- IPEADATA (arquivo local legado `data/raw/brent.xlsx`): Petróleo Brent diário para cobertura de 2000–2007.
- Banco Central do Brasil (API SGS):
  - Série 1: Taxa de câmbio PTAX USD/BRL (venda, diária).
  - Série 11: Taxa Selic diária (% a.d.).
  - Série 432: Meta Selic definida pelo COPOM (% a.a.).
  - Série 4390: Taxa Selic acumulada no mês (% a.m.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BCB_SGS_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
    "?formato=json&dataInicial={start}&dataFinal={end}"
)

HISTORICAL_EVENTS: dict[str, tuple[str, str]] = {
    "2006-07-01": ("Descoberta do Pré-Sal", "#2b5c8f"),
    "2008-09-15": ("Crise Subprime (Lehman)", "#b22222"),
    "2016-05-30": ("Posse Pedro Parente / PPI", "#2e7d32"),
    "2020-03-09": ("Choque COVID-19 & Petróleo", "#d97706"),
    "2023-05-16": ("Fim do PPI / Novo Regime", "#6a1b9a"),
}

REGIME_PERIODS: list[tuple[str, str, str]] = [
    ("2000-01-01", "2006-06-30", "Pré-Descoberta (2000–2006)"),
    ("2006-07-01", "2014-03-16", "Pré-Sal & Subprime (2006–2014)"),
    ("2014-03-17", "2016-05-29", "Crise & Lava Jato (2014–2016)"),
    ("2016-05-30", "2023-05-15", "Era PPI & COVID (2016–2023)"),
    ("2023-05-16", "2030-12-31", "Pós-PPI & Dividendos (2023–2026)"),
]


def fetch_bcb_sgs_chunked(
    series_id: int,
    start_year: int = 2000,
    end_year: int | None = None,
    timeout: int = 30,
) -> pd.Series:
    """Coleta uma série do SGS/BCB em blocos de até 10 anos para contornar o limite da API."""
    if end_year is None:
        end_year = pd.Timestamp.today().year

    frames: list[pd.DataFrame] = []
    for y0 in range(start_year, end_year + 1, 10):
        y1 = min(y0 + 9, end_year)
        url = BCB_SGS_URL.format(
            series_id=series_id,
            start=f"01/01/{y0}",
            end=f"31/12/{y1}",
        )
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload:
            frames.append(pd.DataFrame(payload))

    if not frames:
        raise RuntimeError(f"Nenhum dado retornado pela API do BCB para série {series_id}.")

    df = pd.concat(frames, ignore_index=True)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["data", "valor"]).drop_duplicates(subset=["data"])
    series = df.set_index("data")["valor"].sort_index()
    series.name = f"sgs_{series_id}"
    return series


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas MultiIndex retornadas por versões recentes do yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_yfinance_b3_safe(
    ticker: str,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Baixa dados do Yahoo Finance contornando o bug de horário de verão em 2004-11-02 (B3).

    Em `America/Sao_Paulo`, o dia 2004-11-02 (feriado de Finados) gera `NonExistentTimeError`
    no `yfinance` quando a janela cruza essa data. A coleta é dividida em dois blocos.
    """
    start_ts = pd.Timestamp(start_date)
    split_ts = pd.Timestamp("2004-11-02")

    parts: list[pd.DataFrame] = []
    if start_ts < split_ts:
        p1 = yf.download(
            ticker,
            start=start_date,
            end="2004-11-02",
            auto_adjust=False,
            progress=False,
        )
        if not p1.empty:
            parts.append(_flatten_yf_columns(p1))
        p2_start = "2004-11-03"
    else:
        p2_start = start_date

    p2 = yf.download(
        ticker,
        start=p2_start,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if not p2.empty:
        parts.append(_flatten_yf_columns(p2))

    if not parts:
        raise RuntimeError(f"Falha ao baixar {ticker} via yfinance.")

    out = pd.concat(parts).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    return out


def _load_legacy_brent(raw_dir: Path) -> pd.Series:
    """Carrega a série histórica diária de Brent (IPEADATA) salva em `data/raw/brent.xlsx`."""
    brent_path = raw_dir / "brent.xlsx"
    if not brent_path.exists():
        return pd.Series(dtype=float, name="brent_ipea")

    brent_xl = pd.read_excel(brent_path)
    date_col = "data" if "data" in brent_xl.columns else brent_xl.columns[0]
    val_col = "preço" if "preço" in brent_xl.columns else brent_xl.columns[1]
    brent_xl[date_col] = pd.to_datetime(brent_xl[date_col])
    brent_xl[val_col] = pd.to_numeric(brent_xl[val_col], errors="coerce")
    series = (
        brent_xl.dropna(subset=[date_col, val_col])
        .drop_duplicates(subset=[date_col])
        .set_index(date_col)[val_col]
        .sort_index()
    )
    series.name = "brent_ipea"
    return series


def _accumulate_selic_on_trading_calendar(
    selic_daily_pct: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Acumula a taxa Selic diária exatamente entre datas consecutivas de pregão da B3."""
    selic_log_daily = np.log1p(selic_daily_pct.sort_index() / 100.0)
    # Soma log-retornos em intervalos (prev_date, curr_date] via searchsorted
    s_dates = selic_log_daily.index.values
    cum_log = np.concatenate([[0.0], np.cumsum(selic_log_daily.values)])

    acc = np.zeros(len(trading_dates), dtype=float)
    for idx in range(1, len(trading_dates)):
        t_prev = np.datetime64(trading_dates[idx - 1])
        t_curr = np.datetime64(trading_dates[idx])
        i0 = np.searchsorted(s_dates, t_prev, side="right")
        i1 = np.searchsorted(s_dates, t_curr, side="right")
        if i1 > i0:
            acc[idx] = cum_log[i1] - cum_log[i0]
        else:
            # Fallback para último valor observado antes da data se houver lacuna pontual
            pos = max(0, i1 - 1)
            acc[idx] = selic_log_daily.iloc[pos]
    return pd.Series(acc, index=trading_dates, name="selic_daily_log")


def build_datasets(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Constrói (ou carrega em cache) as bases diária e mensal consolidadas de PETR4 + macro.

    Garante ausência de data leakage no alinhamento de calendários (apenas forward-fill causal)
    e exclui o pregão intradiário incompleto do próprio dia da execução.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    proc_path.mkdir(parents=True, exist_ok=True)

    daily_file = proc_path / "petr4_macro_daily.csv"
    monthly_file = proc_path / "petr4_macro_monthly.csv"

    if not force_download and daily_file.exists() and monthly_file.exists():
        df_daily = pd.read_csv(daily_file, parse_dates=["date"], index_col="date")
        df_monthly = pd.read_csv(monthly_file, parse_dates=["date"], index_col="date")
        return df_daily, df_monthly

    # 1. Coleta Yahoo Finance
    petr_raw = fetch_yfinance_b3_safe("PETR4.SA", start_date="2000-01-01")
    ibov_raw = fetch_yfinance_b3_safe("^BVSP", start_date="2000-01-01")
    bz_raw = fetch_yfinance_b3_safe("BZ=F", start_date="2007-07-01")

    # 2. Coleta Brent IPEADATA + BCB SGS
    brent_ipea = _load_legacy_brent(raw_path)
    brent_full = bz_raw["Close"].combine_first(brent_ipea).sort_index()

    usd_brl_raw = fetch_bcb_sgs_chunked(1, start_year=2000)
    selic_daily_raw = fetch_bcb_sgs_chunked(11, start_year=2000)
    selic_meta_raw = fetch_bcb_sgs_chunked(432, start_year=2000)

    # Salvar snapshots brutos atualizados em data/raw/
    petr_raw.to_csv(raw_path / "petr4_yf_raw.csv")
    ibov_raw.to_csv(raw_path / "ibov_yf_raw.csv")
    brent_full.to_frame("brent_usd").to_csv(raw_path / "brent_combined_raw.csv")
    pd.concat(
        [
            usd_brl_raw.rename("usd_brl"),
            selic_daily_raw.rename("selic_daily_pct"),
            selic_meta_raw.rename("selic_annual_pct"),
        ],
        axis=1,
    ).to_csv(raw_path / "bcb_sgs_raw.csv")

    # 3. Alinhamento causal pelo calendário fechado da B3 (excluindo sessão intradiária corrente)
    last_closed_bcb = min(usd_brl_raw.index.max(), selic_daily_raw.index.max())
    today_norm = pd.Timestamp.today().normalize()
    cutoff_date = min(last_closed_bcb, today_norm - pd.Timedelta(days=1))

    petr_cal = petr_raw.loc[petr_raw.index <= cutoff_date].copy()
    trading_idx = petr_cal.index

    daily = pd.DataFrame(index=trading_idx)
    daily.index.name = "date"
    daily["petr4_close"] = petr_cal["Close"].astype(float)
    daily["petr4_adj_close"] = petr_cal["Adj Close"].astype(float)
    daily["petr4_volume"] = petr_cal["Volume"].astype(float)

    # Apenas forward-fill causal em calendários distintos (sem bfill ou média global)
    daily["ibov_close"] = ibov_raw["Close"].reindex(trading_idx).ffill()
    daily["brent_usd"] = brent_full.reindex(brent_full.index.union(trading_idx)).sort_index().ffill().reindex(trading_idx)
    daily["usd_brl"] = usd_brl_raw.reindex(usd_brl_raw.index.union(trading_idx)).sort_index().ffill().reindex(trading_idx)
    daily["selic_daily_pct"] = selic_daily_raw.reindex(selic_daily_raw.index.union(trading_idx)).sort_index().ffill().reindex(trading_idx)
    daily["selic_annual_pct"] = selic_meta_raw.reindex(selic_meta_raw.index.union(trading_idx)).sort_index().ffill().reindex(trading_idx)

    # Remove apenas eventual dia inicial antes do início da cobertura de alguma série (ex: 2000-01-03 no Brent)
    core_cols = [
        "petr4_close",
        "petr4_adj_close",
        "ibov_close",
        "brent_usd",
        "usd_brl",
        "selic_daily_pct",
        "selic_annual_pct",
    ]
    daily = daily.dropna(subset=core_cols).copy()
    daily["brent_brl"] = daily["brent_usd"] * daily["usd_brl"]

    # Log-preços e log-retornos diários
    daily["log_petr4_close"] = np.log(daily["petr4_close"])
    daily["log_petr4_adj"] = np.log(daily["petr4_adj_close"])
    daily["ret_petr4"] = np.log(daily["petr4_adj_close"] / daily["petr4_adj_close"].shift(1))
    daily["ret_petr4_nominal"] = np.log(daily["petr4_close"] / daily["petr4_close"].shift(1))
    daily["ret_ibov"] = np.log(daily["ibov_close"] / daily["ibov_close"].shift(1))
    daily["ret_brent_usd"] = np.log(daily["brent_usd"] / daily["brent_usd"].shift(1))
    daily["ret_brent_brl"] = np.log(daily["brent_brl"] / daily["brent_brl"].shift(1))
    daily["ret_usd_brl"] = np.log(daily["usd_brl"] / daily["usd_brl"].shift(1))

    daily["selic_daily_log"] = _accumulate_selic_on_trading_calendar(selic_daily_raw, daily.index)
    daily["excess_petr4"] = daily["ret_petr4"] - daily["selic_daily_log"]
    daily["excess_ibov"] = daily["ret_ibov"] - daily["selic_daily_log"]

    # Remover primeira linha (retorno NaN) para garantir base 100% íntegra nas colunas principais
    daily = daily.dropna(subset=["ret_petr4", "ret_ibov", "ret_brent_usd", "ret_usd_brl"]).copy()

    # Volatilidade realizada anualizada (252 pregões/ano)
    daily["vol_22d"] = daily["ret_petr4"].rolling(22).std() * np.sqrt(252)
    daily["vol_252d"] = daily["ret_petr4"].rolling(252).std() * np.sqrt(252)

    # 4. Agregação Mensal (fim de mês) evitando viés de mês corrente incompleto
    month_counts = daily.groupby(daily.index.to_period("M")).size()
    last_period = daily.index[-1].to_period("M")
    daily_for_monthly = daily
    if month_counts.loc[last_period] < 15 and daily.index[-1].day < 25:
        daily_for_monthly = daily.loc[daily.index.to_period("M") < last_period].copy()

    monthly = pd.DataFrame(index=daily_for_monthly.resample("ME").last().index)
    monthly.index.name = "date"
    monthly["petr4_close"] = daily_for_monthly["petr4_close"].resample("ME").last()
    monthly["petr4_adj_close"] = daily_for_monthly["petr4_adj_close"].resample("ME").last()
    monthly["ibov_close"] = daily_for_monthly["ibov_close"].resample("ME").last()
    monthly["brent_usd"] = daily_for_monthly["brent_usd"].resample("ME").last()
    monthly["usd_brl"] = daily_for_monthly["usd_brl"].resample("ME").last()
    monthly["brent_brl"] = daily_for_monthly["brent_brl"].resample("ME").last()
    monthly["selic_annual_pct"] = daily_for_monthly["selic_annual_pct"].resample("ME").last()
    monthly["petr4_volume_mean"] = daily_for_monthly["petr4_volume"].resample("ME").mean()

    monthly["log_petr4_close"] = np.log(monthly["petr4_close"])
    monthly["log_petr4_adj"] = np.log(monthly["petr4_adj_close"])
    monthly["ret_petr4"] = np.log(monthly["petr4_adj_close"] / monthly["petr4_adj_close"].shift(1))
    monthly["ret_petr4_nominal"] = np.log(monthly["petr4_close"] / monthly["petr4_close"].shift(1))
    monthly["ret_ibov"] = np.log(monthly["ibov_close"] / monthly["ibov_close"].shift(1))
    monthly["ret_brent_usd"] = np.log(monthly["brent_usd"] / monthly["brent_usd"].shift(1))
    monthly["ret_brent_brl"] = np.log(monthly["brent_brl"] / monthly["brent_brl"].shift(1))
    monthly["ret_usd_brl"] = np.log(monthly["usd_brl"] / monthly["usd_brl"].shift(1))

    # Acúmulo exato dos log-retornos diários da Selic dentro de cada mês fechado
    monthly["selic_monthly_log"] = daily_for_monthly["selic_daily_log"].resample("ME").sum()
    monthly["selic_monthly_pct"] = np.expm1(monthly["selic_monthly_log"]) * 100.0
    monthly["excess_petr4"] = monthly["ret_petr4"] - monthly["selic_monthly_log"]
    monthly["excess_ibov"] = monthly["ret_ibov"] - monthly["selic_monthly_log"]
    monthly["vol_monthly_ann"] = daily_for_monthly["ret_petr4"].resample("ME").std() * np.sqrt(252)

    monthly = monthly.dropna().copy()

    daily.to_csv(daily_file)
    monthly.to_csv(monthly_file)
    return daily, monthly


def assign_regime(index: Iterable[pd.Timestamp]) -> pd.Series:
    """Classifica cada data no regime político-econômico correspondente da Petrobras."""
    idx = pd.DatetimeIndex(index)
    labels = pd.Series(index=idx, dtype="object")
    for start, end, label in REGIME_PERIODS:
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        labels.loc[mask] = label
    return labels


if __name__ == "__main__":
    d_df, m_df = build_datasets(force_download=True)
    print(f"Base diária: {d_df.shape} | {d_df.index.min().date()} -> {d_df.index.max().date()}")
    print(f"Base mensal: {m_df.shape} | {m_df.index.min().date()} -> {m_df.index.max().date()}")
