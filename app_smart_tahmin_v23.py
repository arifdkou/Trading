import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, date
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import matplotlib.pyplot as plt
from typing import Optional
import io

# ---------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def download_prices(symbols, start, end):
    """
    Çoklu sembolde yfinance DataFrame'i her zaman MultiIndex döndürür:
        Level 0 = OHLC kolonları ('Open','Close','Adj Close',...),
        Level 1 = Semboller.
    Burada 'Adj Close' varsa onu, yoksa 'Close'u seçiyoruz.
    """

    try:
        data = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"Veri indirilemedi: {e}")

    if data.empty:
        raise RuntimeError("yfinance boş veri döndürdü")

    # MultiIndex kolon
    if isinstance(data.columns, pd.MultiIndex):

        level0 = list(data.columns.levels[0])  # OHLC kolonları

        if "Adj Close" in level0:
            px = data.xs("Adj Close", axis=1, level=0)
        elif "Close" in level0:
            px = data.xs("Close", axis=1, level=0)
        else:
            raise RuntimeError(
                f"MultiIndex kolonunda ne 'Adj Close' ne de 'Close' bulunamadı.\n"
                f"Mevcut kolonlar: {level0}"
            )
    else:
        # Tek sembol
        cols = data.columns
        if "Adj Close" in cols:
            px = data[["Adj Close"]]
        elif "Close" in cols:
            px = data[["Close"]]
        else:
            raise RuntimeError(f"Tek sembolde Close kolonu yok. Kolonlar: {cols}")

        sym = symbols[0] if isinstance(symbols, list) else symbols
        px.columns = [sym]

    px = px.dropna()
    if px.empty:
        raise RuntimeError("Close fiyatları boş. Tarih aralığını genişletin.")

    return px


@st.cache_data(show_spinner=False)
def download_ohlc(symbol, start, end):
    """
    ATR hesabı ve mum grafiği için OHLC verisini indirir.
    MultiIndex veya tek seviye kolon yapısına göre kolonları çeker.
    """
    try:
        data = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"OHLC veri indirilemedi: {e}")

    if data.empty:
        raise RuntimeError("OHLC verisi boş döndü.")

    # MultiIndex kolon (('High','XPEV'), ... gibi)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            open_ = data[("Open", symbol)]
            high = data[("High", symbol)]
            low = data[("Low", symbol)]
            close = data[("Close", symbol)]
        except Exception:
            raise RuntimeError(
                f"OHLCV kolonları MultiIndex ama beklenen formatta değil. "
                f"Gelen kolonlar: {list(data.columns)}"
            )
    else:
        # Tek seviye kolon
        needed = {"Open", "High", "Low", "Close"}
        if not needed.issubset(set(data.columns)):
            raise RuntimeError(
                f"OHLC için Open/High/Low/Close kolonları yok. Gelen kolonlar: {list(data.columns)}"
            )
        open_ = data["Open"]
        high = data["High"]
        low = data["Low"]
        close = data["Close"]

    ohlc = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}
    ).dropna()

    return ohlc


def compute_atr(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Klasik ATR(14).
    """
    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def compute_returns(price_df):
    """Doğal log getirileri."""
    ret = np.log(price_df / price_df.shift(1)).dropna()
    return ret


def zlema(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Zero-Lag EMA (ZLEMA) hesaplar.
    length: periyot
    """
    series = series.dropna()
    if series.empty:
        return pd.Series(dtype=float)

    lag = (length - 1) // 2
    shifted = series + (series - series.shift(lag))
    zlema_val = shifted.ewm(span=length, adjust=False).mean()
    return zlema_val


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Basit rolling z-skoru: (x - mean) / std
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / (std + 1e-9)
    return z


def compute_weekly_liquidity(
    price_series: pd.Series,
    n_levels: int = 12,
    week_freq: str = "W-FRI",
):
    """
    Günlük fiyat serisinden haftalık buy/sell likidite seviyelerini çıkarır.

    - Haftalık dipler  → Buy likidite
    - Haftalık tepeler → Sell likidite

    Dönenler:
      liq_df       : Buy/Sell seviyelerini içeren tablo (max n_levels)
      nearest_buy  : Fiyata en yakın buy likidite
      nearest_sell : Fiyata en yakın sell likidite
      weekly_low   : Haftalık dip serisi (tam)
      weekly_high  : Haftalık tepe serisi (tam)
    """
    if price_series.empty:
        empty = pd.DataFrame(columns=["Tür", "Hafta Sonu", "Seviye"])
        return empty, np.nan, np.nan, pd.Series(dtype=float), pd.Series(dtype=float)

    weekly_high = price_series.resample(week_freq).max().iloc[:-1]
    weekly_low = price_series.resample(week_freq).min().iloc[:-1]

    if weekly_high.empty or weekly_low.empty:
        empty = pd.DataFrame(columns=["Tür", "Hafta Sonu", "Seviye"])
        return empty, np.nan, np.nan, weekly_low, weekly_high

    current_price = float(price_series.iloc[-1])

    # Fiyata göre aşağı/yukarı filtrelenmiş, son n_levels hafta
    buy_levels = weekly_low[weekly_low < current_price].tail(n_levels)
    sell_levels = weekly_high[weekly_high > current_price].tail(n_levels)

    df_buy = pd.DataFrame(
        {
            "Tür": "Buy likidite",
            "Hafta Sonu": buy_levels.index.date,
            "Seviye": buy_levels.values,
        }
    )

    df_sell = pd.DataFrame(
        {
            "Tür": "Sell likidite",
            "Hafta Sonu": sell_levels.index.date,
            "Seviye": sell_levels.values,
        }
    )

    liq_df = pd.concat([df_buy, df_sell], ignore_index=True)
    if not liq_df.empty:
        liq_df = liq_df.sort_values("Seviye").reset_index(drop=True)

    nearest_buy = buy_levels.iloc[-1] if len(buy_levels) > 0 else np.nan
    nearest_sell = sell_levels.iloc[0] if len(sell_levels) > 0 else np.nan

    return liq_df, nearest_buy, nearest_sell, weekly_low, weekly_high


# === 12 Haftalık Döngü Eğrisi Hesabı ================================
def compute_12w_cycle_curve(price_series: pd.Series, cycle_days: int = 60):
    """
    Son 'cycle_days' günlük fiyatı alır (varsayılan ~60 gün ≈ 12 hafta),
    0–1 aralığına normalize eder ve döngü eğrisini döndürür.
    """
    if price_series is None or price_series.empty:
        return None, None

    window = min(cycle_days, len(price_series))
    sub = price_series.iloc[-window:]

    mn, mx = sub.min(), sub.max()
    if mx - mn < 1e-9:
        cycle_curve = pd.Series(0.5, index=sub.index)
    else:
        cycle_curve = (sub - mn) / (mx - mn)

    return sub, cycle_curve


# === Likidite Kümesi + Spring (Sweep) Tespiti =======================
def detect_liquidity_patterns(
    ohlc: pd.DataFrame,
    weekly_low: pd.Series,
    weekly_high: pd.Series,
    atr_last: float,
    current_price: float,
    lookback_days: int = 10,
    cluster_window_weeks: int = 12,
):
    """
    - Buy & Sell haftalık seviyelerin birbirine yakın olduğu bir 'likidite kümesi'
      (cluster) bulur.
    - Son lookback_days içindeki fiyatın bu kümenin altına süpürme (spring)
      yapıp yapmadığını kontrol eder.
    """
    info = {
        "cluster_low": np.nan,
        "cluster_high": np.nan,
        "down_sweep": False,
        "comment": "Likidite pattern analizi için yeterli veri yok.",
    }

    if (
        ohlc is None
        or ohlc.empty
        or weekly_low.empty
        or weekly_high.empty
        or np.isnan(atr_last)
    ):
        return info

    # Son cluster_window_weeks haftadaki seviyeler
    buy_last = weekly_low.tail(cluster_window_weeks)
    sell_last = weekly_high.tail(cluster_window_weeks)

    if buy_last.empty or sell_last.empty:
        return info

    # Buy ve Sell seviyeleri arasından ATR mesafesi içinde olan çiftleri ara
    pairs = []
    for b in buy_last.values:
        nearby_sells = sell_last[(sell_last >= b - atr_last) & (sell_last <= b + atr_last)]
        for s in nearby_sells.values:
            low_ = min(b, s)
            high_ = max(b, s)
            center = 0.5 * (b + s)
            dist_to_curr = abs(center - current_price)
            pairs.append((dist_to_curr, low_, high_))

    if not pairs:
        info["comment"] = "Son haftalarda belirgin bir buy/sell likidite kümesi bulunamadı."
        return info

    # Fiyata en yakın cluster'ı seç
    pairs.sort(key=lambda x: x[0])
    _, cluster_low, cluster_high = pairs[0]
    info["cluster_low"] = cluster_low
    info["cluster_high"] = cluster_high

    # Son lookback_days içindeki en düşük ve en yüksek değerler
    recent = ohlc.tail(lookback_days)
    recent_low = recent["Low"].min()

    # Kümenin ALTINA doğru süpürme var mı?
    down_sweep = recent_low < (cluster_low - 0.2 * atr_last)

    info["down_sweep"] = bool(down_sweep)

    if down_sweep and current_price > recent_low:
        info["comment"] = (
            "Fiyat, buy/sell likidite kümesinin ALTINA bir kez fitil atmış ve "
            "sonrasında üzerine geri dönmüş görünüyor. Bu yapı klasik bir "
            "'liquidity sweep / spring' formasyonuna benziyor (dip bölgesi olasılığı ↑)."
        )
    elif down_sweep:
        info["comment"] = (
            "Fiyat, likidite kümesinin altına sarkmış durumda ve henüz toparlanmamış. "
            "Likidite süpürmesi devam ediyor olabilir; dip oluşumu tamamlanmamış olabilir."
        )
    else:
        info["comment"] = (
            "Son günlerde likidite kümesinin altında net bir süpürme görülmüyor. "
            "Fiyat daha çok küme içinde veya üzerinde işlem görüyor."
        )

    return info


def detect_fvgs(ohlc: pd.DataFrame, lookahead: int = 200) -> pd.DataFrame:
    """
    3 mumluk basit FVG tespiti.
    - Bullish FVG: High[i-1] < Low[i+1]
    - Bearish FVG: Low[i-1] > High[i+1]
    'Açık' FVG: Oluştuktan sonra ileriye doğru bakıldığında
       fiyat o gap bölgesine HİÇ dokunmamış olsun.
    lookahead: FVG oluşumundan sonra en fazla kaç bar ileriye bakılacağı
    """
    if ohlc is None or ohlc.empty:
        return pd.DataFrame(columns=["Tip", "Tarih", "AltSeviye", "ÜstSeviye"])

    highs = ohlc["High"]
    lows = ohlc["Low"]
    idx = ohlc.index

    records = []
    n = len(ohlc)

    for i in range(1, n - 1):
        # i-1, i, i+1 mumu kullanıyoruz
        h1 = float(highs.iloc[i - 1])
        l1 = float(lows.iloc[i - 1])
        h3 = float(highs.iloc[i + 1])
        l3 = float(lows.iloc[i + 1])
        t_mid = idx[i]

        # Bullish FVG (gap aşağıda)
        if h1 < l3:
            gap_low = h1
            gap_high = l3
            # ileri bak: gap'e LOW ile dokunan var mı?
            future_lows = lows.iloc[i + 2: i + 2 + lookahead]
            touched = (future_lows <= gap_low).any()
            if not touched:
                records.append(
                    {
                        "Tip": "Bullish FVG",
                        "Tarih": t_mid,
                        "AltSeviye": gap_low,
                        "ÜstSeviye": gap_high,
                    }
                )

        # Bearish FVG (gap yukarıda)
        if l1 > h3:
            gap_low = h3
            gap_high = l1
            # ileri bak: gap'e HIGH ile dokunan var mı?
            future_highs = highs.iloc[i + 2: i + 2 + lookahead]
            touched = (future_highs >= gap_high).any()
            if not touched:
                records.append(
                    {
                        "Tip": "Bearish FVG",
                        "Tarih": t_mid,
                        "AltSeviye": gap_low,
                        "ÜstSeviye": gap_high,
                    }
                )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Tarih").reset_index(drop=True)
    return df


def detect_order_blocks(
    ohlc: pd.DataFrame,
    atr: Optional[pd.Series] = None,
    lookahead: int = 15,
) -> pd.DataFrame:
    """
    Basitleştirilmiş Order Block tespiti:
    - Bullish OB: Güçlü yukarı impuls öncesi son düşüş mumu
    - Bearish OB: Güçlü aşağı impuls öncesi son yükseliş mumu

    'Açık' OB: impuls sonrası mumlar bu bölgeye hiç geri dönmemiş olsun.
    Bölge: [Low, High] olarak alınıyor.
    """
    if ohlc is None or ohlc.empty:
        return pd.DataFrame(columns=["Tip", "Tarih", "AltSeviye", "ÜstSeviye"])

    opens = ohlc["Open"]
    highs = ohlc["High"]
    lows = ohlc["Low"]
    closes = ohlc["Close"]
    idx = ohlc.index

    records = []
    n = len(ohlc)

    for i in range(0, n - lookahead - 1):
        o = float(opens.iloc[i])
        h = float(highs.iloc[i])
        l = float(lows.iloc[i])
        c = float(closes.iloc[i])
        t = idx[i]

        # ATR tabanlı minimum impuls eşiği
        if atr is not None and t in atr.index and not np.isnan(atr.loc[t]):
            thr = 0.5 * float(atr.loc[t])  # yarım ATR üstüne kırılım şartı
        else:
            thr = 0.0

        # ----- Bullish Order Block (önce düşüş mumu, sonra güçlü breakout yukarı) -----
        if c < o:  # düşüş mumu
            future_high = float(highs.iloc[i + 1: i + 1 + lookahead].max())
            if future_high >= h + thr:  # yukarı impuls
                zone_low = l
                zone_high = h

                # impuls sonrası tüm barlarda bu bölgeye LOW/HIGH ile dokunulmuş mu?
                future_lows = lows.iloc[i + 1 + lookahead:]
                future_highs = highs.iloc[i + 1 + lookahead:]

                touched = (
                        ((future_lows <= zone_high) & (future_lows >= zone_low)).any()
                        or ((future_highs <= zone_high) & (future_highs >= zone_low)).any()
                )

                if not touched:
                    records.append(
                        {
                            "Tip": "Bullish OB",
                            "Tarih": t,
                            "AltSeviye": zone_low,
                            "ÜstSeviye": zone_high,
                        }
                    )

        # ----- Bearish Order Block (önce yükseliş mumu, sonra güçlü breakdown aşağı) -----
        if c > o:  # yükseliş mumu
            future_low = float(lows.iloc[i + 1: i + 1 + lookahead].min())
            if future_low <= l - thr:  # aşağı impuls
                zone_low = l
                zone_high = h

                future_lows = lows.iloc[i + 1 + lookahead:]
                future_highs = highs.iloc[i + 1 + lookahead:]

                touched = (
                        ((future_lows <= zone_high) & (future_lows >= zone_low)).any()
                        or ((future_highs <= zone_high) & (future_highs >= zone_low)).any()
                )

                if not touched:
                    records.append(
                        {
                            "Tip": "Bearish OB",
                            "Tarih": t,
                            "AltSeviye": zone_low,
                            "ÜstSeviye": zone_high,
                        }
                    )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("Tarih").reset_index(drop=True)
    return df


def build_imbalance_table(
    ohlc: pd.DataFrame,
    atr: Optional[pd.Series],
    current_price: float,
    max_rows: int = 50,
) -> pd.DataFrame:
    """
    Açık FVG + Açık Order Block bölgelerini tek tabloda birleştirir.
    Fiyata en yakın olanlar üstte olacak şekilde sıralar.
    """
    fvg_df = detect_fvgs(ohlc, lookahead=200)
    ob_df = detect_order_blocks(ohlc, atr=atr, lookahead=15)

    all_df = (
        pd.concat([fvg_df, ob_df], ignore_index=True)
        if (not fvg_df.empty or not ob_df.empty)
        else pd.DataFrame(columns=["Tip", "Tarih", "AltSeviye", "ÜstSeviye"])
    )

    if all_df.empty:
        return all_df

    all_df["MerkezSeviye"] = (all_df["AltSeviye"] + all_df["ÜstSeviye"]) / 2.0
    all_df["Fiyata_Uzaklık_%"] = (all_df["MerkezSeviye"] - current_price) / current_price * 100.0

    all_df = all_df.sort_values("Fiyata_Uzaklık_%", key=lambda s: np.abs(s))
    all_df = all_df.head(max_rows).reset_index(drop=True)

    return all_df


def fit_arimax(main_ret, exog):
    """ARIMAX(1,0,1) modeli."""
    model = SARIMAX(
        main_ret,
        order=(1, 0, 1),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    last_exog = exog.iloc[-1:]
    forecast = res.forecast(steps=1, exog=last_exog)
    mu_daily = float(forecast.iloc[0])

    return res, mu_daily


def fit_garch(residuals):
    """GARCH(1,1) ile volatilite tahmini."""
    scaled = residuals * 100.0
    am = arch_model(scaled, p=1, q=1, mean="Zero", vol="GARCH", dist="normal")
    garch_res = am.fit(disp="off")
    fc = garch_res.forecast(horizon=1)
    var_daily = fc.variance.values[-1, 0]
    sigma_daily = float(np.sqrt(var_daily)) / 100.0
    return garch_res, sigma_daily


def monte_carlo_paths(S0, mu, sigma, horizon, n_paths, seed=42):
    """GBM ile fiyat patikaları."""
    S0 = float(S0)
    mu = float(mu)
    sigma = float(sigma)

    np.random.seed(seed)
    dt = 1.0

    z = np.random.normal(size=(horizon, n_paths))
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    log_price = np.log(S0) + np.cumsum(log_returns, axis=0)
    paths = np.exp(log_price)
    return paths


def calc_smart_money_metrics(prices, rets, main_symbol, spy_sym, kweb_sym):
    """
    Smart Money Index, 12 haftalık cycle pozisyonu,
    SPY & KWEB betaları ve spread z-score.
    """
    main_p = prices[main_symbol]
    main_r = rets[main_symbol]

    # Classic SMI: 20 günlük rolling mean'in z-skoru
    roll = main_r.rolling(20)
    classic_smi = (roll.mean() - roll.mean().mean()) / (roll.std().mean() + 1e-9)
    classic_smi = classic_smi.dropna()

    if classic_smi.empty:
        smi_today = 0.0
    else:
        smi_today = float(classic_smi.iloc[-1])

    # 12 haftalık ~60 gün cycle pozisyonu
    window = min(60, len(main_p))
    sub = main_p.iloc[-window:]
    mn, mx = sub.min(), sub.max()
    if mx - mn < 1e-9:
        cycle_pos = 0.5
    else:
        cycle_pos = float((sub.iloc[-1] - mn) / (mx - mn))

    # Betalar
    if spy_sym in rets.columns:
        spy_r = rets[spy_sym]
    else:
        spy_r = main_r * 0.0

    if kweb_sym in rets.columns:
        kweb_r = rets[kweb_sym]
    else:
        kweb_r = main_r * 0.0

    def beta(x, y):
        if y.var() < 1e-9:
            return 0.0
        return float(np.cov(x, y)[0, 1] / y.var())

    beta_spy = beta(main_r, spy_r)
    beta_kweb = beta(main_r, kweb_r)

    # Spread z-score (KWEB yoksa ana sembolün kendisi ile)
    if kweb_sym in prices.columns:
        kweb_p = prices[kweb_sym]
    else:
        kweb_p = prices[main_symbol] * 0.0 + prices[main_symbol].iloc[0]

    spread = prices[main_symbol] - beta_kweb * kweb_p
    spread_z_series = (spread - spread.rolling(60).mean()) / (
        spread.rolling(60).std() + 1e-9
    )
    spread_z = (
        float(spread_z_series.iloc[-1])
        if not spread_z_series.dropna().empty
        else 0.0
    )

    return smi_today, cycle_pos, beta_spy, beta_kweb, spread_z, classic_smi


def path_probabilities(paths, entry, stop, tp1, tp2, side="Long"):
    """Monte Carlo patikalarından TP / Stop olasılıkları."""
    min_path = paths.min(axis=0)
    max_path = paths.max(axis=0)

    if side == "Long":
        hit_stop = min_path <= stop
        hit_tp1 = max_path >= tp1
        hit_tp2 = max_path >= tp2
    else:
        hit_stop = max_path >= stop
        hit_tp1 = min_path <= tp1
        hit_tp2 = min_path <= tp2

    n = paths.shape[1]
    p_stop = hit_stop.mean()
    p_tp1 = hit_tp1.mean()
    p_tp2 = hit_tp2.mean()
    p_none = max(0.0, 1.0 - min(1.0, p_stop + p_tp1 + p_tp2))

    return p_tp1, p_tp2, p_stop, p_none


def kelly_from_probs(p_tp1, p_tp2, p_stop, p_none, entry, stop, tp1, tp2, side="Long"):
    """Olasılıklardan R biriminde EV ve Kelly."""
    risk_per_share = abs(entry - stop)
    if risk_per_share < 1e-9:
        return 0.0, 0.0, 0.0

    if side == "Long":
        R_tp1 = (tp1 - entry) / risk_per_share
        R_tp2 = (tp2 - entry) / risk_per_share
    else:
        R_tp1 = (entry - tp1) / risk_per_share
        R_tp2 = (entry - tp2) / risk_per_share

    R_stop = -1.0
    R_none = 0.0

    probs = np.array([p_tp1, p_tp2, p_stop, p_none])
    Rs = np.array([R_tp1, R_tp2, R_stop, R_none])

    mu_R = float(np.sum(probs * Rs))
    var_R = float(np.sum(probs * (Rs - mu_R) ** 2))

    if var_R <= 0 or mu_R <= 0:
        kelly = 0.0
    else:
        kelly = mu_R / var_R

    return mu_R, var_R, kelly


def series_slope(series: pd.Series, window: int = 20) -> float:
    """Son window bar için lineer regresyon eğimi."""
    if series is None:
        return 0.0
    s = series.dropna()
    if len(s) < window:
        return 0.0
    y = s.iloc[-window:]
    x = np.arange(len(y))
    return float(np.polyfit(x, y.values, 1)[0])


def detect_phase_from_indicators(
    close_px: pd.Series,
    vwma: pd.Series,
    vwma_z: pd.Series,
    close_minus_vwma_z: pd.Series,
    obv: pd.Series,
):
    """
    Basit faz tespiti:
      - Akümülasyon / Markup / Distribusyon / Markdown / Yatay
    """
    try:
        last_close = float(close_px.iloc[-1])
        last_vw = float(vwma.iloc[-1])
        last_vwz = float(vwma_z.iloc[-1])
        last_cmvz = float(close_minus_vwma_z.iloc[-1])
    except Exception:
        return "Faz tespit edilemedi", "Göstergeler için yeterli veri yok."

    obv_s = series_slope(obv, window=30)

    # Kurallar
    if last_close > last_vw and last_vwz > 1.0 and last_cmvz > 1.0 and obv_s > 0:
        phase = "Markup (Yükseliş Trend Fazı)"
        comment = (
            "Fiyat VWMA(66)'nın üzerinde, VWMA Z-skoru ve Close–VWMA Z-skoru pozitif ve "
            "OBV yukarı eğimli. Bu yapı güçlü bir **trend yükselişi / markup** fazına işaret eder. "
            "Smart money genelde bu fazda pozisyon azaltmaz, trendi sürdürür."
        )
    elif last_close > last_vw and last_cmvz > -0.5 and 0.0 <= last_vwz <= 1.0 and obv_s >= 0:
        phase = "Akümülasyon / Erken Yükseliş"
        comment = (
            "Fiyat VWMA(66)'nın üzerinde veya yakınında, VWMA Z-skoru hafif pozitif, "
            "Close–VWMA Z-skoru çok gerilmemiş ve OBV en azından yatay/pozitif. "
            "Bu yapı **akümülasyon veya erken yükseliş** fazını destekler; "
            "kurumsal taraf yavaş yavaş toplanıyor olabilir."
        )
    elif last_close < last_vw and last_vwz > 0.5 and obv_s <= 0:
        phase = "Distribusyon (Dağıtım Fazı)"
        comment = (
            "Fiyat VWMA(66)'nın altına kaymış, VWMA Z-skoru hâlâ pozitif tarafta ve "
            "OBV eğimi zayıf / negatif. Bu genellikle **dağıtım fazı** ile uyumludur; "
            "önceki yükselişte alınan pozisyonlar kademeli olarak boşaltılıyor olabilir."
        )
    elif last_close < last_vw and last_vwz < -0.5 and last_cmvz < -1.0 and obv_s < 0:
        phase = "Markdown (Düşüş Trend Fazı)"
        comment = (
            "Fiyat VWMA(66)'nın altında, VWMA Z-skoru ve Close–VWMA Z-skoru negatif bölgede "
            "ve OBV aşağı eğimli. Bu yapı **markdown / düşüş trendi** karakterlidir; "
            "trend aşağı, ralliler satış fırsatı olarak kullanılabilir."
        )
    else:
        phase = "Yatay / Geçiş Fazı"
        comment = (
            "VWMA ve Z-skorlar belirgin bir aşırı alım / aşırı satım göstermiyor veya "
            "OBV ile uyumsuz. Bu durum çoğunlukla **yatay band / geçiş** fazına işaret eder; "
            "smart money net bir yön seçmemiş olabilir."
        )

    return phase, comment


def auto_comment(
    side,
    horizon,
    median_price,
    q10,
    q90,
    p_tp1,
    p_tp2,
    p_stop,
    kelly,
    entry,
    stop,
    tp1,
    tp2,
):
    """Ekranın altındaki otomatik özet."""
    direction = "LONG" if side == "Long" else "SHORT"
    risk_reward_1 = abs(tp1 - entry) / abs(entry - stop)
    risk_reward_2 = abs(tp2 - entry) / abs(entry - stop)

    msg = f"""
### 📌 Otomatik Özet – {direction} Senaryosu

- **Tahmin Ufku:** {horizon} gün  
- **Medyan Fiyat:** ≈ **{median_price:.2f} USD**  
- **%10–%90 Bantları:** **{q10:.2f} – {q90:.2f} USD**  

- **TP1'e en az bir kez dokunma olasılığı:** ≈ **{p_tp1*100:.1f}%**  
- **TP2'ye en az bir kez dokunma olasılığı:** ≈ **{p_tp2*100:.1f}%**  
- **Stop'a en az bir kez dokunma olasılığı:** ≈ **{p_stop*100:.1f}%**  

- **R:R (Entry→TP1 / Entry→TP2):** ≈ **{risk_reward_1:.2f} / {risk_reward_2:.2f}**  

- **Yaklaşık Kelly oranı:** ≈ **{kelly*100:.2f}%**  
  (Pratikte bunun **yarısı veya daha azı** genelde tercih edilir.)
"""
    return msg


# ---------------------------------------------------------------------
# Streamlit Arayüzü
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Money Forecast v3",
    layout="wide",
)

st.sidebar.header("⚙️ Parametreler")

main_symbol = st.sidebar.text_input("Ana sembol", value="XPEV")
position_side = st.sidebar.selectbox("Pozisyon yönü", ["Long", "Short"])
start_date = st.sidebar.date_input(
    "Veri başlangıç tarihi", value=date(2021, 1, 1), min_value=date(2005, 1, 1)
)
horizon = st.sidebar.slider("Tahmin ufku (gün)", min_value=10, max_value=180, value=60)

st.sidebar.markdown("---")
st.sidebar.header("💣 Risk Yönetimi")

account_size = st.sidebar.number_input(
    "Hesap büyüklüğü (USD)", min_value=1000.0, value=100000.0, step=1000.0
)
risk_pct = st.sidebar.slider("Trade başına risk (%)", 0.1, 10.0, 2.0, 0.1)
n_paths = st.sidebar.slider(
    "Monte Carlo senaryo sayısı",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000,
)

run_button = st.sidebar.button("🚀 Tahmin Yap")

st.title(
    "📈 Smart Money Forecast v3 – ARIMAX + GARCH + Monte Carlo + Smart Money + Kelly"
)

if not run_button:
    st.info("Soldaki parametreleri ayarlayıp **'Tahmin Yap'** butonuna bas.")
    st.stop()

# ---------------------------------------------------------------------
# Veri indir & hazırlık
# ---------------------------------------------------------------------
symbols = [main_symbol, "SPY", "^VIX", "KWEB"]
end_date = datetime.today().date()

try:
    prices = download_prices(symbols, start_date, end_date)
except Exception as e:
    st.error(f"Veri indirilemedi: {e}")
    st.stop()

missing_cols = [s for s in symbols if s not in prices.columns]
if missing_cols:
    st.warning(
        f"Bu semboller veri setinde bulunamadı: {missing_cols}. Yine de devam ediyorum."
    )
    symbols = [s for s in symbols if s in prices.columns]
    if main_symbol not in symbols:
        st.error("Ana sembol için fiyat verisi yok, işlem yapılamaz.")
        st.stop()

rets = compute_returns(prices)

main_ret = rets[main_symbol]
spy_ret = rets.get("SPY")
vix_ret = rets.get("^VIX")
kweb_ret = rets.get("KWEB")

exog_list, exog_names = [], []
if spy_ret is not None:
    exog_list.append(spy_ret)
    exog_names.append("SPY_ret")
if vix_ret is not None:
    exog_list.append(vix_ret)
    exog_names.append("VIX_ret")
if kweb_ret is not None:
    exog_list.append(kweb_ret)
    exog_names.append("KWEB_ret")

if not exog_list:
    st.error("SPY / VIX / KWEB getirileri bulunamadı, ARIMAX kurulamadı.")
    st.stop()

exog = pd.concat(exog_list, axis=1)
exog.columns = exog_names

# ARIMAX + GARCH
try:
    arimax_res, mu_daily = fit_arimax(main_ret, exog)
except Exception as e:
    st.error(f"ARIMAX modeli kurulurken hata oluştu: {e}")
    st.stop()

try:
    garch_res, sigma_daily = fit_garch(arimax_res.resid.dropna())
except Exception:
    sigma_daily = main_ret.std()
    garch_res = None

S0 = float(prices[main_symbol].iloc[-1])

# ATR(14) ve tam OHLC
ohlc_main: Optional[pd.DataFrame] = None
atr_series: Optional[pd.Series] = None
atr_last = np.nan

try:
    ohlc_main = download_ohlc(main_symbol, start_date, end_date)
    atr_series = compute_atr(ohlc_main, window=14)
    atr_last = (
        float(atr_series.dropna().iloc[-1])
        if not atr_series.dropna().empty
        else np.nan
    )
except Exception as e:
    st.warning(f"ATR hesaplanamadı: {e}")
    ohlc_main = None
    atr_series = None
    atr_last = np.nan

# --- VWMA(66), ZLEMA(20), OBV, VWMA_Z, Close-VWMA_Z ---
close_px = None
volume = None
vwma = None
zlema_20 = None
obv = None
vwma_z = None
close_minus_vwma = None
close_minus_vwma_z = None

try:
    full_ohlc = yf.download(
        main_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )
    if full_ohlc.empty:
        raise RuntimeError("OBV/VWMA için OHLC verisi boş döndü.")

    # Close kolonu – her durumda 1D pandas Series olsun
    if "Close" not in full_ohlc.columns:
        raise RuntimeError("Close kolonu bulunamadı.")

    close_px = full_ohlc["Close"]
    if isinstance(close_px, pd.DataFrame):
        close_px = close_px.iloc[:, 0]
    close_px = close_px.astype(float)

    # Volume – varsa al, yoksa hepsini 1 kabul et
    if "Volume" in full_ohlc.columns:
        volume = full_ohlc["Volume"]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        volume = volume.astype(float)
    else:
        volume = pd.Series(1.0, index=close_px.index)

    # VWMA(66)
    vw_len = 66
    vwma = (close_px * volume).rolling(vw_len).sum() / volume.rolling(vw_len).sum()

    # ZLEMA(20)
    zlema_20 = zlema(close_px, length=20)

    # OBV: numpy tarafında 1D array'e zorla
    price_diff = close_px.diff()
    diff_np = price_diff.to_numpy().reshape(-1)
    vol_np = volume.to_numpy().reshape(-1)

    obv_raw = np.where(
        diff_np > 0,
        vol_np,
        np.where(diff_np < 0, -vol_np, 0.0),
    ).reshape(-1)

    obv = pd.Series(obv_raw, index=close_px.index).cumsum()

    # VWMA Z-skoru ve Close-VWMA Z-skoru (66 bar)
    vw_z_len = 66
    vwma_z = rolling_zscore(vwma, vw_z_len)
    close_minus_vwma = close_px - vwma
    close_minus_vwma_z = rolling_zscore(close_minus_vwma, vw_z_len)

except Exception as e:
    st.warning(f"OBV/VWMA hesaplanırken hata oluştu: {e}")
    vwma = None
    zlema_20 = None
    obv = None
    vwma_z = None
    close_minus_vwma = None
    close_minus_vwma_z = None
    # close_px bırakıyoruz; None ise aşağıda zaten kontrol ediliyor.

# Haftalık likidite – 12 seviyeye kadar
main_price_series = prices[main_symbol]
liq_df, nearest_buy, nearest_sell, weekly_low, weekly_high = compute_weekly_liquidity(
    main_price_series, n_levels=12
)

if not liq_df.empty and not np.isnan(atr_last):
    liq_df["ATR14"] = atr_last

    def _stop_level(row):
        if row["Tür"].startswith("Buy"):
            return row["Seviye"] - atr_last
        else:
            return row["Seviye"] + atr_last

    liq_df["StopSeviyesi"] = liq_df.apply(_stop_level, axis=1)

# Smart Money metrikleri
smi, cycle_pos, beta_spy, beta_kweb, spread_z, smi_series = calc_smart_money_metrics(
    prices, rets, main_symbol, "SPY", "KWEB"
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Son Kapanış (USD)", f"{S0:.2f}")
col2.metric(
    "Smart Money Index",
    f"{smi:.2f}",
    help=">0: görece kurumsal alım, <0: görece dağıtım.",
)
col3.metric(
    "12 Haftalık Cycle Pozisyonu",
    f"{cycle_pos:.2f}",
    help="0: dip bölge, 1: tepe bölge.",
)
col4.metric("β_SPY (Hedge Oranı)", f"{beta_spy:.2f}")
col5.metric(
    "Spread Z-Skoru",
    f"{spread_z:.2f}",
    help="Ana sembol - β_KWEB*KWEB fiyatlarından türetilen z-skoru.",
)

# ---------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------
paths = monte_carlo_paths(S0, mu_daily, sigma_daily, horizon=horizon, n_paths=n_paths)
terminal_prices = paths[-1, :]

q10 = float(np.quantile(terminal_prices, 0.10))
median_price = float(np.median(terminal_prices))
q90 = float(np.quantile(terminal_prices, 0.90))

fan_low = np.quantile(paths, 0.10, axis=1)
fan_high = np.quantile(paths, 0.90, axis=1)
median_path = np.median(paths, axis=1)

# ---------------------------------------------------------------------
# Likiditeye göre otomatik Entry / Stop / TP1 / TP2 (Long + Short)
# ---------------------------------------------------------------------
# Son 12 haftalık en yüksek SELL likidite seviyesi
highest_sell_12w = np.nan
if not weekly_high.empty:
    n_weeks_h = min(12, len(weekly_high))
    sell_window = weekly_high.tail(n_weeks_h).dropna()
    if not sell_window.empty:
        above = sell_window[sell_window >= S0]
        highest_sell_12w = float(above.max()) if not above.empty else float(sell_window.max())

# Son 12 haftalık en düşük BUY likidite seviyesi (Short için TP2)
lowest_buy_12w = np.nan
if not weekly_low.empty:
    n_weeks_b = min(12, len(weekly_low))
    buy_window = weekly_low.tail(n_weeks_b).dropna()
    if not buy_window.empty:
        below = buy_window[buy_window <= S0]
        lowest_buy_12w = float(below.min()) if not below.empty else float(buy_window.min())

auto_entry = round(S0, 2)

if position_side == "Long":
    auto_stop = round(nearest_buy, 2) if not np.isnan(nearest_buy) else round(q10, 2)
    auto_tp1 = round(nearest_sell, 2) if not np.isnan(nearest_sell) else round(q90, 2)
    auto_tp2 = (
        round(highest_sell_12w, 2)
        if not np.isnan(highest_sell_12w)
        else round(q90, 2)
    )
else:  # SHORT
    # Short: zarar yukarıda (sell likidite), hedefler aşağıda (buy likidite)
    auto_stop = round(nearest_sell, 2) if not np.isnan(nearest_sell) else round(q90, 2)
    auto_tp1 = round(nearest_buy, 2) if not np.isnan(nearest_buy) else round(q10, 2)
    auto_tp2 = (
        round(lowest_buy_12w, 2)
        if not np.isnan(lowest_buy_12w)
        else round(q10, 2)
    )

# Sembol veya yön değiştiğinde otomatik reset
if (
    "prev_symbol" not in st.session_state
    or "prev_side" not in st.session_state
    or st.session_state.prev_symbol != main_symbol
    or st.session_state.prev_side != position_side
):
    st.session_state.prev_symbol = main_symbol
    st.session_state.prev_side = position_side
    st.session_state.entry_price = auto_entry
    st.session_state.stop_price = auto_stop
    st.session_state.tp1_price = auto_tp1
    st.session_state.tp2_price = auto_tp2
else:
    st.session_state.setdefault("entry_price", auto_entry)
    st.session_state.setdefault("stop_price", auto_stop)
    st.session_state.setdefault("tp1_price", auto_tp1)
    st.session_state.setdefault("tp2_price", auto_tp2)

# ---------------------------------------------------------------------
# Trade parametreleri – kullanıcı inputları
# ---------------------------------------------------------------------
st.subheader("🎯 Trade Parametreleri (Entry / Stop / TP1 / TP2)")

tc1, tc2, tc3 = st.columns(3)
with tc1:
    entry_price = st.number_input(
        "Giriş fiyatı (Entry)",
        min_value=0.0,
        value=float(st.session_state.entry_price),
        step=0.01,
        key="entry_price",
    )
with tc2:
    stop_price = st.number_input(
        "Stop fiyatı (Stop)",
        min_value=0.0,
        value=float(st.session_state.stop_price),
        step=0.01,
        key="stop_price",
    )
with tc3:
    tp1_price = st.number_input(
        "Hedef 1 fiyatı (TP1)",
        min_value=0.0,
        value=float(st.session_state.tp1_price),
        step=0.01,
        key="tp1_price",
    )

tc4, tc5 = st.columns(2)
with tc4:
    tp2_price = st.number_input(
        "Hedef 2 fiyatı (TP2)",
        min_value=0.0,
        value=float(st.session_state.tp2_price),
        step=0.01,
        key="tp2_price",
    )

with tc5:
    tp1_close_ratio = st.slider(
        "TP1'de kapatılacak pozisyon oranı",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

# ---------------------------------------------------------------------
# Haftalık likidite tablosu
# ---------------------------------------------------------------------
st.subheader("🧊 Haftalık Buy / Sell Likidite Seviyeleri (Haftalık Dip / Tepe)")

if liq_df.empty:
    st.info("Haftalık likidite seviyelerini hesaplamak için yeterli veri yok.")
else:
    fmt_cols = {"Seviye": "{:.2f}"}
    if "ATR14" in liq_df.columns:
        fmt_cols["ATR14"] = "{:.2f}"
    if "StopSeviyesi" in liq_df.columns:
        fmt_cols["StopSeviyesi"] = "{:.2f}"

    st.dataframe(liq_df.style.format(fmt_cols), use_container_width=True)

    explanation_lines = []
    if not np.isnan(nearest_buy):
        explanation_lines.append(
            f"- Fiyata en yakın **buy likidite** seviyesi (aşağıda): **{nearest_buy:.2f} USD**"
        )
    if not np.isnan(nearest_sell):
        explanation_lines.append(
            f"- Fiyata en yakın **sell likidite** seviyesi (yukarıda): **{nearest_sell:.2f} USD**"
        )
    if not np.isnan(atr_last):
        explanation_lines.append(
            f"- Güncel **ATR(14)** ≈ **{atr_last:.2f} USD**; tabloda stop seviyeleri buna göre hesaplandı."
        )

    if explanation_lines:
        st.markdown("**Fiyata Göre Öne Çıkan Seviyeler:**")
        for line in explanation_lines:
            st.markdown(line)

# ---------------------------------------------------------------------
# 🔥 Son 12 Haftalık Likidite Yoğunluk Isı Haritası
# ---------------------------------------------------------------------
st.subheader("🔥 Son 12 Haftalık Likidite Yoğunluk Isı Haritası")

valid_buy = weekly_low.dropna()
valid_sell = weekly_high.dropna()

if valid_buy.empty or valid_sell.empty:
    st.info("Isı haritası için yeterli haftalık dip/tepe verisi yok.")
else:
    n_weeks = min(12, len(valid_buy), len(valid_sell))
    buy_lastN = valid_buy.tail(n_weeks).values
    sell_lastN = valid_sell.tail(n_weeks).values

    all_vals = np.concatenate([buy_lastN, sell_lastN])
    if np.any(np.isfinite(all_vals)):
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)
        if vmin == vmax:
            vmax = vmin + 1e-3

        bins = np.linspace(vmin, vmax, 30)
        buy_counts, _ = np.histogram(buy_lastN, bins=bins)
        sell_counts, _ = np.histogram(sell_lastN, bins=bins)

        matrix = np.vstack([buy_counts, sell_counts])  # 2 x (nbin-1)

        fig_heat, ax_heat = plt.subplots(figsize=(8, 2.5))
        im = ax_heat.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=[bins[0], bins[-1], 0, 2],
        )
        ax_heat.set_yticks([0.5, 1.5])
        ax_heat.set_yticklabels(["Buy likidite", "Sell likidite"])
        ax_heat.set_xlabel("Fiyat")
        ax_heat.set_title(f"{main_symbol} – Son 12 Haftalık Likidite Yoğunluğu")
        cbar = plt.colorbar(im, ax=ax_heat)
        cbar.set_label("Seviye adedi (fiyat bandı başına)")
        st.pyplot(fig_heat)
    else:
        st.info("Isı haritası için geçerli fiyat seviyesi bulunamadı.")

# ---------------------------------------------------------------------
# 🔍 Likidite Pattern Analizi (cluster + sweep)
# ---------------------------------------------------------------------
pattern_info = detect_liquidity_patterns(
    ohlc_main, weekly_low, weekly_high, atr_last, S0
)

st.subheader("🔍 Likidite Pattern Yorumu")
st.write(pattern_info["comment"])

if not np.isnan(pattern_info["cluster_low"]):
    st.markdown(
        f"- **Likidite kümesi** yaklaşık: "
        f"**{pattern_info['cluster_low']:.2f} – {pattern_info['cluster_high']:.2f} USD**"
    )
    if pattern_info["down_sweep"]:
        st.markdown(
            "- Son günlerde bu kümenin **altına fitil atan bir süpürme** görüldü "
            "(buy stopları temizleme hareketi)."
        )

# 📏 Son 12 Haftalık Buy/Sell Likidite Yatay Çizgi + Mum Grafiği
st.subheader("📏 Son 12 Haftalık Buy / Sell Likidite Seviyeleri – Yatay Çizgiler")

if valid_buy.empty or valid_sell.empty or ohlc_main is None:
    st.info("Yatay çizgi grafiği için yeterli haftalık dip/tepe veya OHLC verisi yok.")
else:
    n_weeks = min(12, len(valid_buy), len(valid_sell))
    buy_lastN = valid_buy.tail(n_weeks)
    sell_lastN = valid_sell.tail(n_weeks)

    # Son ~120 günlük OHLC verisi (12 haftayı kapsasın)
    ohlc_last = ohlc_main.tail(120)

    fig_lines, ax_lines = plt.subplots(figsize=(8, 4))

    # --- Mini mum + fitil çizimi ---
    for idx_, row in ohlc_last.iterrows():
        o_ = row["Open"]
        h_ = row["High"]
        l_ = row["Low"]
        c_ = row["Close"]

        color = "green" if c_ >= o_ else "red"

        # Fitil
        ax_lines.vlines(idx_, l_, h_, color=color, linewidth=0.8)

        # Gövde (Open–Close arası daha kalın çizgi)
        ax_lines.vlines(idx_, o_, c_, color=color, linewidth=3)

    # Mumlar için legend'da çizgi göstermek
    ax_lines.plot([], [], color="black", linewidth=1.5, label="Fiyat (mum + fitil)")

    # Buy likidite: kesik çizgiler
    first_buy = True
    for lvl in buy_lastN.values:
        ax_lines.axhline(
            lvl,
            linestyle="--",
            alpha=0.7,
            linewidth=1.0,
            label="Buy likidite" if first_buy else None,
        )
        first_buy = False

    # Sell likidite: noktalı çizgiler
    first_sell = True
    for lvl in sell_lastN.values:
        ax_lines.axhline(
            lvl,
            linestyle=":",
            alpha=0.7,
            linewidth=1.0,
            label="Sell likidite" if first_sell else None,
        )
        first_sell = False

    # VWMA(66) ve ZLEMA(20) çizgileri (mumların üzerinde hareketli ortalamalar)
    if vwma is not None:
        vwma_last = vwma.reindex(ohlc_last.index)
        ax_lines.plot(
            ohlc_last.index,
            vwma_last,
            linewidth=1.5,
            label="VWMA(66)",
        )
    if zlema_20 is not None:
        zlema_last = zlema_20.reindex(ohlc_last.index)
        ax_lines.plot(
            ohlc_last.index,
            zlema_last,
            linewidth=1.5,
            label="ZLEMA(20)",
        )

    # Likidite kümesi bölgesini gölgele
    if not np.isnan(pattern_info["cluster_low"]):
        ax_lines.axhspan(
            pattern_info["cluster_low"],
            pattern_info["cluster_high"],
            alpha=0.15,
            label="Likidite kümesi",
        )

    # Entry / Stop / TP seviyeleri (yatay çizgiler)
    ax_lines.axhline(
        entry_price,
        linestyle="-",
        linewidth=1.4,
        label="Entry"
    )
    ax_lines.axhline(
        stop_price,
        linestyle="--",
        linewidth=1.4,
        label="Stop"
    )
    ax_lines.axhline(
        tp1_price,
        linestyle="-.",
        linewidth=1.4,
        label="TP1"
    )
    ax_lines.axhline(
        tp2_price,
        linestyle=":",
        linewidth=1.4,
        label="TP2"
    )

    # 🔎 Bugünkü fiyat noktasını ve en yakın likidite seviyelerini işaretle
    today_x = ohlc_last.index[-1]
    today_y = ohlc_last["Close"].iloc[-1]

    # Bugünkü fiyat
    ax_lines.scatter(today_x, today_y, s=60, color="blue", zorder=5)
    ax_lines.annotate(
        f"Bugün: {today_y:.2f}",
        xy=(today_x, today_y),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="blue",
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    # En yakın BUY likidite
    if not np.isnan(nearest_buy):
        ax_lines.annotate(
            f"En Yakın BUY ({nearest_buy:.2f})",
            xy=(today_x, nearest_buy),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green"),
        )

    # En yakın SELL likidite
    if not np.isnan(nearest_sell):
        ax_lines.annotate(
            f"En Yakın SELL ({nearest_sell:.2f})",
            xy=(today_x, nearest_sell),
            xytext=(10, 20),
            textcoords="offset points",
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax_lines.set_title(f"{main_symbol} – Son 12 Haftalık Likidite + VWMA(66) & ZLEMA(20)")
    ax_lines.set_ylabel("Fiyat (USD)")
    ax_lines.legend(loc="upper left")
    st.pyplot(fig_lines)

# ---------------------------------------------------------------------
# 🧱 Açık FVG ve Order Block Bölgeleri Tablosu
# ---------------------------------------------------------------------
st.subheader("🧱 Açık (Mitige Edilmemiş) FVG ve Order Block Bölgeleri")

if ohlc_main is None or ohlc_main.empty:
    st.info("FVG ve Order Block analizi için yeterli OHLC verisi yok.")
else:
    imbalance_df = build_imbalance_table(ohlc_main, atr_series, S0)

    if imbalance_df.empty:
        st.info("Şu anda açık (hiç test edilmemiş) FVG veya Order Block bölgesi bulunamadı.")
    else:
        st.dataframe(
            imbalance_df.style.format(
                {
                    "AltSeviye": "{:.2f}",
                    "ÜstSeviye": "{:.2f}",
                    "MerkezSeviye": "{:.2f}",
                    "Fiyata_Uzaklık_%": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown(
            """
**Not:**
- *Bullish FVG*: Aşağıda boşluk bırakmış, henüz fitille test edilmemiş gap alanı.
- *Bearish FVG*: Yukarıda boşluk bırakmış, henüz test edilmemiş gap alanı.
- *Bullish OB*: Güçlü yukarı impuls öncesi son düşüş mumu; fiyat henüz bu bölgeye geri dönmedi.
- *Bearish OB*: Güçlü aşağı impuls öncesi son yükseliş mumu; fiyat henüz bu bölgeye geri dönmedi.
- Fiyata_Uzaklık_% değeri, bölgenin merkezinin güncel fiyata göre yüzdesel mesafesini gösterir.
"""
        )

# ---------------------------------------------------------------------
# 📊 OBV, VWMA Z-skoru ve Close–VWMA Z-skoru Tablosu + Faz Analizi
# ---------------------------------------------------------------------
st.subheader("📊 OBV, VWMA Z-Skoru ve Close–VWMA Z-Skoru Tablosu")

if (
    vwma is None
    or obv is None
    or vwma_z is None
    or close_minus_vwma is None
    or close_minus_vwma_z is None
    or close_px is None
):
    st.info("OBV ve VWMA göstergeleri için hala veri üretilemedi. (Hem ana hesap hem fallback başarısız.)")
else:
    ind_df = pd.DataFrame(
        {
            "Close": close_px,
            "Volume": volume,
            "OBV": obv,
            "VWMA(66)": vwma,
            "VWMA_Z": vwma_z,
            "Close-VWMA": close_minus_vwma,
            "Close-VWMA_Z": close_minus_vwma_z,
        }
    )

    last_rows = ind_df.tail(60)

    st.dataframe(
        last_rows.style.format(
            {
                "Close": "{:.2f}",
                "VWMA(66)": "{:.2f}",
                "VWMA_Z": "{:.2f}",
                "Close-VWMA": "{:.2f}",
                "Close-VWMA_Z": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    phase_name, phase_comment = detect_phase_from_indicators(
        close_px, vwma, vwma_z, close_minus_vwma_z, obv
    )

    last_vwz = float(vwma_z.iloc[-1])
    last_cmvz = float(close_minus_vwma_z.iloc[-1])

    pc1, pc2 = st.columns(2)
    pc1.metric("VWMA Z-Skoru (Bugün)", f"{last_vwz:.2f}")
    pc2.metric("Close-VWMA Z-Skoru (Bugün)", f"{last_cmvz:.2f}")

    st.markdown(f"### Faz: **{phase_name}**")
    st.markdown(phase_comment)

# ---------------------------------------------------------------------
# Olasılıklar & Kelly
# ---------------------------------------------------------------------
p_tp1, p_tp2, p_stop, p_none = path_probabilities(
    paths, entry_price, stop_price, tp1_price, tp2_price, side=position_side
)

mu_R, var_R, kelly = kelly_from_probs(
    p_tp1,
    p_tp2,
    p_stop,
    p_none,
    entry_price,
    stop_price,
    tp1_price,
    tp2_price,
    side=position_side,
)

trade_risk_dollar = account_size * (risk_pct / 100.0)
risk_per_share = abs(entry_price - stop_price)
position_size = trade_risk_dollar / risk_per_share if risk_per_share > 0 else 0.0

half_kelly = max(0.0, kelly / 2.0)
recommended_risk_pct = min(risk_pct, half_kelly * 100.0)

# Smart Money karar tablosu (bugün)
decision_df = pd.DataFrame(
    {
        "Tarih": [end_date],
        "SMI": [round(smi, 2)],
        "CyclePos": [round(cycle_pos, 2)],
        "SpreadZ": [round(spread_z, 2)],
        "Kelly(%)": [round(kelly * 100, 2)],
        "P_TP1(%)": [round(p_tp1 * 100, 2)],
        "P_TP2(%)": [round(p_tp2 * 100, 2)],
        "P_Stop(%)": [round(p_stop * 100, 2)],
    }
)

if kelly <= 0 or p_stop > p_tp1:
    karar_text = "Smart money temkinli bölgede – küçük pozisyon veya bekle modu daha uygun."
else:
    karar_text = (
        "Smart money nötr / hafif pozitif bölgede – yine de küçük pozisyon boyutu mantıklı."
    )

decision_df["Karar"] = [karar_text]

st.subheader("⏱️ Smart Money Karar Tablosu (Bugün)")
st.dataframe(decision_df, use_container_width=True)

# ---------------------------------------------------------------------
# Smart Money modelleri özeti
# ---------------------------------------------------------------------
st.subheader("🧠 Smart Money Modelleri Özeti")

classic_today = float(smi_series.iloc[-1]) if not smi_series.empty else 0.0

rolling_window = 60
rolling_mean = smi_series.rolling(rolling_window).mean()
rolling_std = smi_series.rolling(rolling_window).std()
rolling_z = (smi_series - rolling_mean) / (rolling_std + 1e-9)
rolling_today = (
    float(rolling_z.iloc[-1]) if not rolling_z.dropna().empty else 0.0
)

# Volume-weighted SMI (kabaca fiyat*hacim ile)
if main_symbol in prices.columns:
    try:
        vol_download = yf.download(main_symbol, start=start_date, end=end_date, progress=False)
        if "Volume" in vol_download.columns:
            vol2 = vol_download["Volume"]
            vol2 = vol2.reindex(smi_series.index).fillna(method="ffill")
            vw_smi_series = (smi_series * (vol2 / vol2.mean())).rolling(20).mean()
            vw_today = (
                float(vw_smi_series.iloc[-1])
                if not vw_smi_series.dropna().empty
                else 0.0
            )
        else:
            vw_today = 0.0
    except Exception:
        vw_today = 0.0
else:
    vw_today = 0.0

# Rejim (uzun dönem SMI ortalamasının işareti)
regime_mean = smi_series.rolling(120).mean()
regime_val = (
    float(regime_mean.iloc[-1]) if not regime_mean.dropna().empty else 0.0
)
if regime_val > 0.1:
    regime_comment = "Rejim: pozitif / alım ağırlıklı"
elif regime_val < -0.1:
    regime_comment = "Rejim: negatif / dağıtım ağırlıklı"
else:
    regime_comment = "Rejim: nötr / karışık"

# Divergence: son 30 günde fiyat yukarı – SMI aşağı veya tersi
div_window = 30
if len(main_price_series) > div_window and len(smi_series) > div_window:
    price_last = main_price_series.iloc[-div_window:]
    smi_last = smi_series.iloc[-div_window:]
    price_slope = np.polyfit(np.arange(div_window), price_last.values, 1)[0]
    smi_slope = np.polyfit(np.arange(div_window), smi_last.values, 1)[0]

    if price_slope > 0 and smi_slope < 0:
        div_comment = "Negatif divergence: fiyat ↑, SMI ↓"
    elif price_slope < 0 and smi_slope > 0:
        div_comment = "Pozitif divergence: fiyat ↓, SMI ↑"
    else:
        div_comment = "Son pencerede belirgin divergence yok."
else:
    div_comment = "Veri yetersiz, divergence tespit edilemedi."

models_df = pd.DataFrame(
    [
        ["Classic SMI", round(classic_today, 4), "Klasik 20g SMI – ham z-skoru"],
        ["Rolling SMI (returns)", round(rolling_today, 4), "60g z-skor normalize edilmiş SMI"],
        ["Volume-weighted SMI", round(vw_today, 4), "Hacimle ağırlıklandırılmış SMI"],
        ["Regime", round(regime_val, 4), regime_comment],
        ["Divergence", None, div_comment],
    ],
    columns=["Model", "Değer", "Yorum"],
)

st.dataframe(models_df, use_container_width=True)

# ---------------------------------------------------------------------
# 📆 12 Haftalık Döngü Göstergesi (Özel Panel)
# ---------------------------------------------------------------------
st.subheader("📆 12 Haftalık Döngü Göstergesi")

cycle_prices, cycle_curve = compute_12w_cycle_curve(main_price_series, cycle_days=60)

if cycle_prices is None or cycle_curve is None:
    st.info("Döngü göstergesi için yeterli fiyat verisi yok.")
else:
    approx_week = cycle_pos * 12.0  # 0–1 → 0–12 hafta

    fig_cycle, ax_cycle = plt.subplots(figsize=(8, 3.5))
    ax_cycle.plot(
        cycle_prices.index,
        cycle_curve.values,
        label="Normalize fiyat (0–1)",
        linewidth=1.5,
    )

    # Dip / tepe bantları
    ax_cycle.axhline(0.2, linestyle="--", alpha=0.7, label="Dip bölgesi (~0–2. hafta)")
    ax_cycle.axhline(0.8, linestyle="--", alpha=0.7, label="Tepe bölgesi (~10–12. hafta)")

    # Bugünkü nokta
    ax_cycle.scatter(
        cycle_prices.index[-1],
        cycle_curve.iloc[-1],
        s=40,
        color="black",
        zorder=5,
        label="Bugün",
    )

    ax_cycle.set_ylim(-0.05, 1.05)
    ax_cycle.set_ylabel("Cycle pozisyonu (0–1)")
    ax_cycle.set_title(
        f"{main_symbol} – 12 Haftalık Döngü (CyclePos={cycle_pos:.2f}, ≈ {approx_week:.1f}. hafta)"
    )
    ax_cycle.legend()
    st.pyplot(fig_cycle)

    st.markdown(
        f"**Döngüdeki yaklaşık konum:** `CyclePos = {cycle_pos:.2f}` → "
        f"**≈ {approx_week:.1f}. hafta**  (0 = dip, 1 = tepe)."
    )

# ---------------------------------------------------------------------
# Tahmin özeti kutuları
# ---------------------------------------------------------------------
st.subheader("📊 Tahmin Özeti (Fiyat Dağılımı)")

oc1, oc2, oc3 = st.columns(3)
oc1.metric(f"{horizon} Gün Sonrası Medyan Fiyat", f"{median_price:.2f} USD")
oc2.metric("%10 Alt Band", f"{q10:.2f} USD")
oc3.metric("%90 Üst Band", f"{q90:.2f} USD")

st.subheader("📌 Path-based TP1 / TP2 / Stop Olasılıkları")

oc4, oc5, oc6, oc7 = st.columns(4)
oc4.metric("P(TP1'e en az bir kez dokunma)", f"{p_tp1*100:.2f} %")
oc5.metric("P(TP2'ye en az bir kez dokunma)", f"{p_tp2*100:.2f} %")
oc6.metric("P(Stop'a en az bir kez dokunma)", f"{p_stop*100:.2f} %")
oc7.metric("P(Hiçbiri görülmüyor)", f"{p_none*100:.2f} %")

st.subheader("💼 Pozisyon Boyutu, EV ve Kelly")

oc8, oc9, oc10, oc11 = st.columns(4)
oc8.metric("Trade başına risk ($)", f"{trade_risk_dollar:.2f}")
oc9.metric("Önerilen pozisyon büyüklüğü (adet)", f"{position_size:.0f}")
oc10.metric("P&L ortalaması (R)", f"{mu_R:.2f}")
oc11.metric("Approx. Kelly oranı", f"{kelly*100:.2f} %")

st.caption(
    "Kelly ≈ μ/σ² yaklaşımı ile hesaplanır. R birimindeki P&L dağılımından gelir. "
    "Pratikte genelde **yarım Kelly veya daha düşüğü** tercih edilir."
)

# ---------------------------------------------------------------------
# SMI zaman serisi grafiği
# ---------------------------------------------------------------------
st.subheader("📉 Smart Money Index Zaman Serisi")

if smi_series.dropna().empty:
    st.info("SMI serisi için yeterli veri yok.")
else:
    fig_smi, ax_smi = plt.subplots(figsize=(8, 4))
    ax_smi.plot(smi_series.index, smi_series.values, label="SMI (20g rolling)")
    ax_smi.plot(
        smi_series.index,
        smi_series.rolling(20).mean().values,
        label="SMI 20g ortalama",
    )
    ax_smi.axhline(0, linestyle="--", label="Nötr seviye")

    ax_smi.set_title(f"{main_symbol} – Smart Money Index")
    ax_smi.set_ylabel("SMI")
    ax_smi.legend()
    st.pyplot(fig_smi)

# ---------------------------------------------------------------------
# Fan Chart + Stop/TP çizgileri + Fibo on/off
# ---------------------------------------------------------------------
st.subheader("📉 Fan Chart – Olasılık Bantları")

show_fibo = st.checkbox("Fibo seviyelerini göster (son 120 gün swing high/low)")
st.caption(
    "Not: Tarayıcı tarafında mouse ile çizim yerine, son 120 günün high–low aralığına göre Fibo seviyeleri otomatik hesaplanıyor."
)

fig, ax = plt.subplots(figsize=(8, 4))
days = np.arange(1, horizon + 1)

ax.plot(days, median_path, label="Medyan Senaryo")
ax.fill_between(days, fan_low, fan_high, alpha=0.3, label="%10–%90 Bandı")

# Başlangıç fiyatı
ax.axhline(S0, linestyle="--", label="Başlangıç Fiyatı")

# Entry / Stop / TP yatay çizgileri
ax.axhline(entry_price, linestyle="-", linewidth=1.2, color= "yellow", label="Entry")
ax.axhline(stop_price, linestyle="--", linewidth=1.2, color= "red", label="Stop")
ax.axhline(tp1_price, linestyle="-.", linewidth=1.2, color= "green",label="TP1")
ax.axhline(tp2_price, linestyle=":", linewidth=1.2, color= "green",label="TP2")

# Opsiyonel: Fibo seviyeleri
if show_fibo and ohlc_main is not None and not ohlc_main.empty:
    recent_ohlc = ohlc_main.tail(120)
    if not recent_ohlc.empty:
        swing_low = recent_ohlc["Low"].min()
        swing_high = recent_ohlc["High"].max()
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

        first_fib = True
        for r in fib_ratios:
            if position_side == "Long":
                level = swing_low + (swing_high - swing_low) * r
            else:  # Short için ters yönlü yorum
                level = swing_high - (swing_high - swing_low) * r

            ax.axhline(
                level,
                linestyle="--",
                linewidth=0.8,
                alpha=0.4,
                label=f"Fibo {r:.3f}" if first_fib else None,
            )
            first_fib = False

ax.set_xlabel("Gün")
ax.set_ylabel("Fiyat (USD)")
ax.set_title(f"{main_symbol} – ARIMAX + GARCH + Monte Carlo")
ax.legend()
st.pyplot(fig)


# ---------------------------------------------------------------------
# 📄 TXT RAPOR OLUŞTURMA
# ---------------------------------------------------------------------


def generate_text_report():
    buffer = io.StringIO()

    buffer.write("SMART MONEY – OTOMATİK ANALİZ RAPORU\n")
    buffer.write("=====================================\n\n")

    buffer.write(f"Sembol: {main_symbol}\n")
    buffer.write(f"Tarih: {end_date}\n")
    buffer.write(f"Son Kapanış: {S0:.2f} USD\n\n")

    buffer.write("---- 1) SMART MONEY METRİKLERİ ----\n")
    buffer.write(f"Smart Money Index (SMI): {smi:.2f}\n")
    buffer.write(f"12 Haftalık Cycle Pozisyonu: {cycle_pos:.2f}\n")
    buffer.write(f"Spread Z-Skoru: {spread_z:.2f}\n")
    buffer.write(f"β_SPY: {beta_spy:.2f}, β_KWEB: {beta_kweb:.2f}\n\n")

    buffer.write("Rejim Yorumu:\n")
    buffer.write(f"- {regime_comment}\n")
    buffer.write(f"Divergence Analizi: {div_comment}\n\n")

    buffer.write("---- 2) FAZ ANALİZİ ----\n")
    buffer.write(f"Faz: {phase_name}\n")
    buffer.write(f"{phase_comment}\n\n")

    buffer.write("---- 3) HAFTALIK LİKİDİTE ANALİZİ ----\n")
    if not np.isnan(nearest_buy):
        buffer.write(f"En Yakın BUY Likidite: {nearest_buy:.2f}\n")
    if not np.isnan(nearest_sell):
        buffer.write(f"En Yakın SELL Likidite: {nearest_sell:.2f}\n")
    if not np.isnan(pattern_info['cluster_low']):
        buffer.write(
            f"Likidite Kümesi: {pattern_info['cluster_low']:.2f} – {pattern_info['cluster_high']:.2f}\n"
        )
    buffer.write(f"Sweep Durumu: {'VAR' if pattern_info['down_sweep'] else 'YOK'}\n")
    buffer.write(f"Likidite Yorumu: {pattern_info['comment']}\n\n")

    buffer.write("---- 4) HAREKETLİ ORTALAMALAR ----\n")
    buffer.write(f"VWMA Z-Skoru: {last_vwz:.2f}\n")
    buffer.write(f"Close-VWMA Z-Skoru: {last_cmvz:.2f}\n")
    buffer.write("\n")

    buffer.write("---- 5) MONTE CARLO SONUÇLARI ----\n")
    buffer.write(f"Medyan Fiyat ({horizon} gün): {median_price:.2f}\n")
    buffer.write(f"%10–%90 bandı: {q10:.2f} – {q90:.2f}\n")
    buffer.write(f"TP1 olasılığı: {p_tp1*100:.2f}%\n")
    buffer.write(f"TP2 olasılığı: {p_tp2*100:.2f}%\n")
    buffer.write(f"Stop olasılığı: {p_stop*100:.2f}%\n")
    buffer.write(f"Hiçbirine dokunmama: {p_none*100:.2f}%\n\n")

    buffer.write("---- 6) POZİSYON & KELLY ----\n")
    buffer.write(f"Entry: {entry_price}\n")
    buffer.write(f"Stop: {stop_price}\n")
    buffer.write(f"TP1: {tp1_price}\n")
    buffer.write(f"TP2: {tp2_price}\n")
    buffer.write(f"Pozisyon Büyüklüğü: {position_size:.0f} adet\n")
    buffer.write(f"Kelly Oranı: {kelly*100:.2f}%\n")
    buffer.write(f"Önerilen Max Risk: {recommended_risk_pct:.2f}%\n")
    buffer.write("\n")

    buffer.write("---- 7) OTOMATİK ÖZET ----\n")
    auto_text = auto_comment(
        position_side,
        horizon,
        median_price,
        q10,
        q90,
        p_tp1,
        p_tp2,
        p_stop,
        kelly,
        entry_price,
        stop_price,
        tp1_price,
        tp2_price,
    )
    buffer.write(auto_text + "\n")

    return buffer.getvalue()


# ---- Buton: TXT rapor indir ----
report_text = generate_text_report()
st.download_button(
    label="📄 TXT Raporu İndir",
    data=report_text,
    file_name=f"{main_symbol}_smartmoney_report.txt",
    mime="text/plain",
)

# ---------------------------------------------------------------------
# Otomatik yorum
# ---------------------------------------------------------------------
comment = auto_comment(
    position_side,
    horizon,
    median_price,
    q10,
    q90,
    p_tp1,
    p_tp2,
    p_stop,
    kelly,
    entry_price,
    stop_price,
    tp1_price,
    tp2_price,
)
st.markdown(comment)
