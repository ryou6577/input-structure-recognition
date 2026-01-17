import os
import numpy as np
import pandas as pd
import pyupbit
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# 1) 데이터 수집
# =========================
def fetch_upbit_ohlcv(market="KRW-BTC", interval="minute60", count=2000):
    """
    Upbit OHLCV fetch
    interval: minute1, minute3, minute5, minute10, minute15, minute30, minute60,
              minute240, day, week, month
    count: up to 2000 per call
    """
    df = pyupbit.get_ohlcv(market, interval=interval, count=count)
    if df is None or len(df) == 0:
        raise RuntimeError("Failed to fetch OHLCV from Upbit.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


# =========================
# 2) 피처 생성 (최소 버전)
# =========================
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV -> basic derived features
    * 첫 단추 단계: 너무 많은 지표 금지 (중복 + 과적합 위험)
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # log returns
    out["ret_1"] = np.log(close).diff()

    # range-based volatility proxies
    out["hl_range"] = np.log(high / low)
    out["co_range"] = np.log(close / open_)

    # rolling vol
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_30"] = out["ret_1"].rolling(30).std()

    # volume features
    out["v_log"] = np.log(volume.replace(0, np.nan))
    out["v_chg"] = out["v_log"].diff()
    out["v_z30"] = (out["v_log"] - out["v_log"].rolling(30).mean()) / (
        out["v_log"].rolling(30).std() + 1e-9
    )

    # simple momentum
    out["mom_10"] = np.log(close).diff(10)
    out["mom_30"] = np.log(close).diff(30)

    # cleaning
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


# =========================
# 3) Subspace similarity / drift
# =========================
def subspace_similarity(U_ref, U_now):
    """
    Compare two subspaces with principal angles
    U_ref, U_now: (d x k) orthonormal bases
    Returns:
      sim in [0,1] (mean cos), singular values
    """
    M = U_ref.T @ U_now
    s = np.linalg.svd(M, compute_uv=False)  # singular values = cos(principal angles)
    sim = float(np.mean(s))
    return sim, s


def subspace_drift(U_prev, U_now):
    """
    drift = 1 - mean(cos(principal angles))
    """
    sim, s = subspace_similarity(U_prev, U_now)
    return 1.0 - sim, s


# =========================
# 4) PCA 구조 진단 (전역)
# =========================
def pca_structure_report(X: pd.DataFrame, n_components=None, title="PCA Structure"):
    """
    X: features (T x d)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=n_components)
    pca.fit(Xs)

    eigvals = pca.explained_variance_
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)

    # participation ratio = effective dimension
    pr = (eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12)

    print("=" * 70)
    print(title)
    print(f"Samples(T)={len(X)}, Features(d)={X.shape[1]}")
    print(f"Effective dimension (Participation Ratio) = {pr:.2f}")
    print("Top components cumulative explained variance:")
    for k in [1, 2, 3, 5, 8, 10]:
        if k <= len(cum):
            print(f"  k={k:2d}: {cum[k-1]*100:6.2f}%")
    print("=" * 70)

    # plots
    plt.figure()
    plt.plot(ratios, marker="o")
    plt.title(title + " - Explained Variance Ratio")
    plt.xlabel("PC index")
    plt.ylabel("Explained variance ratio")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(cum, marker="o")
    plt.title(title + " - Cumulative Explained Variance")
    plt.xlabel("PC count")
    plt.ylabel("Cumulative explained variance")
    plt.grid(True)
    plt.show()

    return {
        "pca": pca,
        "eigvals": eigvals,
        "ratios": ratios,
        "cum": cum,
        "pr": pr
    }


# =========================
# 5) 롤링 구조 + drift + anchor similarity
# =========================
def rolling_structure_with_drift_and_anchor(
    X: pd.DataFrame,
    window=500,
    k=3,
    anchor_mode="first",
    anchor_time=None
):
    """
    Rolling PCA:
      - PR(effective dimension)
      - cum2, cum5
      - consecutive drift (top-k subspace)
      - anchor similarity (identity vs fixed reference)

    anchor_mode:
      - "first": first available rolling window becomes anchor
      - "time": anchor is set at first t >= anchor_time
    """
    scaler = StandardScaler()
    Xv = X.values
    idx = pd.to_datetime(X.index)

    prs, cum2, cum5 = [], [], []
    drift_cons, anchor_sims, anchor_drifts = [], [], []
    times = []

    U_prev = None
    U_anchor = None
    anchor_set = False

    if anchor_time is not None:
        anchor_time = pd.to_datetime(anchor_time)

    for i in range(window, len(Xv)):
        chunk = Xv[i - window:i]
        chunk = scaler.fit_transform(chunk)

        pca = PCA()
        pca.fit(chunk)

        eigvals = pca.explained_variance_
        ratios = pca.explained_variance_ratio_
        cum = np.cumsum(ratios)

        pr = (eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12)

        prs.append(pr)
        cum2.append(cum[1] if len(cum) > 1 else np.nan)
        cum5.append(cum[4] if len(cum) > 4 else np.nan)

        # subspace basis: (d x k)
        U_now = pca.components_[:k].T

        # consecutive drift
        if U_prev is None:
            drift_cons.append(np.nan)
        else:
            d, _ = subspace_drift(U_prev, U_now)
            drift_cons.append(d)

        # set anchor
        t_now = idx[i]
        if not anchor_set:
            if anchor_mode == "first":
                U_anchor = U_now
                anchor_set = True
            elif anchor_mode == "time":
                if anchor_time is None:
                    raise ValueError("anchor_mode='time' requires anchor_time like '2025-12-20'.")
                if t_now >= anchor_time:
                    U_anchor = U_now
                    anchor_set = True
            else:
                raise ValueError("anchor_mode must be 'first' or 'time'.")

        # anchor similarity
        if U_anchor is None:
            anchor_sims.append(np.nan)
            anchor_drifts.append(np.nan)
        else:
            sim, _ = subspace_similarity(U_anchor, U_now)
            anchor_sims.append(sim)
            anchor_drifts.append(1.0 - sim)

        U_prev = U_now
        times.append(t_now)

    df_roll = pd.DataFrame({
        "pr_effdim": prs,
        "cum2": cum2,
        "cum5": cum5,
        f"drift_top{k}": drift_cons,
        f"anchor_sim_top{k}": anchor_sims,
        f"anchor_drift_top{k}": anchor_drifts
    }, index=pd.to_datetime(times))

    # ---------------- plots
    plt.figure()
    plt.plot(df_roll.index, df_roll["pr_effdim"])
    plt.title(f"Rolling Effective Dimension (window={window})")
    plt.xlabel("time")
    plt.ylabel("effective dimension (PR)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(df_roll.index, df_roll["cum2"], label="cum2")
    plt.plot(df_roll.index, df_roll["cum5"], label="cum5")
    plt.title(f"Rolling Cumulative Explained Variance (window={window})")
    plt.xlabel("time")
    plt.ylabel("cumulative variance")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(df_roll.index, df_roll[f"drift_top{k}"])
    plt.title(f"Rolling Subspace Drift top{k} (window={window})")
    plt.xlabel("time")
    plt.ylabel("drift (1-mean cos)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(df_roll.index, df_roll[f"anchor_sim_top{k}"])
    plt.title(f"Anchor Similarity top{k} (anchor_mode={anchor_mode}, anchor_time={anchor_time})")
    plt.xlabel("time")
    plt.ylabel("similarity (mean cos)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(df_roll.index, df_roll[f"anchor_drift_top{k}"])
    plt.title(f"Anchor Drift top{k} (anchor_mode={anchor_mode}, anchor_time={anchor_time})")
    plt.xlabel("time")
    plt.ylabel("anchor drift (1-sim)")
    plt.grid(True)
    plt.show()

    return df_roll


# =========================
# main
# =========================
if __name__ == "__main__":
    market = "KRW-BTC"
    interval = "minute60"
    count = 2000
    window = 500
    k = 3

    # anchor 기준점: 전이 전 시점 권장
    # 예) "2025-12-20"
    anchor_mode = "time"
    anchor_time = "2025-12-20"

    df = fetch_upbit_ohlcv(market=market, interval=interval, count=count)
    print(df.tail())

    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{market}_{interval}_{count}.csv", encoding="utf-8-sig")

    X = make_features(df)
    print("Feature head:\n", X.head())
    print("Feature columns:", list(X.columns))

    report = pca_structure_report(X, title=f"{market} {interval} PCA Structure")

    # 롤링 구조 + drift + anchor
    roll = rolling_structure_with_drift_and_anchor(
        X,
        window=window,
        k=k,
        anchor_mode=anchor_mode,
        anchor_time=anchor_time
    )
    roll.to_csv(f"data/{market}_{interval}_rolling_structure_with_anchor.csv", encoding="utf-8-sig")

    # loading 출력
    loading = pd.DataFrame(
        report["pca"].components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(X.shape[1])]
    )
    print("\nTop loadings (abs) for PC1:")
    print(loading.iloc[:, :5].sort_values("PC1", key=np.abs, ascending=False))
