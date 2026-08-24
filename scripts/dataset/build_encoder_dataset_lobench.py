"""
build_encoder_dataset_lobench.py — Converte i 7 CSV di LOBench in un NPZ pronto
per l'encoder.

LOBench fornisce dati già z-scored globalmente per ogni stock, con schema:
    index (timestamp string), BidPrice10..1, AskPrice1..10, BidVolume10..1, AskVolume1..10

Il tuo encoder si aspetta un book in formato (N, 2, L, 2) con:
    axis 1: sides (0=bid, 1=ask)
    axis 2: level (0=best, L-1=deepest)
    axis 3: [price, volume]

Oltre al book grezzo, salvo mid-prices z-score per permettere al Dataset
successivo di applicare la normalizzazione (price - mid) / depth_scale.
Anche stock_ids e day_ids sono salvati per gli split.

Uso:
    python -m scripts.dataset.build_encoder_dataset_lobench \
        --raw_dir data/lobench/raw \
        --out data/lobench_processed.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column ordering: LOBench CSV uses different level ordering per side.
# Encoder expects level 0 = best, level L-1 = deepest.
# ---------------------------------------------------------------------------

# Bid: BidPrice1 = best (highest). Salgo 1..10 per avere best-first.
BID_PRICE_COLS = [f"BidPrice{i}"  for i in range(1, 11)]
BID_VOL_COLS   = [f"BidVolume{i}" for i in range(1, 11)]

# Ask: AskPrice1 = best (lowest). Salgo 1..10 per avere best-first.
ASK_PRICE_COLS = [f"AskPrice{i}"  for i in range(1, 11)]
ASK_VOL_COLS   = [f"AskVolume{i}" for i in range(1, 11)]


def filter_canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical LOBench row filters without changing row order.

    Extra columns are deliberately preserved.  The Experiment 01 metadata
    sidecar uses this property to carry the original zero-based CSV data-row
    index through exactly the same filtering operation as the NPZ builder.
    """
    ts = pd.to_datetime(df["index"])
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    keep_time = minute_of_day < (14 * 60 + 57)
    filtered = df.loc[keep_time]
    keep_book = (
        (filtered["BidPrice1"] - filtered["BidPrice10"]).abs() > 1e-6
    ) & (
        (filtered["AskPrice1"] - filtered["AskPrice10"]).abs() > 1e-6
    )
    return filtered.loc[keep_book].reset_index(drop=True)


def process_csv(csv_path: Path, stock_id: int, subsample_stride: int = 1
                ) -> dict[str, np.ndarray]:
    """Load a single LOBench CSV and return structured arrays."""
    print(f"  [read] {csv_path.name} (stock_id={stock_id})")
    df = pd.read_csv(csv_path)

    # ----- FILTRO ANOMALIE LOBENCH -----
    # Il dataset LOBench resampla a 3s tutti gli orari di trading SZSE,
    # incluso il "close auction" 14:57:00-15:00:00 dove il book si "collassa"
    # ad un singolo prezzo di equilibrio (tutti i 10 livelli uguali).
    # Questi sample hanno (price_z - mid_z)/tick_z patologico (>1000) e
    # destabilizzano il training. Li rimuoviamo qui.
    n_before = len(df)

    # Filtro 1: orario. Escludo close auction (14:57:00 - 15:00:00).
    # Il continuous trading SZSE chiude alle 14:57:00. Conservo 09:30-14:57.
    df = filter_canonical_rows(df)

    n_after = len(df)
    print(f"    filtered: {n_before - n_after:,} righe rimosse "
          f"({100*(n_before-n_after)/n_before:.2f}%) "
          f"-> {n_after:,} righe rimaste")

    if subsample_stride > 1:
        df = df.iloc[::subsample_stride].reset_index(drop=True)
    N = len(df)

    # Build book (N, 2, L, 2) — level 0 = best
    book = np.zeros((N, 2, 10, 2), dtype=np.float32)
    book[:, 0, :, 0] = df[BID_PRICE_COLS].to_numpy(dtype=np.float32)
    book[:, 0, :, 1] = df[BID_VOL_COLS  ].to_numpy(dtype=np.float32)
    book[:, 1, :, 0] = df[ASK_PRICE_COLS].to_numpy(dtype=np.float32)
    book[:, 1, :, 1] = df[ASK_VOL_COLS  ].to_numpy(dtype=np.float32)

    # Mid-prices in z-score units (used later for per-sample centering)
    mid_z = (book[:, 0, 0, 0] + book[:, 1, 0, 0]) / 2.0  # (N,)

    # Calcolo min_spread_z per ricostruire price_std.
    # LOBench ha normalizzato come (price_raw - mean) / std con UNO std per stock.
    # La minima differenza osservabile tra prezzi fisici e' 1 tick = 0.01 RMB.
    # Quindi il minimo spread z-scored osservato = 0.01 / price_std_stock.
    # Salvandolo possiamo de-normalizzare i prezzi (a meno di una costante
    # additiva irrilevante, dato che in pipeline facciamo price - mid).
    spread_z = book[:, 1, 0, 0] - book[:, 0, 0, 0]  # AskPrice1 - BidPrice1
    spread_pos = spread_z[spread_z > 0]
    min_spread_z = float(spread_pos.min()) if len(spread_pos) else 1e-3
    # Equivalenza: min_spread_z = 0.01 / price_std => price_std = 0.01/min_spread_z
    price_std_rmb = 0.01 / min_spread_z
    print(f"    min_spread_z={min_spread_z:.6f}  "
          f"=> price_std={price_std_rmb:.3f} RMB")

    # Parse timestamps to extract day_id
    ts = pd.to_datetime(df["index"])
    day_id = (ts.dt.dayofyear - ts.dt.dayofyear.min()).to_numpy(dtype=np.int32)

    stock_ids = np.full(N, stock_id, dtype=np.int32)

    return {
        "book":          book,
        "mid_z":         mid_z.astype(np.float32),
        "stock_ids":     stock_ids,
        "day_ids":       day_id,
        "min_spread_z":  min_spread_z,
        "price_std_rmb": price_std_rmb,
        "n":             N,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",          type=str, required=True,
                        help="Directory contenente gli sz*.csv di LOBench")
    parser.add_argument("--out",              type=str,
                        default="data/lobench_processed.npz")
    parser.add_argument("--subsample_stride", type=int, default=1,
                        help="Prendi 1 riga ogni N. 1 = full data, 3 = 1/3 data.")
    parser.add_argument("--max_samples",      type=int, default=None,
                        help="Hard cap su numero totale di snapshot (per debug)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    csv_files = sorted(raw_dir.glob("sz*_processed.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nessun sz*_processed.csv in {raw_dir}")

    print(f"Trovati {len(csv_files)} file CSV:")
    for f in csv_files:
        print(f"  - {f.name}")
    print(f"Subsample stride: {args.subsample_stride}")

    # Process each CSV
    chunks = []
    for stock_id, csv_path in enumerate(csv_files):
        chunk = process_csv(csv_path, stock_id, args.subsample_stride)
        chunks.append(chunk)
        print(f"    -> {chunk['n']:,} righe")

    # Concatenate
    book      = np.concatenate([c["book"]      for c in chunks], axis=0)
    mid_z     = np.concatenate([c["mid_z"]     for c in chunks], axis=0)
    stock_ids = np.concatenate([c["stock_ids"] for c in chunks], axis=0)
    day_ids   = np.concatenate([c["day_ids"]   for c in chunks], axis=0)
    # Stats per-stock (arrays indicizzati da stock_id 0..n_stocks-1)
    min_spread_z_per_stock = np.array(
        [c["min_spread_z"] for c in chunks], dtype=np.float32
    )
    price_std_rmb_per_stock = np.array(
        [c["price_std_rmb"] for c in chunks], dtype=np.float32
    )
    N = len(book)
    print(f"\nTotale: {N:,} snapshot")

    # Optional hard cap
    if args.max_samples is not None and N > args.max_samples:
        idx = np.random.default_rng(42).choice(N, args.max_samples, replace=False)
        idx = np.sort(idx)
        book      = book[idx]
        mid_z     = mid_z[idx]
        stock_ids = stock_ids[idx]
        day_ids   = day_ids[idx]
        N = len(book)
        print(f"Capped a {N:,}")

    # Nota: non facciamo clipping globale dei volumi qui.
    # La normalizzazione downstream (LOBenchDataset) e' per-stock:
    #   vol_shifted = vol_z - vol_z.min()  (stock-specific)
    #   vol_normalized = vol_shifted / p99(vol_shifted>0)
    # che gestisce correttamente gli outlier per-stock.

    # Report statistics for sanity
    print(f"\nStatistiche finali:")
    print(f"  N totali            : {N:,}")
    print(f"  Stocks              : {np.unique(stock_ids).tolist()}")
    print(f"  Day range           : [{day_ids.min()}, {day_ids.max()}]")
    print(f"  Price range (z)     : [{book[:, :, :, 0].min():.3f}, {book[:, :, :, 0].max():.3f}]")
    print(f"  Volume range (z)    : [{book[:, :, :, 1].min():.3f}, {book[:, :, :, 1].max():.3f}]")
    print(f"  Mid-price range (z) : [{mid_z.min():.3f}, {mid_z.max():.3f}]")
    print(f"  Per-stock reconstruction stats:")
    for i in range(len(min_spread_z_per_stock)):
        n_i = int((stock_ids == i).sum())
        print(f"    stock {i} ({n_i:>8,} rows): "
              f"min_spread_z={min_spread_z_per_stock[i]:.6f}  "
              f"price_std={price_std_rmb_per_stock[i]:.3f} RMB")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        book=book,
        mid_z=mid_z,
        stock_ids=stock_ids,
        day_ids=day_ids,
        min_spread_z_per_stock=min_spread_z_per_stock,
        price_std_rmb_per_stock=price_std_rmb_per_stock,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSalvato in {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
