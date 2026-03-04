#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd


def _iter_files(root: Path, suffix: str) -> Iterator[Path]:
    if not root.exists():
        return
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    p = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(p)
                    elif entry.is_file(follow_symlinks=False) and p.name.endswith(suffix):
                        yield p
        except Exception:
            continue


def _to_iso_date_yyyymmdd(raw: str) -> str:
    s = str(raw or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return ""


def _norm_ticker(raw: str) -> str:
    s = str(raw or "").upper().strip()
    if s.endswith(".US"):
        s = s[:-3]
    return s


@dataclass
class ConsolidationStats:
    stooq_files: int = 0
    stooq_rows: int = 0
    stooq_bad_rows: int = 0
    fred_files: int = 0
    submissions_files: int = 0
    submissions_errors: int = 0
    companyfacts_files: int = 0
    companyfacts_errors: int = 0
    companyfacts_rows: int = 0


def _write_stooq_prices(stooq_root: Path, out_csv: Path, stats: ConsolidationStats) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
        for p in _iter_files(stooq_root, ".txt"):
            stats.stooq_files += 1
            try:
                with p.open("r", encoding="utf-8", errors="ignore", newline="") as r:
                    rr = csv.reader(r)
                    _ = next(rr, None)  # header
                    for row in rr:
                        if len(row) < 9:
                            stats.stooq_bad_rows += 1
                            continue
                        d = _to_iso_date_yyyymmdd(row[2])
                        t = _norm_ticker(row[0])
                        if not d or not t:
                            stats.stooq_bad_rows += 1
                            continue
                        o, h, l, c = row[4], row[5], row[6], row[7]
                        vol = row[8]
                        # Stooq does not ship adjusted close separately in these files.
                        w.writerow([d, t, o, h, l, c, c, vol])
                        stats.stooq_rows += 1
            except Exception:
                stats.stooq_bad_rows += 1


def _write_fred_macro(fred_root: Path, out_csv: Path, stats: ConsolidationStats) -> None:
    frames: list[pd.DataFrame] = []
    for f in sorted(fred_root.glob("*.csv")):
        stats.fred_files += 1
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.shape[1] < 2:
            continue
        date_col = df.columns[0]
        value_col = df.columns[1]
        tmp = df[[date_col, value_col]].copy()
        series_name = f.stem
        tmp.columns = ["date", series_name]
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        tmp[series_name] = pd.to_numeric(tmp[series_name], errors="coerce")
        tmp = tmp.dropna(subset=["date"])
        frames.append(tmp)
    if not frames:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["date"]).to_csv(out_csv, index=False)
        return
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date").drop_duplicates("date", keep="last")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)


def _latest_form_date(recent: dict, form_name: str) -> str:
    forms = recent.get("form", []) or []
    fdates = recent.get("filingDate", []) or []
    best = ""
    for i, form in enumerate(forms):
        if str(form).upper().strip() != form_name:
            continue
        d = str(fdates[i]) if i < len(fdates) else ""
        if d and d > best:
            best = d
    return best


def _write_sec_submissions(
    submissions_root: Path,
    out_csv: Path,
    stats: ConsolidationStats,
    max_files: int = 0,
) -> Dict[str, str]:
    cik_to_ticker: Dict[str, str] = {}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cik",
                "ticker",
                "name",
                "exchange",
                "sic",
                "sic_description",
                "entity_type",
                "fiscal_year_end",
                "state_of_incorporation",
                "latest_10k_date",
                "latest_10q_date",
                "latest_8k_date",
                "recent_filings_count",
                "source_file",
            ]
        )

        for p in _iter_files(submissions_root, ".json"):
            if max_files > 0 and stats.submissions_files >= max_files:
                break
            stats.submissions_files += 1
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as r:
                    payload = json.load(r)
            except Exception:
                stats.submissions_errors += 1
                continue

            cik = str(payload.get("cik", "")).strip()
            tickers = payload.get("tickers", []) or []
            ticker = _norm_ticker(tickers[0]) if tickers else ""
            if cik and ticker:
                cik_to_ticker[cik] = ticker

            recent = ((payload.get("filings") or {}).get("recent") or {})
            accession = recent.get("accessionNumber", []) or []
            w.writerow(
                [
                    cik,
                    ticker,
                    payload.get("name", ""),
                    (payload.get("exchanges", []) or [""])[0],
                    payload.get("sic", ""),
                    payload.get("sicDescription", ""),
                    payload.get("entityType", ""),
                    payload.get("fiscalYearEnd", ""),
                    payload.get("stateOfIncorporation", ""),
                    _latest_form_date(recent, "10-K"),
                    _latest_form_date(recent, "10-Q"),
                    _latest_form_date(recent, "8-K"),
                    len(accession),
                    str(p),
                ]
            )
    return cik_to_ticker


def _iter_selected_fact_points(payload: dict, keep_tags: set[str]) -> Iterator[dict]:
    facts = payload.get("facts", {}) or {}
    for taxonomy, tags in facts.items():
        if not isinstance(tags, dict):
            continue
        for tag, info in tags.items():
            if keep_tags and tag not in keep_tags:
                continue
            units = (info or {}).get("units", {}) or {}
            for unit, points in units.items():
                if not isinstance(points, list):
                    continue
                for pt in points:
                    if isinstance(pt, dict):
                        yield {
                            "taxonomy": taxonomy,
                            "tag": tag,
                            "unit": unit,
                            "end": pt.get("end", ""),
                            "val": pt.get("val", ""),
                            "filed": pt.get("filed", ""),
                            "form": pt.get("form", ""),
                            "fy": pt.get("fy", ""),
                            "fp": pt.get("fp", ""),
                            "accn": pt.get("accn", ""),
                            "frame": pt.get("frame", ""),
                        }


def _write_sec_companyfacts(
    companyfacts_root: Path,
    out_csv: Path,
    cik_to_ticker: Dict[str, str],
    keep_tags: set[str],
    stats: ConsolidationStats,
    max_files: int = 0,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cik",
                "ticker",
                "entity_name",
                "taxonomy",
                "tag",
                "unit",
                "end",
                "val",
                "filed",
                "form",
                "fy",
                "fp",
                "accn",
                "frame",
                "source_file",
            ]
        )
        for p in _iter_files(companyfacts_root, ".json"):
            if max_files > 0 and stats.companyfacts_files >= max_files:
                break
            stats.companyfacts_files += 1
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as r:
                    payload = json.load(r)
            except Exception:
                stats.companyfacts_errors += 1
                continue
            cik = str(payload.get("cik", "")).strip()
            ticker = cik_to_ticker.get(cik, "")
            entity = payload.get("entityName", "")
            for point in _iter_selected_fact_points(payload, keep_tags):
                w.writerow(
                    [
                        cik,
                        ticker,
                        entity,
                        point["taxonomy"],
                        point["tag"],
                        point["unit"],
                        point["end"],
                        point["val"],
                        point["filed"],
                        point["form"],
                        point["fy"],
                        point["fp"],
                        point["accn"],
                        point["frame"],
                        str(p),
                    ]
                )
                stats.companyfacts_rows += 1


def _safe_remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_remove_files(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate local Data FF folder into compact CSVs.")
    parser.add_argument("--data-ff-root", required=True, help="Path like '/Users/.../Desktop/Data FF'")
    parser.add_argument("--out-dir", required=True, help="Output directory for consolidated artifacts")
    parser.add_argument(
        "--companyfacts-tags",
        default=(
            "Assets,Liabilities,StockholdersEquity,Revenues,NetIncomeLoss,"
            "OperatingIncomeLoss,GrossProfit,LongTermDebtNoncurrent,"
            "CashAndCashEquivalentsAtCarryingValue,EntityCommonStockSharesOutstanding,"
            "CommonStockSharesOutstanding,EarningsPerShareBasic,EarningsPerShareDiluted"
        ),
        help="Comma-separated SEC fact tags to keep",
    )
    parser.add_argument(
        "--max-sec-files",
        type=int,
        default=0,
        help="Limit SEC files for smoke runs; 0 means no limit.",
    )
    parser.add_argument(
        "--skip-companyfacts",
        action="store_true",
        help="Skip companyfacts parse (submissions only).",
    )
    parser.add_argument(
        "--prune-source",
        action="store_true",
        help="Delete source subfolders after successful consolidation.",
    )
    args = parser.parse_args()

    data_ff_root = Path(args.data_ff_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    stooq_root = data_ff_root / "Stooq_data"
    fred_root = data_ff_root / "FRED"
    submissions_root = data_ff_root / "SEC_XBRL_submissions"
    companyfacts_root = data_ff_root / "SEC_XBRL_companyfacts"

    stats = ConsolidationStats()
    tags = {x.strip() for x in str(args.companyfacts_tags).split(",") if x.strip()}

    print(f"[1/4] Stooq -> {out_dir / 'prices.csv'}", flush=True)
    _write_stooq_prices(stooq_root, out_dir / "prices.csv", stats)
    print(
        f"  files={stats.stooq_files} rows={stats.stooq_rows} bad_rows={stats.stooq_bad_rows}",
        flush=True,
    )

    print(f"[2/4] FRED -> {out_dir / 'macro.csv'}", flush=True)
    _write_fred_macro(fred_root, out_dir / "macro.csv", stats)
    print(f"  files={stats.fred_files}", flush=True)

    print(f"[3/4] SEC submissions -> {out_dir / 'sec_submissions_entities.csv'}", flush=True)
    cik_to_ticker = _write_sec_submissions(
        submissions_root,
        out_dir / "sec_submissions_entities.csv",
        stats,
        max_files=max(0, int(args.max_sec_files)),
    )
    print(
        f"  files={stats.submissions_files} errors={stats.submissions_errors} ticker_map={len(cik_to_ticker)}",
        flush=True,
    )

    if args.skip_companyfacts:
        print("[4/4] SEC companyfacts skipped (--skip-companyfacts).", flush=True)
    else:
        print(f"[4/4] SEC companyfacts -> {out_dir / 'sec_companyfacts_selected.csv'}", flush=True)
        _write_sec_companyfacts(
            companyfacts_root,
            out_dir / "sec_companyfacts_selected.csv",
            cik_to_ticker=cik_to_ticker,
            keep_tags=tags,
            stats=stats,
            max_files=max(0, int(args.max_sec_files)),
        )
        print(
            f"  files={stats.companyfacts_files} errors={stats.companyfacts_errors} rows={stats.companyfacts_rows}",
            flush=True,
        )

    manifest = {
        "data_ff_root": str(data_ff_root),
        "out_dir": str(out_dir),
        "tags": sorted(tags),
        "max_sec_files": int(args.max_sec_files),
        "skip_companyfacts": bool(args.skip_companyfacts),
        "stats": asdict(stats),
        "outputs": [
            str(out_dir / "prices.csv"),
            str(out_dir / "macro.csv"),
            str(out_dir / "sec_submissions_entities.csv"),
            str(out_dir / "sec_companyfacts_selected.csv"),
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}", flush=True)

    if args.prune_source:
        if not (out_dir / "prices.csv").exists() or not (out_dir / "macro.csv").exists():
            raise RuntimeError("Refusing to prune source because core consolidated outputs are missing.")
        _safe_remove_tree(stooq_root)
        _safe_remove_tree(fred_root)
        _safe_remove_tree(submissions_root)
        if not args.skip_companyfacts:
            _safe_remove_tree(companyfacts_root)
        # remove DS_Store files left behind
        _safe_remove_files([data_ff_root / ".DS_Store"])
        print("Source folders pruned.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
