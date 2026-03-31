"""
utils/df_utils.py

DataFrame utilities for loading and summarizing AkitaSF optimization results.

Loading functions
-----------------
load_bed_fold               : load Akita BED file filtered to a single fold
load_parameter_results      : load result TSVs across a parameter sweep
load_optimization_results   : load result TSVs across target-strength directories
load_indep_runs_results     : load result TSVs from independent-run (seed) directories
simple_load_results         : load result TSVs from arbitrary subdirectories

Annotation helpers
------------------
build_optimization_table    : pair each window with the next as its target (circular shift)
parse_target_from_dirname   : extract target strength from a directory name
parse_dot_distance_from_dirname : extract dot distance from a directory name

Summary
-------
summarize_by_target         : per-target success rate and edit statistics
"""

import os
import pandas as pd
import re
from pathlib import Path
import logging


def load_parameter_results(
    results_dir: str,
    param_name: str,
    param_values: list,
    folds: list[int],
    tsv_suffix: str = "selected_genomic_windows_centered_chrom_states_results.tsv",
) -> pd.DataFrame:
    """Load optimisation result TSVs across parameter values and folds.

    Expects files at:
        <results_dir>/<param_name>/<param_name>_<value>/fold<fold>_<tsv_suffix>

    Parameters
    ----------
    results_dir : str
        Base results directory, e.g. `.../optimizations/boundaries`.
    param_name : str
        Swept parameter name, e.g. 'lambda', 'tau', 'eps'.
    param_values : list
        Values to load, e.g. [0.01, 0.1, 1.0].
    folds : list[int]
        Fold indices to load, e.g. [0, 1, 2].
    tsv_suffix : str
        Filename suffix after 'fold{fold}_'.

    Returns
    -------
    pd.DataFrame with columns 'fold' and <param_name> appended.
    """
    records = []
    for val in param_values:
        for fold in folds:
            path = os.path.join(
                results_dir,
                param_name,
                f"{param_name}_{val}",
                f"fold{fold}_{tsv_suffix}",
            )
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue
            df = pd.read_csv(path, sep="\t")
            df["fold"]       = fold
            df[param_name]   = val
            records.append(df)

    if not records:
        raise FileNotFoundError(f"No result files found for {param_name} in {results_dir}")

    df_all = pd.concat(records, ignore_index=True)
    print(f"Total windows loaded: {len(df_all)}")
    return df_all


def parse_target_from_dirname(dirname: str) -> float:
    """Extract target strength from a directory name.

    Handles optional 'neg'/'pos' sign prefix and 'p' as decimal separator.

    Parameters
    ----------
    dirname : str
        Directory name, e.g. 'boundary_neg0p5' or 'flame_pos1p0'.

    Returns
    -------
    float
        Parsed target value, e.g. -0.5 or 1.0.
    """
    match = re.search(r'(neg|pos)?(\d+)p(\d+)', dirname)
    if not match:
        raise ValueError(f"Could not parse target value from directory name: '{dirname}'")
    sign = -1 if match.group(1) == "neg" else 1
    integer_part = match.group(2)
    decimal_part = match.group(3)
    return sign * float(f"{integer_part}.{decimal_part}")


def parse_dot_distance_from_dirname(dirname: str) -> float:
    """Extract target dot distance from a directory name.

    Parameters
    ----------
    dirname : str
        Directory name, e.g. 'dot_d30' or 'dot_d50'.

    Returns
    -------
    float
        Parsed distance value, e.g. 30.0 or 50.0.
    """
    match = re.search(r'd(\d+)', dirname)
    if not match:
        raise ValueError(f"Could not parse dot distance from directory name: '{dirname}'")
    
    return float(match.group(1))


def load_optimization_results(
    result_dirs: list[str],
    base_dir: Path,
    folds: range,
    parser_func=parse_target_from_dirname
) -> pd.DataFrame:
    """Load optimization result TSVs across target-strength directories.

    Parameters
    ----------
    result_dirs : list[str]
        Subdirectory names under base_dir, e.g. ['boundary_neg0p5', 'boundary_neg0p4'].
    base_dir : Path
        Root directory containing the subdirectories.
    folds : range
        Fold indices to load, e.g. range(8).
    parser_func : callable
        Function mapping a directory name to a numeric target value.
        Defaults to parse_target_from_dirname.

    Returns
    -------
    pd.DataFrame
        Concatenated results with added 'fold' and 'target' columns.
    """
    dfs = []
    
    for dirname in result_dirs:
        # Use the passed parser function
        target = parser_func(dirname)
        dir_path = base_dir / dirname
        
        for fold in folds:
            fname = f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv"
            fpath = dir_path / fname
            
            if not fpath.exists():
                print(f"Warning: file not found, skipping: {fpath}")
                continue
                
            df = pd.read_csv(fpath, sep="\t")
            df["fold"] = fold
            df["target"] = target
            dfs.append(df)
            
    if not dfs:
        raise RuntimeError("No TSV files were loaded. Check your paths.")
    return pd.concat(dfs, ignore_index=True)


def load_indep_runs_results(
    indep_runs_dir: Path,
    folds: range = range(4),
    seed_prefix: str = "seed",
) -> pd.DataFrame:
    """Load optimization result TSVs from independent-run (seed) directories.

    Expects the following structure:
        indep_runs_dir/
            seed0/fold0_..._results.tsv
            seed0/fold1_..._results.tsv
            seed1/fold0_..._results.tsv
            ...

    Parameters
    ----------
    indep_runs_dir : Path
        Directory containing seed subdirectories.
    folds : range
        Fold indices to load (default range(4)).
    seed_prefix : str
        Prefix of seed subdirectory names (default 'seed').

    Returns
    -------
    pd.DataFrame
        Concatenated results with added 'fold' and 'run' columns.
    """
    seed_dirs = sorted(
        [d for d in indep_runs_dir.iterdir() if d.is_dir() and d.name.startswith(seed_prefix)],
        key=lambda d: int(d.name.removeprefix(seed_prefix)),
    )
    if not seed_dirs:
        raise RuntimeError(f"No seed directories found in {indep_runs_dir}")

    dfs = []
    for seed_dir in seed_dirs:
        run = int(seed_dir.name.removeprefix(seed_prefix))
        for fold in folds:
            fname = f"fold{fold}_selected_genomic_windows_centered_chrom_states_results.tsv"
            fpath = seed_dir / fname
            if not fpath.exists():
                print(f"Warning: file not found, skipping: {fpath}")
                continue
            df = pd.read_csv(fpath, sep="\t")
            df["fold"] = fold
            df["run"] = run
            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No TSV files were loaded from {indep_runs_dir}")

    return pd.concat(dfs, ignore_index=True)


def summarize_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize optimization outcomes per target strength.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated results table with columns [target, n_edits,
        optimization_success, last_accepted_step].

    Returns
    -------
    pd.DataFrame
        One row per target strength with columns:
        - n_total           : total number of optimization windows
        - n_no_edits        : windows where no edits were accepted
        - n_no_depletion    : windows with edits but optimization_success == False
        - success_rate_pct  : percentage of successful optimizations
        - mean_n_edits      : mean accepted edits (successful runs only)
        - mean_last_step    : mean last accepted step (successful runs only)
    """
    rows = []
    for target, grp in df.groupby("target"):
        n_total       = len(grp)
        no_edits      = grp["n_edits"] == 0
        successful    = grp["optimization_success"]

        n_no_edits    = no_edits.sum()
        n_no_depletion = (~successful & ~no_edits).sum()
        success_rate  = 100 * successful.mean()

        successful_grp = grp[successful]
        mean_n_edits   = successful_grp["n_edits"].mean()
        mean_last_step = successful_grp["last_accepted_step"].mean()

        rows.append({
            "target":            target,
            "n_total":           n_total,
            "n_no_edits":        n_no_edits,
            "n_no_depletion":    n_no_depletion,
            "success_rate_pct":  round(success_rate, 1),
            "mean_n_edits":      round(mean_n_edits, 1),
            "mean_last_step":    round(mean_last_step, 1),
        })

    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def simple_load_results(
    result_dirs: list[str],
    base_dir: Path,
    folds: range,
    tsv_suffix: str,
) -> pd.DataFrame:
    """Load result TSVs from arbitrary subdirectories and concatenate.

    Parameters
    ----------
    result_dirs : list[str]
        Subdirectory names under base_dir.
    base_dir : Path
        Root directory containing the subdirectories.
    folds : range
        Fold indices to load.
    tsv_suffix : str
        Filename suffix; expected pattern is fold{N}_{tsv_suffix}.

    Returns
    -------
    pd.DataFrame
        Concatenated results from all found TSV files.
    """
    dfs = []

    for dirname in result_dirs:
        dir_path = base_dir / dirname

        for fold in folds:
            fname = f"fold{fold}_{tsv_suffix}"
            fpath = dir_path / fname

            if not fpath.exists():
                print(f"Warning: file not found, skipping: {fpath}")
                continue

            df = pd.read_csv(fpath, sep="\t")
            dfs.append(df)

    if not dfs:
        raise RuntimeError("No TSV files were loaded. Check your paths.")
    return pd.concat(dfs, ignore_index=True)


def load_bed_fold(bed_file: str, fold: int) -> pd.DataFrame:
    """Load the Akita BED file and filter to a specific fold.

    Parameters
    ----------
    bed_file:
        Path to the sequences.bed file (chrom, start, end, fold columns).
    fold:
        Integer fold index to select (matched against the string "fold{fold}").

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with columns chrom, start, end, fold.
    """
    df = pd.read_csv(bed_file, sep="\t", header=None, names=["chrom", "start", "end", "fold"])
    df_fold = df[df["fold"] == f"fold{fold}"].reset_index(drop=True)
    logging.info(f"Loaded {len(df_fold)} windows for fold {fold}.")
    return df_fold


def build_optimization_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each window with the next one as its optimization target.

    Adds target_chrom, target_start, and target_end columns by shifting the
    coordinate columns by one row (circular: the last window's target is the
    first window). Also initializes a last_accepted_step column to -1.

    Parameters
    ----------
    df : pd.DataFrame
        Table with columns [chrom, start, end].

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns: target_chrom, target_start,
        target_end, last_accepted_step.
    """
    df = df.copy()
    df["target_chrom"] = df["chrom"].shift(-1).fillna(df["chrom"].iloc[0])
    df["target_start"] = df["start"].shift(-1).fillna(df["start"].iloc[0]).astype(int)
    df["target_end"]   = df["end"].shift(-1).fillna(df["end"].iloc[0]).astype(int)
    df["last_accepted_step"] = -1
    return df