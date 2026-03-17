import os
import pandas as pd
import re
from pathlib import Path


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
    """
    Extract target strength from a directory name.
    Handles optional 'neg'/'pos' sign prefix and 'p' as decimal separator.
    Examples:
        boundary_neg0p5  -> -0.5
        boundary_neg0p4  -> -0.4
        flame_pos1p0     ->  1.0
    """
    match = re.search(r'(neg|pos)?(\d+)p(\d+)', dirname)
    if not match:
        raise ValueError(f"Could not parse target value from directory name: '{dirname}'")
    sign = -1 if match.group(1) == "neg" else 1
    integer_part = match.group(2)
    decimal_part = match.group(3)
    return sign * float(f"{integer_part}.{decimal_part}")


def load_optimization_results(
    result_dirs: list[str],
    base_dir: Path,
    folds: range,
) -> pd.DataFrame:
    """
    Load all optimization result TSVs from the given subdirectories,
    annotate with fold number and target strength, and concatenate.

    Args:
        result_dirs: List of subdirectory names under base_dir.
        base_dir:    Root directory containing the subdirectories.
        folds:       Iterable of fold indices (default 0–7).

    Returns:
        Concatenated DataFrame with added 'fold' and 'target' columns.
    """
    dfs = []

    for dirname in result_dirs:
        target = parse_target_from_dirname(dirname)
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
        raise RuntimeError("No TSV files were loaded. Check your paths and directory names.")

    return pd.concat(dfs, ignore_index=True)


def load_indep_runs_results(
    indep_runs_dir: Path,
    folds: range = range(4),
    seed_prefix: str = "seed",
) -> pd.DataFrame:
    """
    Load all optimization result TSVs from independent runs directory structure:
        indep_runs_dir/
            seed0/fold0_selected_genomic_windows_centered_chrom_states_results.tsv
            seed0/fold1_...
            seed1/fold0_...
            ...

    Args:
        indep_runs_dir: Path to the directory containing seed subdirectories.
        folds:          Iterable of fold indices (default 0–4).
        seed_prefix:    Prefix of seed subdirectory names (default "seed").

    Returns:
        Concatenated DataFrame with added 'fold' and 'run' columns.
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
