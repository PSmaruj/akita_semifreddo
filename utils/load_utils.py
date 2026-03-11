import os
import pandas as pd

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