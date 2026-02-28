import logging
import pandas as pd

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