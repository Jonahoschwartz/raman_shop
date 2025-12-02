
import itertools
import os
import random
import re
import tarfile
import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

# IUPAC degenerate nucleotide codes mapping to regex patterns
# Used for flexible matching in DNA barcode flanking regions
IUPAC_CODES = {
    # Standard bases
    "A": "A", "C": "C", "G": "G", "T": "T",
    # Two-base degeneracies
    "R": "[AG]",  # puRine
    "Y": "[CT]",  # pYrimidine
    "S": "[GC]",  # Strong (3 H-bonds)
    "W": "[AT]",  # Weak (2 H-bonds)
    "K": "[GT]",  # Keto
    "M": "[AC]",  # aMino
    # Three-base degeneracies
    "B": "[CGT]",  # not A
    "D": "[AGT]",  # not C
    "H": "[ACT]",  # not G
    "V": "[ACG]",  # not T
    # Four-base degeneracy
    "N": "[ACGT]"  # aNy base
}

# Standard 20 amino acids for protein analysis
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Progress bar format for ColabFold MSA generation
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


# ==============================================================================
# DNA SEQUENCE ANALYSIS FUNCTIONS
# ==============================================================================

def iupac_to_regex(motif: str) -> str:
    """
    Convert an IUPAC-degenerate DNA motif into a regex pattern.

    This function enables flexible pattern matching by converting IUPAC ambiguity
    codes (like R for A or G) into their corresponding regex character classes.

    Args:
        motif (str): DNA sequence containing IUPAC codes (e.g., "ATGCN")

    Returns:
        str: Regex pattern equivalent (e.g., "ATGC[ACGT]")

    Examples:
        >>> iupac_to_regex("ATGR")
        'ATG[AG]'
        >>> iupac_to_regex("NNATGC")
        '[ACGT][ACGT]ATGC'
    """
    return "".join(IUPAC_CODES.get(base.upper(), base) for base in motif)


def extract_between_flanks(sequence: str, flanks: tuple[str, str]) -> Union[str, None]:
    """
    Extract substring between two degenerate flanking sequences in DNA.

    This function is commonly used for barcode extraction from amplicon sequences,
    where barcodes are located between known primer or adapter sequences that may
    contain IUPAC degeneracies.

    Args:
        sequence (str): Input DNA sequence (IUPAC-compliant)
        flanks (tuple[str, str]): 5' and 3' flanking sequences (may contain IUPAC codes)

    Returns:
        str: Extracted sequence between flanks
        "None": If either flank is missing or flanks are in wrong order
        "Multiple": If either flank occurs more than once
        "INVALID_SEQ": If sequence contains invalid characters

    Examples:
        >>> extract_between_flanks("ATGCCCGTAG", ("ATG", "TAG"))
        'CCCG'
        >>> extract_between_flanks("ATGCCCGTAGATGCCC", ("ATG", "TAG"))
        'Multiple'
    """
    # Validate input sequence for valid IUPAC characters
    if not isinstance(sequence, str) or not re.fullmatch(r"[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+", sequence):
        return "INVALID_SEQ"

    sequence = sequence.upper()
    flank_5, flank_3 = flanks

    # Convert IUPAC flanks to regex patterns for flexible matching
    regex_5 = iupac_to_regex(flank_5)
    regex_3 = iupac_to_regex(flank_3)

    # Find all occurrences of each flank
    matches_5 = list(re.finditer(regex_5, sequence))
    matches_3 = list(re.finditer(regex_3, sequence))

    # Handle cases where flanks are missing or occur multiple times
    if len(matches_5) == 0 or len(matches_3) == 0:
        return "None"
    if len(matches_5) > 1 or len(matches_3) > 1:
        return "Multiple"

    # Extract sequence between the flanks
    start = matches_5[0].end()  # Position after 5' flank
    end = matches_3[0].start()  # Position before 3' flank

    # Ensure flanks are in correct order (5' before 3')
    return sequence[start:end] if start <= end else "None"


# ==============================================================================
# DATA VISUALIZATION FUNCTIONS
# ==============================================================================

def filter_graph(
        series: pd.Series,
        num_cutoffs: int = 100,
        max_cutoff: float = None,
        color: str = "blue",
        marker: str = "o",
        title: str = "Percentage of Values Remaining vs. Cutoff",
        xlabel: str = "Cutoff Value",
        ylabel: str = "Percent of Values ≥ Cutoff",
        grid: bool = True,
        save_path: str = None,
) -> None:
    """
    Generate a filter graph showing percentage of values above various cutoffs.

    This visualization is particularly useful for determining optimal filtering
    thresholds for quality scores, expression levels, or other continuous metrics
    in bioinformatics data.

    Args:
        series (pd.Series): Numeric data to analyze
        num_cutoffs (int): Number of threshold values to test (default: 100)
        max_cutoff (float): Maximum cutoff value to consider (default: series max)
        color (str): Line color (default: "blue")
        marker (str): Point marker style (default: "o")
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        grid (bool): Whether to show background grid (default: True)
        save_path (str): Optional path to save the plot

    Raises:
        TypeError: If input is not a pandas Series
        ValueError: If series is empty or contains non-numeric values

    Examples:
        >>> quality_scores = pd.Series([10, 15, 20, 25, 30, 35, 40])
        >>> filter_graph(quality_scores, title="Quality Score Distribution")
    """
    # Input validation
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    series = series.dropna()
    if not np.issubdtype(series.dtype, np.number):
        raise ValueError("Series must contain numeric values.")
    if series.empty:
        raise ValueError("Series has no valid (non-null) entries.")

    # Determine range of cutoff values to test
    min_val = series.min()
    max_val = series.max() if max_cutoff is None else min(max_cutoff, series.max())
    cutoffs = np.unique(np.linspace(min_val, max_val, num_cutoffs))

    # Calculate percentage of values above each cutoff
    percentages = [(series >= cutoff).sum() / len(series) * 100 for cutoff in cutoffs]

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(cutoffs, percentages, marker=marker, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if grid:
        plt.grid(True)

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_barcode_hist(
        series: pd.Series,
        bins: Union[str, int, list] = "auto",
        title: str = "Histogram of String Frequencies",
        xlabel: str = "Number of Occurrences",
        ylabel: str = "Number of Unique Strings",
        color: str = "cornflowerblue",
        figsize: tuple = (10, 6),
        save_path: str = None,
        show_values: bool = False,
) -> None:
    """
    Create histogram showing frequency distribution of string values.

    This function is particularly useful for analyzing barcode distributions,
    showing how many barcodes appear once, twice, etc. This helps identify
    sequencing errors, PCR duplicates, or highly abundant sequences.

    Args:
        series (pd.Series): Series containing string values (e.g., DNA barcodes)
        bins (str|int|list): Histogram binning strategy (default: "auto")
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        color (str): Bar color (default: "cornflowerblue")
        figsize (tuple): Figure dimensions (default: (10, 6))
        save_path (str): Optional path to save the plot
        show_values (bool): Whether to annotate bars with counts (default: False)

    Examples:
        >>> barcodes = pd.Series(['ATGC', 'ATGC', 'GCTA', 'GCTA', 'GCTA', 'TTTT'])
        >>> plot_barcode_hist(barcodes, title="Barcode Frequency Distribution")
    """
    # Count occurrences of each unique string
    value_counts = series.value_counts()

    # Set up automatic binning based on maximum count
    if bins == "auto":
        bins = range(1, value_counts.max() + 2)

    # Configure seaborn style
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)

    # Create histogram of value frequencies
    ax = sns.histplot(value_counts, bins=bins, color=color, edgecolor='black')

    # Set labels and title
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Optionally annotate bars with their heights
    if show_values:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width() / 2, height + 0.5, int(height),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")

    plt.show()


# ==============================================================================
# POSITION-SPECIFIC SCORING MATRIX (PSSM) FUNCTIONS
# ==============================================================================

def pssm_from_DNA_msa(msa_file: str, output_json: bool = True) -> pd.DataFrame:
    """
    Generate Position-Specific Scoring Matrix (PSSM) from DNA multiple sequence alignment.

    A PSSM represents the log-likelihood of observing each nucleotide at each position
    in a sequence alignment, relative to background frequencies. This is useful for
    motif finding, binding site prediction, and sequence scoring.

    Args:
        msa_file (str): Path to FASTA format multiple sequence alignment file
        output_json (bool): Whether to save PSSM as JSON file (default: True)

    Returns:
        pd.DataFrame: PSSM with nucleotides as rows and positions as columns

    Notes:
        - Uses pseudocount of 0.25 to avoid log(0) issues
        - Scores are in log2 scale
        - Background frequency assumes equal probability (0.25) for each base

    Examples:
        >>> pssm = pssm_from_DNA_msa("alignment.fasta")
        >>> print(pssm.iloc[:, 0])  # Scores for first position
    """
    # Read alignment to determine length
    with open(msa_file, 'r') as f:
        first_tag = f.readline()
        first_aligned_seq = f.readline()
        alignment_length = len(first_aligned_seq.strip())

    # Initialize Position Frequency Matrix with pseudocounts
    # Pseudocount of 0.25 prevents log(0) and provides smoothing
    bases = ['A', 'G', 'C', 'T']
    pfm = pd.DataFrame(0.25, index=bases, columns=range(1, alignment_length + 1))

    # Count base occurrences at each position
    with open(msa_file, 'r') as f:
        for tag, seq in itertools.zip_longest(f, f, fillvalue=None):
            if seq is None:
                continue
            seq = seq.strip()
            for i, base in enumerate(seq):
                if base in pfm.index:
                    pfm.loc[base, i + 1] += 1

    # Convert frequencies to probabilities
    col_sum = pfm.sum(axis=0)
    ppm = pfm.divide(col_sum / 4)  # Normalize by expected frequency (0.25)

    # Convert to log2 scale to get PSSM
    pssm = ppm.applymap(np.log2)

    # Save as JSON if requested
    if output_json:
        out_path = f"{msa_file.split('/')[-1].split('.')[0]}_pssm.json"
        pssm.to_json(out_path)

    return pssm


def pssm_from_protein_msa(msa_file: str, output_json: bool = True) -> pd.DataFrame:
    """
    Generate Position-Specific Scoring Matrix (PSSM) from protein multiple sequence alignment.

    Similar to DNA PSSM but designed for protein sequences with 20 standard amino acids.
    Useful for protein motif analysis, domain identification, and sequence scoring.

    Args:
        msa_file (str): Path to FASTA format protein alignment file
        output_json (bool): Whether to save PSSM as JSON file (default: True)

    Returns:
        pd.DataFrame: PSSM with amino acids as rows and positions as columns

    Notes:
        - Uses pseudocount of 0.05 (smaller than DNA due to 20 vs 4 characters)
        - Scores are in log2 scale
        - Background frequency assumes equal probability (0.05) for each amino acid

    Examples:
        >>> pssm = pssm_from_protein_msa("protein_alignment.fasta")
        >>> conserved_pos = pssm.max(axis=0) > 2  # Highly conserved positions
    """
    # Read first sequence to determine alignment length
    with open(msa_file, 'r') as f:
        first_tag = f.readline()
        first_seq = f.readline()
        alignment_length = len(first_seq.strip())

    # Initialize Position Frequency Matrix with pseudocounts
    # Smaller pseudocount (0.05) due to larger alphabet size
    pfm = pd.DataFrame(0.05, index=AMINO_ACIDS, columns=range(1, alignment_length + 1))

    # Count amino acid occurrences at each position
    with open(msa_file, 'r') as f:
        for tag, seq in itertools.zip_longest(f, f, fillvalue=None):
            if seq is None:
                continue
            seq = seq.strip().upper()
            for i, aa in enumerate(seq):
                if aa in pfm.index:
                    pfm.loc[aa, i + 1] += 1

    # Convert frequencies to probabilities
    col_sum = pfm.sum(axis=0)
    ppm = pfm.divide(col_sum / 20)  # Normalize by expected frequency (0.05)

    # Convert to log2 scale to get PSSM
    pssm = ppm.applymap(np.log2)

    # Save as JSON if requested
    if output_json:
        out_path = f"{msa_file.split('/')[-1].split('.')[0]}_protein_pssm.json"
        pssm.to_json(out_path)

    return pssm


# ==============================================================================
# MULTIPLE SEQUENCE ALIGNMENT (MSA) FUNCTIONS
# ==============================================================================

def a3m_to_fasta(a3m_str: str) -> str:
    """
    Convert a3m format MSA to standard FASTA alignment format.

    The a3m format includes lowercase letters representing insertions relative
    to the consensus sequence. This function removes these insertions to create
    a standard aligned FASTA file.

    Args:
        a3m_str (str): Multiple sequence alignment in a3m format

    Returns:
        str: Alignment in FASTA format with insertions removed

    Notes:
        - Lowercase letters (insertions) are stripped from sequences
        - Header lines (starting with '>') are preserved
        - Results in a standard columnar alignment

    Examples:
        >>> a3m_data = ">seq1\\nATGCatgcGTAC\\n>seq2\\nATGC----GTAC"
        >>> fasta_data = a3m_to_fasta(a3m_data)
        >>> print(fasta_data)
        >seq1
        ATGCGTAC
        >seq2
        ATGC----GTAC
    """
    fasta_lines = []
    for line in a3m_str.splitlines():
        if line.startswith(">"):
            # Preserve header lines
            fasta_lines.append(line)
        else:
            # Remove lowercase letters (insertions) from sequence lines
            clean_line = re.sub(r"[a-z]", "", line)
            fasta_lines.append(clean_line)
    return "\n".join(fasta_lines)


def get_msa_from_sequence(sequence: str, prefix: str = "msa_output", save_dir: str = ".") -> tuple[str, str]:
    """
    Generate multiple sequence alignment from a single protein sequence using ColabFold.

    This function submits a protein sequence to the ColabFold MSA server, retrieves
    the resulting alignment, and saves it in both a3m and FASTA formats.

    Args:
        sequence (str): Input protein sequence (single-letter amino acid codes)
        prefix (str): Prefix for output filenames (default: "msa_output")
        save_dir (str): Directory to save alignment files (default: current directory)

    Returns:
        tuple[str, str]: Paths to (a3m_file, fasta_file)

    Notes:
        - Requires internet connection to access ColabFold servers
        - May take several minutes for large sequences
        - Creates temporary files during processing

    Examples:
        >>> a3m_path, fasta_path = get_msa_from_sequence("MKLLVLSLILSLVLVYII", "myprotein")
        >>> print(f"MSA saved to {fasta_path}")
    """
    # Generate MSA using ColabFold MMseqs2 server
    msa_a3m = run_mmseqs2(sequence, prefix)

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define output file paths
    a3m_path = os.path.join(save_dir, f"{prefix}_msa.a3m")
    fasta_path = os.path.join(save_dir, f"{prefix}_msa.fasta")

    # Save original a3m format
    with open(a3m_path, "w") as f:
        f.write(msa_a3m)

    # Convert to FASTA format and save
    msa_fasta = a3m_to_fasta(msa_a3m)
    with open(fasta_path, "w") as f:
        f.write(msa_fasta)

    return a3m_path, fasta_path


def run_mmseqs2(
        x: Union[str, list],
        prefix: str,
        use_env: bool = True,
        use_filter: bool = True,
        use_templates: bool = False,
        filter: Optional[bool] = None,
        host_url: str = "https://a3m.mmseqs.com"
) -> Union[str, list]:
    """
    Submit protein sequence(s) to ColabFold MMseqs2 server for MSA generation.

    This function interfaces with the ColabFold web service to generate multiple
    sequence alignments using the MMseqs2 algorithm. It handles job submission,
    status monitoring, and result retrieval.

    Args:
        x (str|list): Protein sequence(s) to align
        prefix (str): Directory prefix for temporary files
        use_env (bool): Include environmental sequences (default: True)
        use_filter (bool): Apply sequence filtering (default: True)
        use_templates (bool): Use template sequences (default: False)
        filter (bool): Override for use_filter parameter
        host_url (str): ColabFold server URL

    Returns:
        str|list: MSA in a3m format (string for single input, list for multiple)

    Raises:
        Exception: If server returns error or is under maintenance

    Notes:
        - Automatically handles job queuing and rate limiting
        - Downloads and extracts results from tar.gz archives
        - Caches results to avoid redundant server calls
        - May take several minutes depending on sequence length and server load
    """

    def submit(seqs: list, mode: str, N: int = 101) -> dict:
        """Submit sequences to ColabFold server."""
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            try:
                res = requests.post(f'{host_url}/ticket/msa',
                                    data={'q': query, 'mode': mode},
                                    timeout=6.02)
                break
            except requests.exceptions.Timeout:
                continue

        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def status(ID: str) -> dict:
        """Check job status on ColabFold server."""
        while True:
            try:
                res = requests.get(f'{host_url}/ticket/{ID}', timeout=6.02)
                break
            except requests.exceptions.Timeout:
                continue

        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def download(ID: str, path: str) -> None:
        """Download results from ColabFold server."""
        while True:
            try:
                res = requests.get(f'{host_url}/result/download/{ID}', timeout=6.02)
                break
            except requests.exceptions.Timeout:
                continue

        with open(path, "wb") as out:
            out.write(res.content)

    # Process input sequences
    seqs = [x] if isinstance(x, str) else x

    # Handle filter parameter override
    if filter is not None:
        use_filter = filter

    # Determine search mode based on options
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # Set up working directory
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    tar_gz_file = f'{path}/out.tar.gz'
    N, REDO = 101, True

    # Process sequences and create mapping
    seqs_unique = sorted(list(set(seqs)))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # Submit job if results don't already exist
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)  # Rough time estimate

        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")
                out = submit(seqs_unique, mode, N)

                # Handle rate limiting
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    time.sleep(5 + random.randint(0, 5))
                    out = submit(seqs_unique, mode, N)

                # Handle server errors
                if out["status"] == "ERROR":
                    raise Exception("MMseqs2 API error. Please check your input sequence.")

                if out["status"] == "MAINTENANCE":
                    raise Exception("MMseqs2 API is under maintenance. Try again later.")

                # Monitor job progress
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])

                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                # Download results when complete
                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

        download(ID, tar_gz_file)

    # Define expected output files
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env:
        a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract results if not already done
    if not os.path.isfile(a3m_files[0]):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Parse a3m files and organize by sequence ID
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file, "r") as f:
            for line in f:
                if len(line) > 0:
                    # Handle null characters in downloaded files
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True

                    # Parse sequence headers
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []

                    a3m_lines[M].append(line)

    # Reconstruct MSAs for requested sequences
    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    # Return single string for single input, list for multiple inputs
    if isinstance(x, str):
        return a3m_lines[0]
    else:
        return a3m_lines


def read_fasta(filepath):
    """
    Read a FASTA file and return a dictionary {header: sequence}.

    Parameters
    ----------
    filepath : str
        Path to the FASTA file.

    Returns
    -------
    dict
        Dictionary with FASTA headers (without ">") as keys
        and DNA sequences as values (uppercase, no line breaks).
    """
    sequences = {}
    header = None
    seq_chunks = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous entry
                if header:
                    sequences[header] = "".join(seq_chunks).upper()
                header = line[1:].split()[0]  # drop ">", take first token
                seq_chunks = []
            else:
                seq_chunks.append(line)
        # save last entry
        if header:
            sequences[header] = "".join(seq_chunks).upper()

    return sequences


def split_fastq_by_indices(fastq_file, index_csv, n=100, min_len=4000, output_dir="output_fastq"):
    """
    Splits a FASTQ file into multiple FASTQ files based on index pairs from a CSV.
    Matching is done on both standard and reverse complement sequences,
    and indexes can appear on either end of the read.

    Parameters:
        fastq_file (str or Path): Path to the input FASTQ file.
        index_csv (str or Path): CSV with columns: well,row,column
        n (int): Number of bases from each end to check against indexes.
        min_len (int): Minimum read length to consider.
        output_dir (str): Directory to store output FASTQ files.

    Returns:
        pandas.DataFrame: Table with read counts per well, plus totals.
    """
    fastq_file = Path(fastq_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load CSV mapping: (row_index, col_index) -> well
    index_map = {}
    with open(index_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            well = row['well'].strip()
            row_idx = row['row'].strip()
            col_idx = row['column'].strip()
            index_map[(row_idx, col_idx)] = well

    # Prepare file handles and counters
    well_files = {well: open(output_dir / f"{well}.fastq", 'w') for well in index_map.values()}
    well_counts = defaultdict(int)
    total_reads = 0
    filtered_reads = 0

    def revcomp(s):
        return str(Seq(s).reverse_complement())

    # Process FASTQ
    for record in SeqIO.parse(fastq_file, "fastq"):
        total_reads += 1
        seq = str(record.seq)

        if len(seq) < min_len:
            continue

        filtered_reads += 1

        seq_start = seq[:n]
        seq_end = seq[-n:]
        seq_rc = revcomp(seq)
        seq_rc_start = seq_rc[:n]
        seq_rc_end = seq_rc[-n:]

        assigned = False
        for (row_idx, col_idx), well in index_map.items():
            if ((row_idx in seq_start and col_idx in seq_end) or
                    (col_idx in seq_start and row_idx in seq_end) or
                    (row_idx in seq_rc_start and col_idx in seq_rc_end) or
                    (col_idx in seq_rc_start and row_idx in seq_rc_end)):
                SeqIO.write(record, well_files[well], "fastq")
                well_counts[well] += 1
                assigned = True
                break

        if not assigned:
            pass  # Optionally handle unassigned reads

    # Close files
    for f in well_files.values():
        f.close()

    # Create summary DataFrame
    all_wells = sorted(set(index_map.values()))
    df = pd.DataFrame({
        "well": all_wells,
        "assigned_reads": [well_counts[well] for well in all_wells]
    })

    df.loc[len(df)] = ["__TOTAL_ASSIGNED__", sum(well_counts.values())]
    df.loc[len(df)] = ["__TOTAL_FILTERED__", filtered_reads]
    df.loc[len(df)] = ["__TOTAL_INPUT__", total_reads]
    df.loc[len(df)] = ["__UNASSIGNED__", filtered_reads - sum(well_counts.values())]

    return df

def write_fasta(sequences, output_file, prefix="seq", wrap=80):
    """
    Write sequences to a FASTA file.
    Args:
        sequences (list of str or dict): List of sequences (strings) or dict of {header: sequence}.
        output_file (str): Path to the output FASTA file.
        prefix (str): Prefix for generated sequence headers if `sequences` is a list (default = 'seq').
        wrap (int): Number of characters per line in sequence (default = 80).
    """
    with open(output_file, "w") as f:
        if isinstance(sequences, dict):
            for header, seq in sequences.items():
                f.write(f">{header}\n")
                for i in range(0, len(seq), wrap):
                    f.write(f"{seq[i:i+wrap]}\n")
        elif isinstance(sequences, list):
            for i, seq in enumerate(sequences, start=1):
                f.write(f">{prefix}{i}\n")
                for j in range(0, len(seq), wrap):
                    f.write(f"{seq[j:j+wrap]}\n")
        else:
            raise TypeError("`sequences` must be a list of strings or a dict of {header: sequence}")


def load_fastq_sequences(filepath):
    """
    Load sequences from a FASTQ file into a list.

    Parameters:
    - filepath (str): Path to the FASTQ file.

    Returns:
    - List[str]: List of nucleotide sequences.
    """
    sequences = []
    with open(filepath, 'r') as f:
        while True:
            header = f.readline()
            if not header:
                break  # End of file
            seq = f.readline().strip()
            f.readline()  # Plus line
            f.readline()  # Quality line
            sequences.append(seq)
    return sequences
