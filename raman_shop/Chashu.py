"""
DNA Sequence Manipulation and Analysis Utilities

This module provides functions for:
- Random sequence generation
- Primer design for mutagenesis
- Restriction site analysis
- Quality control for oligonucleotide pools
- Sequence translation and motif visualization
"""

import os
import re
import glob
from typing import List, Union, Dict
from itertools import product

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.Restriction import RestrictionBatch
from Bio.Data import IUPACData

# ============================================================================
# CONSTANTS
# ============================================================================

DEGENERATE_CODES = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
    'K': ['G', 'T'], 'M': ['A', 'C'],
    'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'], 'N': ['A', 'C', 'G', 'T']
}

CODON_TABLE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
    'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
    'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
    'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
    'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
    'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
}


# ============================================================================
# SEQUENCE GENERATION
# ============================================================================

def randnt(n: int, gc_content: float = 0.5) -> str:
    """
    Generate a random nucleotide sequence with specified length and GC content.

    Parameters
    ----------
    n : int
        Desired length of random nucleotide string in basepairs.
    gc_content : float, optional
        Desired proportion of string that will be G's and C's. Default is 0.5.

    Returns
    -------
    str
        Random nucleotide sequence with user-specified length and GC content.

    Raises
    ------
    ValueError
        If n is not an integer or gc_content is not a float.
    AssertionError
        If gc_content is not between 0.0 and 1.0.
    """
    if not isinstance(n, int):
        raise ValueError("n input must be an integer")
    if not isinstance(gc_content, float):
        raise ValueError("gc_content input must be a float")
    assert 0.0 <= gc_content <= 1.0, 'gc_content must be between 0.0 and 1.0'

    gcs = int(n * gc_content)
    ats = n - gcs

    gc_list = [random.choice(['G', 'C']) for _ in range(gcs)]
    at_list = [random.choice(['A', 'T']) for _ in range(ats)]

    seq = at_list + gc_list
    random.shuffle(seq)

    return ''.join(seq)


def simple_randnt(n: int) -> str:
    """
    Generate a fully random nucleotide sequence of length n.

    Parameters
    ----------
    n : int
        Desired sequence length.

    Returns
    -------
    str
        Random nucleotide sequence.
    """
    return ''.join(random.choice(['G', 'C', 'A', 'T']) for _ in range(n))


# ============================================================================
# SEQUENCE MANIPULATION
# ============================================================================

def reverse_complement(sequence: str) -> str:
    """
    Calculate the reverse complement of a DNA sequence.

    Parameters
    ----------
    sequence : str
        The input DNA sequence (e.g., "AGCT").

    Returns
    -------
    str
        The reverse complemented DNA sequence.
    """
    complement_map = {
        'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
        'a': 't', 't': 'a', 'c': 'g', 'g': 'c'
    }

    reversed_sequence = sequence[::-1]
    complemented_sequence = "".join(
        complement_map.get(base, base) for base in reversed_sequence
    )

    return complemented_sequence


def translate(dna: str) -> str:
    """
    Translate DNA to protein using standard codon table (stops='*').

    Parameters
    ----------
    dna : str
        DNA sequence to translate.

    Returns
    -------
    str
        Translated protein sequence with 'X' for unknown codons and '*' for stops.
    """
    protein = []
    dna = dna.upper().replace("U", "T")

    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3]
        protein.append(CODON_TABLE.get(codon, "X"))

    return "".join(protein)


def mutation_string_from_dna(dna1: str, dna2: str) -> List[str]:
    """
    Translate two DNA ORFs and return mutation strings.

    Parameters
    ----------
    dna1 : str
        First DNA sequence (wild-type).
    dna2 : str
        Second DNA sequence (mutant).

    Returns
    -------
    List[str]
        List of mutations in format: OLD_AA + position + NEW_AA.
        Returns ["WT"] if sequences are identical.
        Only substitutions are reported (not indels).
    """
    prot1 = translate(dna1)
    prot2 = translate(dna2)
    mutations = []
    length = min(len(prot1), len(prot2))

    for pos in range(length):
        aa1, aa2 = prot1[pos], prot2[pos]
        if aa1 != aa2:
            mutations.append(f"{aa1}{pos + 1}{aa2}")  # 1-based position

    return mutations if mutations else ["WT"]


# ============================================================================
# DEGENERATE SEQUENCE HANDLING
# ============================================================================

def degenerate_to_regex(seq: str) -> str:
    """
    Convert a degenerate DNA sequence into a regex pattern.

    Parameters
    ----------
    seq : str
        DNA sequence with IUPAC degenerate codes.

    Returns
    -------
    str
        Regular expression pattern matching all possible sequences.
    """
    iupac_dict = {**IUPACData.ambiguous_dna_values}
    regex = ""

    for base in seq.upper():
        if base in iupac_dict:
            regex += f"[{iupac_dict[base]}]"
        else:
            regex += base

    return regex


def generate_sequences(degenerate_sequences: List[str]) -> Dict[str, List[str]]:
    """
    Generate all possible sequences from degenerate sequence codes.

    Parameters
    ----------
    degenerate_sequences : List[str]
        List of sequences containing degenerate bases.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping each degenerate sequence to all possible expansions.
    """
    all_possible_sequences = {}

    for seq in degenerate_sequences:
        options = [DEGENERATE_CODES[b] for b in seq]
        combos = product(*options)
        all_possible_sequences[seq] = [''.join(c) for c in combos]

    return all_possible_sequences


# ============================================================================
# RESTRICTION SITE ANALYSIS
# ============================================================================

def find_restriction_site_counts(
        sequence: str,
        enzymes: List = None,
        recognition_seqs: List[str] = None
) -> dict:
    """
    Count the number of recognition sites in both strands of a DNA sequence.

    Parameters
    ----------
    sequence : str
        DNA sequence (5' to 3').
    enzymes : List, optional
        List of Biopython Restriction enzyme classes (e.g., [EcoRI, HindIII]).
    recognition_seqs : List[str], optional
        List of recognition site strings, may include degenerate bases.

    Returns
    -------
    dict
        Keys are enzyme or recognition sequence names, values are counts of
        sites found in both strands.
    """
    seq = sequence.upper()
    rev_seq = str(Seq(seq).reverse_complement())
    results = {}

    # Use Biopython enzymes
    if enzymes:
        rb = RestrictionBatch(enzymes)
        fwd_hits = rb.search(Seq(seq))
        rev_hits = rb.search(Seq(rev_seq))

        for enzyme in rb.enzymes:
            count = len(fwd_hits.get(enzyme, [])) + len(rev_hits.get(enzyme, []))
            results[str(enzyme)] = count

    # Use raw recognition sequences with degenerate base support
    if recognition_seqs:
        for recog in recognition_seqs:
            recog = recog.upper()
            recog_regex = degenerate_to_regex(recog)
            pattern = re.compile(recog_regex)

            fwd_matches = pattern.findall(seq)
            rev_matches = pattern.findall(rev_seq)
            results[recog] = len(fwd_matches) + len(rev_matches)

    return results


# ============================================================================
# PRIMER DESIGN FOR MUTAGENESIS
# ============================================================================

def generate_suni_mutagensis_primers(
        WT_seq: str,
        left_flank: str,
        right_flank: str,
        tm_left_min: float = 59,
        tm_left_max: float = 66,
        target_tm_right: float = 61,
        min_len_left: int = 18,
        max_len_left: int = 40,
        min_len_right: int = 18,
        max_len_right: int = 40
) -> dict:
    """
    Generate a smart nicking library with adjustable homology arm parameters.

    Parameters
    ----------
    WT_seq : str
        The wild-type coding sequence.
    left_flank : str
        Flanking sequence upstream of WT_seq.
    right_flank : str
        Flanking sequence downstream of WT_seq.
    tm_left_min : float, optional
        Minimum acceptable Tm for left homology arm. Default is 59.
    tm_left_max : float, optional
        Maximum acceptable Tm for left homology arm. Default is 66.
    target_tm_right : float, optional
        Target Tm for right homology arm. Default is 61.
    min_len_left : int, optional
        Minimum length of the left homology arm. Default is 18.
    max_len_left : int, optional
        Maximum length of the left homology arm. Default is 40.
    min_len_right : int, optional
        Minimum length of the right homology arm. Default is 18.
    max_len_right : int, optional
        Maximum length of the right homology arm. Default is 40.

    Returns
    -------
    dict
        Dictionary of oligos with codon positions as keys.
    """
    codon_num = int(len(WT_seq) / 3)
    full_seq = left_flank + WT_seq + right_flank
    oligos_dict = {}

    # Statistics tracking
    best_left_lens = []
    best_right_lens = []
    best_left_tms = []
    best_right_tms = []
    best_lens = []
    best_tms = []

    SSSs = 0
    SSWs = 0
    SWSs = 0
    weak_clamps = 0
    NNSs = 0
    NNKs = 0

    SW_dict = {'A': 'W', 'T': 'W', 'C': 'S', 'G': 'S'}

    for codon in range(1, codon_num):
        codon_start = len(left_flank) + codon * 3
        codon_stop = codon_start + 3
        wt_codon = full_seq[codon_start:codon_stop]

        # Find best left homology arm
        temp_dict_left = {}
        temp_winners_list = []
        SW_winners_dict = {'SSS': [], 'SSW': [], 'SWS': []}

        for i in range(min_len_left, max_len_left + 1):
            left_tm = mt.Tm_NN(Seq(full_seq[codon_start - i:codon_start]))
            temp_dict_left[i] = left_tm

            if tm_left_min <= left_tm <= tm_left_max:
                temp_winners_list.append(i)
                left_clamp = ''.join([
                    SW_dict[base]
                    for base in full_seq[codon_start - i:codon_start][0:3]
                ])
                if left_clamp in SW_winners_dict:
                    SW_winners_dict[left_clamp].append(i)

        winner_found = False
        best_len = 0
        best_tm = 0
        double_synth = False

        # Prioritize SSS, then SSW, then SWS clamps
        for clamp_type in ['SSS', 'SSW', 'SWS']:
            if not winner_found:
                for i in SW_winners_dict[clamp_type]:
                    best_len = i
                    best_tm = temp_dict_left[i]
                    winner_found = True

                    if clamp_type == 'SSS':
                        SSSs += 1
                    elif clamp_type == 'SSW':
                        SSWs += 1
                    elif clamp_type == 'SWS':
                        SWSs += 1
                    break

        if winner_found:
            best_left_seq = full_seq[codon_start - best_len:codon_start]
            best_lens.append(best_len)
            best_left_lens.append(best_len)
            best_tms.append(best_tm)
            best_left_tms.append(best_tm)
        else:
            # Adjust for lack of good clamp
            best_len, best_tm = min(
                temp_dict_left.items(),
                key=lambda x: abs((tm_left_min + 4) - x[1])
            )
            best_left_seq = full_seq[codon_start - best_len:codon_start]
            best_left_lens.append(best_len)
            best_tms.append(best_tm)
            best_left_tms.append(best_tm)
            double_synth = True
            weak_clamps += 1

        # Find best right homology arm
        temp_dict_right = {
            i: mt.Tm_NN(Seq(full_seq[codon_stop:codon_stop + i]))
            for i in range(min_len_right, max_len_right + 1)
        }
        best_len, best_tm = min(
            temp_dict_right.items(),
            key=lambda x: abs(target_tm_right - x[1])
        )
        best_right_seq = full_seq[codon_stop:codon_stop + best_len]
        best_lens.append(best_len)
        best_right_lens.append(best_len)
        best_tms.append(best_tm)
        best_right_tms.append(best_tm)

        # Choose degenerate codon based on WT codon
        if wt_codon[-1] in ['A', 'C', 'G']:
            degen_codon = 'NNK'
            NNKs += 1
        else:
            degen_codon = 'NNS'
            NNSs += 1

        # Assemble oligo
        oligo = best_left_seq + degen_codon + best_right_seq
        oligos_dict[codon + 1] = oligo

        if double_synth:
            oligos_dict[str(codon + 1) + 'repeat'] = oligo

    # Plot statistics
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.violinplot(
        ax=axes[0],
        data=[best_left_tms, best_right_tms],
        color='salmon',
        cut=0
    )
    axes[0].set_xlabel('Best tms')
    axes[0].set_xticklabels(['left', 'right'])

    sns.violinplot(
        ax=axes[1],
        data=[best_left_lens, best_right_lens],
        color='cornflowerblue',
        cut=0
    )
    axes[1].set_xlabel('Best lens')
    axes[1].set_xticklabels(['left', 'right'])

    # Print summary statistics
    total = codon_num - 1
    print(f'total = {total}')
    print(f'SSS = {SSSs}, {SSSs / total:.3f}')
    print(f'SSW = {SSWs}, {SSWs / total:.3f}')
    print(f'SWS = {SWSs}, {SWSs / total:.3f}')
    print(f'weak_clamps = {weak_clamps}, {weak_clamps / total:.3f}')
    print(f'NNKs = {NNKs}')
    print(f'NNSs = {NNSs}')

    return oligos_dict


def gen_smart_nicking_lib(
        WT_seq: str,
        left_flank: str,
        right_flank: str,
        tm_left_min: float,
        tm_left_max: float,
        target_tm_right: float,
        min_len_left: int,
        max_len_left: int,
        min_len_right: int,
        max_len_right: int
) -> dict:
    """
    Generate smart nicking library (alternative implementation).

    From: https://github.com/lehner-lab/SUNi_mutagenesis/blob/main/SUNi_mutagenesis.ipynb

    This is the number of codons INCLUDING the start codon, which we will not mutagenize.

    Parameters
    ----------
    WT_seq : str
        Wild-type coding sequence.
    left_flank : str
        Upstream flanking sequence.
    right_flank : str
        Downstream flanking sequence.
    tm_left_min : float
        Minimum Tm for left arm.
    tm_left_max : float
        Maximum Tm for left arm.
    target_tm_right : float
        Target Tm for right arm.
    min_len_left : int
        Minimum left arm length.
    max_len_left : int
        Maximum left arm length.
    min_len_right : int
        Minimum right arm length.
    max_len_right : int
        Maximum right arm length.

    Returns
    -------
    dict
        Dictionary of designed oligos.
    """
    codon_num = int(len(WT_seq) / 3)
    full_seq = left_flank + WT_seq + right_flank
    oligos_dict = {}

    best_left_lens = []
    best_right_lens = []
    best_left_tms = []
    best_right_tms = []
    best_lens = []
    best_tms = []

    SSSs = 0
    SSWs = 0
    SWSs = 0
    weak_clamps = 0
    NNSs = 0
    NNKs = 0

    SW_dict = {'A': 'W', 'T': 'W', 'C': 'S', 'G': 'S'}

    # Start the range with 1, so that we skip the start codon
    for codon in range(1, codon_num):
        # Define the codon
        codon_start = len(left_flank) + codon * 3
        codon_stop = len(left_flank) + (codon * 3) + 3
        wt_codon = full_seq[codon_start:codon_stop]

        # Find best homology arm on the left
        temp_dict_left = {}
        clamp_dict_left = {}
        temp_winners_list = []
        SW_winners_dict = {'SSS': [], 'SSW': [], 'SWS': []}

        for i in range(min_len_left, max_len_left + 1):
            left_tm = mt.Tm_NN(Seq(full_seq[codon_start - i:codon_start]))
            temp_dict_left[i] = left_tm

            if tm_left_min <= left_tm <= tm_left_max:
                temp_winners_list.append(i)
                left_clamp = ''.join([
                    SW_dict[j]
                    for j in full_seq[codon_start - i:codon_start][0:3]
                ])
                if left_clamp in ['SSS', 'SSW', 'SWS']:
                    SW_winners_dict[left_clamp].append(i)

        winner_found = False
        best_len = 0
        best_tm = 0
        double_synth = False

        # First look for SSS
        for i in temp_winners_list:
            if not winner_found:
                if i in SW_winners_dict['SSS']:
                    best_len = i
                    best_tm = temp_dict_left[i]
                    SSSs += 1
                    winner_found = True

        # If no SSS found, then look for SSW or SWS
        for i in temp_winners_list:
            if not winner_found:
                if i in SW_winners_dict['SSW']:
                    best_len = i
                    best_tm = temp_dict_left[i]
                    SSWs += 1
                    winner_found = True
                elif i in SW_winners_dict['SWS']:
                    best_len = i
                    best_tm = temp_dict_left[i]
                    SWSs += 1
                    winner_found = True

        if winner_found:
            best_left_seq = full_seq[codon_start - best_len:codon_start]
            best_lens.append(best_len)
            best_left_lens.append(best_len)
            best_tms.append(best_tm)
            best_left_tms.append(best_tm)
        else:
            # Try to adjust for the lack of good clamp, use Tm + 4
            best_len, best_tm = min(
                temp_dict_left.items(),
                key=lambda x: abs((tm_left_min + 4) - x[1])
            )
            best_left_seq = full_seq[codon_start - best_len:codon_start]
            best_tms.append(best_tm)
            best_left_tms.append(best_tm)
            double_synth = True
            weak_clamps += 1

        # Find best homology arm on the right
        temp_dict_right = {}
        for i in range(min_len_right, max_len_right + 1):
            temp_dict_right[i] = mt.Tm_NN(
                Seq(full_seq[codon_stop:codon_stop + i])
            )

        best_len, best_tm = min(
            temp_dict_right.items(),
            key=lambda x: abs(target_tm_right - x[1])
        )
        best_right_seq = full_seq[codon_stop:codon_stop + best_len]
        best_lens.append(best_len)
        best_right_lens.append(best_len)
        best_tms.append(best_tm)
        best_right_tms.append(best_tm)

        # Assemble the oligo
        if wt_codon[-1] in ['A', 'C', 'G']:
            degen_codon = 'NNK'
            NNKs += 1
        elif wt_codon[-1] in ['T']:
            degen_codon = 'NNS'
            NNSs += 1

        oligo = best_left_seq + degen_codon + best_right_seq
        oligos_dict[codon + 1] = oligo

        if double_synth:
            oligos_dict[str(codon + 1) + 'repeat'] = oligo

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.violinplot(
        ax=axes[0],
        data=[best_left_tms, best_right_tms],
        color='salmon',
        cut=0
    )
    axes[0].set_xlabel('Best tms')
    axes[0].set_xticklabels(['left', 'right'])

    sns.violinplot(
        ax=axes[1],
        data=[best_left_lens, best_right_lens],
        color='cornflowerblue',
        cut=0
    )
    axes[1].set_xlabel('Best lens')
    axes[1].set_xticklabels(['left', 'right'])

    # Print statistics (divide by codon_num-1, since we aren't mutating the start codon)
    total = codon_num - 1
    print(f'total = {total}')
    print(f'SSS = {SSSs}, {SSSs / total:.3f}')
    print(f'SSW = {SSWs}, {SSWs / total:.3f}')
    print(f'SWS = {SWSs}, {SWSs / total:.3f}')
    print(f'weak_clamps = {weak_clamps}, {weak_clamps / total:.3f}')
    print(f'NNKs = {NNKs}')
    print(f'NNSs = {NNSs}')

    return oligos_dict


# ============================================================================
# FILE I/O
# ============================================================================

def dict_to_fastq(seq_dict: dict, output_path: str) -> None:
    """
    Save a dictionary of DNA sequences as a FASTQ file.

    Parameters
    ----------
    seq_dict : dict
        Dictionary where keys are sequence IDs and values are DNA sequences.
    output_path : str
        Output FASTQ file path.
    """
    with open(output_path, 'w') as f:
        for seq_id, sequence in seq_dict.items():
            # 1) FASTQ header line starts with '@'
            f.write(f"@{seq_id}\n")

            # 2) The DNA sequence
            f.write(f"{sequence}\n")

            # 3) Separator line, usually just '+'
            f.write("+\n")

            # 4) Quality string (same length as the sequence). 'I' ~ high quality
            quality = "I" * len(sequence)
            f.write(f"{quality}\n")


# ============================================================================
# QUALITY CONTROL
# ============================================================================

def run_oligo_qc_nnovy(
        pool_primers_path: str,
        data_base_path: str,
        available_restriction_sites: List[str]
) -> pd.DataFrame:
    """
    Perform quality control checks on oligonucleotide sequences before synthesis.

    Parameters
    ----------
    pool_primers_path : str
        Path to the Excel/CSV file containing forward and reverse primers.
        Expected columns: 'fwd', 'rev'.
    data_base_path : str
        Directory path containing the input oligo design files (.csv, .xlsx, etc).
        Each file represents a separate design objective.
    available_restriction_sites : List[str]
        List of restriction enzyme recognition sequences expected in each oligo
        (e.g., ['GGTCTC', 'CGTCTC']).

    Returns
    -------
    pd.DataFrame
        A combined DataFrame of all oligos that have passed quality control,
        with issues logged to stdout.
    """

    # ---------------------------- Helper Functions ----------------------------

    def read_file(filename: str) -> pd.DataFrame:
        """Read a file into a pandas DataFrame, auto-detecting format."""
        ext = filename.split('.')[-1]

        if ext == 'tsv':
            return pd.read_csv(filename, sep='\t')
        elif ext == 'csv':
            return pd.read_csv(filename)
        elif ext == 'txt':
            return pd.read_csv(filename, sep=" ", header=None)
        elif ext in ['xlsx', 'xls']:
            return pd.read_excel(filename, index_col=0)
        elif ext == 'pkl':
            return pd.read_pickle(filename)

        raise ValueError(f"Unsupported file extension: {ext}")

    def determine_complement(nuc_sequence: str, invert: bool = True) -> str:
        """Return the reverse complement of a DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        comp_seq = ''.join(complement.get(base, base) for base in nuc_sequence)
        return comp_seq[::-1] if invert else comp_seq

    def reformat_dfs(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Standardize input DataFrames:
        - Ensure oligo sequences are stored under the 'oligo' column.
        - Convert all sequences to uppercase.
        """
        for k, v in df_dict.items():
            if 'oligo_seq' in v.columns:
                v.rename(columns={'oligo_seq': 'oligo'}, inplace=True)
            elif len(v.columns) == 0:
                v = pd.DataFrame(v.index.tolist(), columns=['oligo'])
            v['oligo'] = v['oligo'].str.upper()
            df_dict[k] = v
        return df_dict

    def return_filename_of_seq(
            seq: str,
            df_dict: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Return a list of filenames in which a given sequence appears."""
        return [f'{k}: {seq}' for k, v in df_dict.items() if seq in v['oligo'].values]

    def count_cut_sites(sequence: str, cut_site: str) -> int:
        """
        Count the total number of occurrences of a restriction site and its
        reverse complement in a sequence.
        """
        sites = [cut_site, determine_complement(cut_site, invert=True)]
        return sum(sequence.count(site) for site in sites)

    # ---------------------------- Load Inputs ----------------------------

    # Load pool primer pairs (fwd, rev)
    pool_df = read_file(pool_primers_path)

    # Load all design files in the directory
    file_paths = glob.glob(os.path.join(data_base_path, '*'))
    df_dict = {
        os.path.basename(f): read_file(f)
        for f in file_paths if 'combined_seqs_to_order.csv' not in f
    }

    # Standardize sequence formatting
    df_dict = reformat_dfs(df_dict)

    # Combine all datasets into a single DataFrame
    all_dfs = pd.concat(df_dict.values(), ignore_index=True)

    # ---------------------------- Duplicate Sequence Check ----------------------------

    duplicates = all_dfs[all_dfs.duplicated('oligo', keep=False)]
    unique_duplicates = set(duplicates['oligo'])

    if unique_duplicates:
        print('ATTENTION, duplicate sequences found across datasets:')
        for dup in unique_duplicates:
            for ref in return_filename_of_seq(dup, df_dict):
                print(f"{ref} occurs {sum(all_dfs['oligo'] == dup)} times")
    else:
        print('No duplicates found across design files.\n')

    # ---------------------------- Length Check ----------------------------

    max_length = max(all_dfs['oligo'].str.len())
    threshold = int(max_length * 0.9)
    short_oligos = set(all_dfs[all_dfs['oligo'].str.len() < threshold]['oligo'])

    print(f"Max oligo length: {max_length} nt")

    if short_oligos:
        print(f"ATTENTION, sequences shorter than {threshold} nt:")
        for seq in short_oligos:
            for ref in return_filename_of_seq(seq, df_dict):
                print(ref)
    else:
        print(f"All sequences are between {threshold} and {max_length} nt.\n")

    # ---------------------------- Restriction Site Check ----------------------------

    incorrect_rest_sites = []
    detected_cut_sites = []

    for _, row in all_dfs.iterrows():
        oligo = row['oligo']

        # Try to detect the restriction site between position 10-30
        probable_cut = oligo[10:30]
        match = [rs for rs in available_restriction_sites if rs in probable_cut]

        # Expand the search window if needed
        idx = 30
        while not match and idx < len(oligo) // 2:
            idx += 1
            match = [rs for rs in available_restriction_sites if rs in oligo[10:idx]]

        if not match:
            print(f"WARNING: No valid restriction site found in sequence:\n{oligo}")
            continue

        cut_site = match[0]
        detected_cut_sites.append(cut_site)

        # Count how many times the site (and rev. comp.) occurs
        if count_cut_sites(oligo, cut_site) != 2:
            incorrect_rest_sites.append(oligo)

    if incorrect_rest_sites:
        print("ATTENTION: Sequences with incorrect number of cut sites:")
        for seq in set(incorrect_rest_sites):
            print(return_filename_of_seq(seq, df_dict))
    else:
        print("All sequences have exactly two cut sites.\n")

    # Print how often each cut site occurs
    print("Detected cut site frequencies:")
    for site in available_restriction_sites:
        print(f"{site}: {detected_cut_sites.count(site)} sequences")
    for k, v in df_dict.items():
        print(f"{k}: {len(v)} oligos")

    # ---------------------------- Primer Overlap Check ----------------------------

    primer_dict = {}
    for name, df in df_dict.items():
        primers = df['oligo'].str[:15].unique()
        try:
            primer_idxs = [pool_df[pool_df['fwd'] == p].index[0] for p in primers]
            primer_dict[name] = primer_idxs
        except:
            pass  # Skip if primer not found in pool_df

    # Check if the same primer is used in multiple pools
    overlaps = []
    for k1, v1 in primer_dict.items():
        for k2, v2 in primer_dict.items():
            if k1 != k2:
                for idx in set(v1) & set(v2):
                    overlaps.append(
                        f"ERROR: {k1} and {k2} both use primer index {idx}"
                    )

    if overlaps:
        print('\n'.join(overlaps))
    else:
        print("No overlapping primers identified across pools.\n")

    # ---------------------------- Primer Pair Consistency ----------------------------

    print("Checking forward/reverse primer consistency...")
    error_seqs = []
    all_fw_idxs = set(i for sub in primer_dict.values() for i in sub)

    for _, row in all_dfs.iterrows():
        oligo = row['oligo']

        # Search for matching fwd and rev primers in sequence
        fws = [
            pool_df.loc[i, 'fwd']
            for i in all_fw_idxs
            if pool_df.loc[i, 'fwd'] in oligo
        ]
        rvs = [
            determine_complement(pool_df.loc[i, 'rev'])
            for i in all_fw_idxs
            if determine_complement(pool_df.loc[i, 'rev']) in oligo
        ]

        # Only one match for each direction is allowed
        if len(set(fws)) != 1 or len(set(rvs)) != 1:
            error_seqs.append(oligo)
            continue

        fw_idx = pool_df[pool_df['fwd'] == fws[0]].index[0]
        rv_idx = pool_df[pool_df['rev'] == determine_complement(rvs[0])].index[0]

        # Exception for known shared reverse primers
        if fw_idx != rv_idx and rv_idx not in [18, 116]:
            error_seqs.append(oligo)

    if error_seqs:
        print("ATTENTION: Sequences with mismatched forward/reverse primers:")
        for seq in error_seqs:
            print(return_filename_of_seq(seq, df_dict))
    else:
        print("All primer pairs are valid and unique.\n")

    # ---------------------------- Summary Output ----------------------------

    for name, idxs in primer_dict.items():
        print(f"{name} uses primer indices: {sorted(idxs)}")
    print("QC checks complete.\n")

    return all_dfs


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def barcode_coverage_needed_based_on_initial_data(
        counts: pd.Series,
        n: int = 1,
        coverage_threshold: float = 1.0,
        confidence: float = 0.999,
        max_coverage_multiple: float = 20.0,
        step: float = 0.5,
        num_simulations: int = 1000,
        random_seed: int = None
) -> dict:
    """
    Estimate how many picks (scaled by library size) are needed to cover
    a target percentage of the library with at least `n` hits at a given confidence.

    Parameters
    ----------
    counts : pd.Series
        Observed counts of each variant in the original sample.
    n : int, optional
        Minimum number of times each variant must be seen. Default is 1.
    coverage_threshold : float, optional
        Fraction of variants to cover at least `n` times (e.g., 0.95). Default is 1.0.
    confidence : float, optional
        Required probability of achieving this coverage (e.g., 0.999). Default is 0.999.
    max_coverage_multiple : float, optional
        Maximum multiple of library size to try (e.g., 20 means 20× library size).
        Default is 20.0.
    step : float, optional
        Granularity of multiples to test (e.g., 0.5 for half-step). Default is 0.5.
    num_simulations : int, optional
        Number of simulation runs per multiple. Default is 1000.
    random_seed : int, optional
        Seed for reproducibility. Default is None.

    Returns
    -------
    dict or None
        Dictionary containing:
        {
            "required_multiple": float,
            "required_draws": int,
            "library_size": int,
            "success_rate": float,
        }
        Returns None if no multiple satisfies the requirement within limits.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    weights = counts.values
    total = weights.sum()
    probs = weights / total
    lib_size = len(probs)

    multiples = np.arange(step, max_coverage_multiple + step, step)

    for mult in multiples:
        draw_count = int(mult * lib_size)
        success_counts = []

        for _ in range(num_simulations):
            sampled = np.random.choice(lib_size, size=draw_count, p=probs)
            freqs = np.bincount(sampled, minlength=lib_size)
            coverage = np.sum(freqs >= n) / lib_size
            success_counts.append(coverage)

        passed = np.mean([c >= coverage_threshold for c in success_counts])

        if passed >= confidence:
            return {
                "required_multiple": mult,
                "required_draws": draw_count,
                "library_size": lib_size,
                "success_rate": passed
            }

    return None  # No acceptable multiple found within limits


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_motif_logomaker(
        list_of_sequences: List[str],
        title: str = 'Sequence Motif',
        color_scheme: str = 'classic',
        custom_colors: dict = None,
        figsize: tuple = (10, 3),
        ylabel: str = 'Probability',
        xlabel: str = 'Position',
        information_content: bool = False,
        save_path: str = None,
        dpi: int = 300,
        show_plot: bool = True
):
    """
    Generate a sequence logo from a list of DNA sequences.

    Parameters
    ----------
    list_of_sequences : List[str]
        List of DNA sequences (all must be same length).
    title : str, optional
        Title for the plot. Default is 'Sequence Motif'.
    color_scheme : str, optional
        Color scheme: 'classic', 'greys', 'charge', 'chemistry', 'hydrophobicity'
        or None to use custom_colors. Default is 'classic'.
    custom_colors : dict, optional
        Custom colors for nucleotides, e.g.
        {'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'orange'}.
    figsize : tuple, optional
        Figure size (width, height). Default is (10, 3).
    ylabel : str, optional
        Y-axis label. Default is 'Probability'.
    xlabel : str, optional
        X-axis label. Default is 'Position'.
    information_content : bool, optional
        If True, use information content (bits) instead of probability.
        Default is False.
    save_path : str, optional
        Path to save the figure (e.g., 'motif.png', 'motif.pdf').
    dpi : int, optional
        Resolution for saved figure. Default is 300.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    tuple
        (motif, fig) - The Bio.motifs.Motif object and matplotlib Figure.
    """
    from Bio import motifs
    from Bio.Seq import Seq
    import logomaker
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert to Seq objects and create motif
    bio_sequences = [Seq(x) for x in list_of_sequences]
    m = motifs.create(bio_sequences)

    # Get position frequency matrix
    pfm = m.counts.normalize(pseudocounts=0.5)

    # Convert to dataframe for logomaker
    bases = ['A', 'C', 'G', 'T']
    data = {base: list(pfm[base]) for base in bases}
    df = pd.DataFrame(data)

    # Convert to information content if requested
    if information_content:
        import numpy as np
        ic_df = df.copy()

        for i in range(len(df)):
            row = df.iloc[i]
            # Calculate entropy
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in row)
            # Information content = max_entropy - entropy
            max_entropy = 2  # for DNA (log2(4))
            ic = max_entropy - entropy
            # Scale probabilities by information content
            ic_df.iloc[i] = row * ic

        df = ic_df
        if ylabel == 'Probability':
            ylabel = 'Information Content (bits)'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create logo with custom colors if provided
    if custom_colors:
        logo = logomaker.Logo(df, color_scheme=None, ax=ax)
        # Apply custom colors
        for base, color in custom_colors.items():
            if base in bases:
                logo.style_glyphs_in_sequence(sequence=base, color=color)
    else:
        logo = logomaker.Logo(df, color_scheme=color_scheme, ax=ax)

    # Set labels and title
    logo.ax.set_ylabel(ylabel, fontsize=12)
    logo.ax.set_xlabel(xlabel, fontsize=12)
    logo.ax.set_title(title, fontsize=14, fontweight='bold')

    # Add grid for easier reading
    logo.ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    # Show plot
    if show_plot:
        plt.show()

    return m, fig