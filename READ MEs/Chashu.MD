![alt text](../pngs/Chashu.png "Title")

## DNA Sequence Manipulation and Analysis Utilities

**Tools for Molecular Biology Workflows**

This package provides utilities for DNA sequence manipulation, primer design, quality control, and analysis. 

---

## Sequence Generation

### `randnt(n, gc_content=0.5)`
Generates a random nucleotide sequence with specified length and GC content.

**Parameters:**
- `n`: Length of sequence in base pairs
- `gc_content`: Desired proportion of G's and C's (0.0–1.0)

**Example:**
```python
random_seq = randnt(100, gc_content=0.6)  # 100bp sequence with 60% GC
```

---

### `simple_randnt(n)`
Generate a fully random nucleotide sequence of length n with equal base probabilities.

**Example:**
```python
seq = simple_randnt(50)  # 50bp random sequence
```

---

## Sequence Manipulation

### `reverse_complement(sequence)`
Calculate the reverse complement of a DNA sequence. Handles both upper and lowercase input.

**Example:**
```python
rc = reverse_complement("ATCG")  # Returns "CGAT"
```

---

### `translate(dna)`
Translate DNA to protein using the standard codon table. Stop codons are represented as `*` and unknown codons as `X`.

**Example:**
```python
protein = translate("ATGGCCTAT")  # Returns "MAY"
```

---

### `mutation_string_from_dna(dna1, dna2)`
Compare two DNA sequences and return a list of amino acid mutations in standard notation (e.g., `"A42T"`).

**Example:**
```python
mutations = mutation_string_from_dna(wt_seq, mutant_seq)
# Returns ["A15V", "K42R"] or ["WT"] if identical
```

---

## Degenerate Sequence Handling

### `degenerate_to_regex(seq)`
Convert a degenerate DNA sequence (with IUPAC codes) into a regex pattern for searching.

**Example:**
```python
pattern = degenerate_to_regex("ATNNG")  # Matches AT followed by any 2 bases, then G
```

---

### `generate_sequences(degenerate_sequences)`
Expand degenerate sequences into all possible non-degenerate variants.

**Example:**
```python
variants = generate_sequences(["NNK", "NNS"])
# Returns dictionary mapping each to all possible codons
```

---

## Restriction Site Analysis

### `find_restriction_site_counts(sequence, enzymes=None, recognition_seqs=None)`
Count restriction sites in both strands of a DNA sequence. Supports both Biopython enzyme objects and custom recognition sequences with degenerate bases.

**Parameters:**
- `sequence`: DNA sequence (5' to 3')
- `enzymes`: List of Biopython Restriction enzyme classes (e.g., `[EcoRI, BamHI]`)
- `recognition_seqs`: List of recognition sequences (e.g., `["GAATTC", "GGATCC"]`)

**Example:**
```python
from Bio.Restriction import EcoRI, BamHI

# Using enzyme objects
counts = find_restriction_site_counts(my_seq, enzymes=[EcoRI, BamHI])

# Using recognition sequences with degenerate bases
counts = find_restriction_site_counts(my_seq, recognition_seqs=["GGTCTC", "AATNNTS"])
# Returns: {'EcoRI': 2, 'BamHI': 0} or {'GGTCTC': 1, 'AATNNTS': 3}
```

---

## Primer Design for Mutagenesis

### `generate_suni_mutagensis_primers(WT_seq, left_flank, right_flank, ...)`
Design a smart nicking mutagenesis library with optimized homology arms and degenerate codons (NNK/NNS).

**Parameters:**
- `WT_seq`: Wild-type coding sequence
- `left_flank`: Upstream flanking sequence
- `right_flank`: Downstream flanking sequence
- `tm_left_min`, `tm_left_max`: Acceptable Tm range for left arm (default: 59–66°C)
- `target_tm_right`: Target Tm for right arm (default: 61°C)
- `min_len_left`, `max_len_left`: Left arm length range (default: 18–40bp)
- `min_len_right`, `max_len_right`: Right arm length range (default: 18–40bp)

**Returns:**
Dictionary of oligos with codon positions as keys. Automatically:
- Prioritizes strong 5' clamps (SSS > SSW > SWS)
- Balances Tm across primers
- Chooses NNK or NNS based on WT codon
- Generates plots showing Tm and length distributions
- Flags primers needing double synthesis

**Example:**
```python
primers = generate_suni_mutagensis_primers(
    WT_seq=my_gene,
    left_flank=upstream_50bp,
    right_flank=downstream_50bp,
    tm_left_min=60,
    tm_left_max=65,
    target_tm_right=62
)
# Returns: {2: "ACGTACGT...NNK...TGCATGCA", 3: "GCTAGCTA...NNS...ATCGATCG", ...}
```

---

### `gen_smart_nicking_lib(WT_seq, left_flank, right_flank, ...)`
Alternative implementation of smart nicking library design (from [Lehner Lab SUNi mutagenesis](https://github.com/lehner-lab/SUNi_mutagenesis)).

**Parameters:** Same as `generate_suni_mutagensis_primers`

**Example:**
```python
lib = gen_smart_nicking_lib(
    WT_seq=gene,
    left_flank=flank_5p,
    right_flank=flank_3p,
    tm_left_min=59,
    tm_left_max=66,
    target_tm_right=61,
    min_len_left=18,
    max_len_left=40,
    min_len_right=18,
    max_len_right=40
)
```

---

## File I/O

### `dict_to_fastq(seq_dict, output_path)`
Save a dictionary of sequences as a FASTQ file with high-quality scores.

**Example:**
```python
sequences = {
    "oligo_1": "ATCGATCGATCG",
    "oligo_2": "GCTAGCTAGCTA"
}
dict_to_fastq(sequences, "output.fastq")
```

---

## Quality Control

### `run_oligo_qc_nnovy(pool_primers_path, data_base_path, available_restriction_sites)`
Comprehensive quality control for oligonucleotide synthesis orders. Checks for duplicates, length consistency, restriction sites, and primer compatibility.

**Parameters:**
- `pool_primers_path`: Path to CSV/Excel with columns `'fwd'` and `'rev'`
- `data_base_path`: Directory containing oligo design files (CSV, XLSX, etc.)
- `available_restriction_sites`: List of expected restriction sites (e.g., `['GGTCTC', 'CGTCTC']`)

**QC Checks Performed:**
1. **Duplicate detection** across all design files
2. **Length validation** (flags sequences <90% of max length)
3. **Restriction site verification** (ensures exactly 2 sites per oligo)
4. **Primer overlap check** (prevents same primer in multiple pools)
5. **Forward/reverse primer pairing** (validates consistency)

**Example:**
```python
validated_oligos = run_oligo_qc_nnovy(
    pool_primers_path="primers.xlsx",
    data_base_path="oligo_designs/",
    available_restriction_sites=['GGTCTC', 'CGTCTC']
)

# Prints detailed QC report and returns combined DataFrame
```

**Output:**
```
No duplicates found across design files.
Max oligo length: 150 nt
All sequences are between 135 and 150 nt.
All sequences have exactly two cut sites.
Detected cut site frequencies:
GGTCTC: 45 sequences
CGTCTC: 23 sequences
...
QC checks complete.
```

---

## Statistical Analysis

### `barcode_coverage_needed_based_on_initial_data(counts, n=1, coverage_threshold=1.0, ...)`
Estimate sequencing depth needed to achieve target coverage of a barcode library.

**Parameters:**
- `counts`: Pandas Series of observed variant counts
- `n`: Minimum reads required per variant (default: 1)
- `coverage_threshold`: Fraction of library to cover (default: 1.0 = 100%)
- `confidence`: Required confidence level (default: 0.999)
- `max_coverage_multiple`: Maximum library-size multiple to test (default: 20×)
- `step`: Granularity of testing (default: 0.5)
- `num_simulations`: Monte Carlo iterations (default: 1000)
- `random_seed`: For reproducibility

**Returns:**
Dictionary with required sampling depth or `None` if target is unachievable.

**Example:**
```python
import pandas as pd

initial_counts = pd.Series([100, 50, 25, 10, 5, 2, 1, 1])  # Variant frequencies

result = barcode_coverage_needed_based_on_initial_data(
    counts=initial_counts,
    n=3,  # Need 3+ reads per variant
    coverage_threshold=0.95,  # Cover 95% of library
    confidence=0.99
)

print(result)
# {'required_multiple': 15.0, 'required_draws': 120, 'library_size': 8, 'success_rate': 0.991}
# Interpretation: Need to sample 15× library size (120 total picks) to achieve 95% coverage with 3+ reads
```

---

## Visualization

### `generate_motif_logomaker(list_of_sequences, title='Sequence Motif', ...)`
Create a sequence logo from aligned DNA sequences using `logomaker` and `Bio.motifs`.

**Parameters:**
- `list_of_sequences`: List of aligned DNA sequences (must be same length)
- `title`: Plot title
- `color_scheme`: `'classic'`, `'greys'`, `'charge'`, `'chemistry'`, or `'hydrophobicity'`
- `custom_colors`: Dictionary for custom coloring (e.g., `{'A': 'red', 'T': 'blue'}`)
- `figsize`: Figure dimensions (default: `(10, 3)`)
- `information_content`: Use information content (bits) instead of probability
- `save_path`: Path to save figure (e.g., `'motif.png'`)
- `dpi`: Resolution for saved figure (default: 300)
- `show_plot`: Display the plot

**Returns:**
Tuple of `(motif_object, figure)`

**Example:**
```python
sequences = [
    "ATCGATCG",
    "ATCGATCG",
    "ATCGATGG",
    "ATCGCTCG"
]

motif, fig = generate_motif_logomaker(
    sequences,
    title="Binding Site Motif",
    color_scheme='classic',
    information_content=True,
    save_path="motif_logo.png"
)
```
