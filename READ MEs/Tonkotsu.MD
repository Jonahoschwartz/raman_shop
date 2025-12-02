![alt text](../pngs/Tonkatsu.png "Title")

# Tonkatsu

## Bioinformatics Utilities for DNA/Protein Analysis

**Tools for Sequence Processing, Alignment, and Visualization**

This package provides utilities for DNA barcode extraction, sequence alignment generation, position-specific scoring matrices (PSSMs), and data visualization. 

---

## DNA Sequence Analysis

### `iupac_to_regex(motif)`
Convert IUPAC-degenerate DNA motifs into regex patterns for flexible sequence matching.

**Parameters:**
- `motif`: DNA sequence containing IUPAC ambiguity codes (e.g., "ATGR", "NNNATGC")

**Returns:**
String containing the equivalent regex pattern with character classes.

**Example:**
```python
# Convert degenerate motif to regex
pattern = iupac_to_regex("ATGR")  # Returns 'ATG[AG]'
pattern = iupac_to_regex("NNATGC")  # Returns '[ACGT][ACGT]ATGC'

# Use in pattern matching
import re
sequence = "ATGATGCCCGATAG"
matches = re.findall(iupac_to_regex("ATGR"), sequence)
```

**Supported IUPAC Codes:**
- Standard bases: A, C, G, T
- Two-base: R (A/G), Y (C/T), S (G/C), W (A/T), K (G/T), M (A/C)
- Three-base: B (not A), D (not C), H (not G), V (not T)
- Any base: N (A/C/G/T)

---

### `extract_between_flanks(sequence, flanks)`
Extract DNA barcodes or sequences between two flanking regions that may contain IUPAC degeneracies.

**Parameters:**
- `sequence`: Input DNA sequence (IUPAC-compliant)
- `flanks`: Tuple of (5' flank, 3' flank) sequences with optional IUPAC codes

**Returns:**
- `str`: Extracted sequence between flanks
- `"None"`: Either flank missing or in wrong order
- `"Multiple"`: One or both flanks appear multiple times
- `"INVALID_SEQ"`: Sequence contains invalid characters

**Example:**
```python
# Basic barcode extraction
barcode = extract_between_flanks("ATGCCCGTAGTTT", ("ATG", "TAG"))
print(barcode)  # 'CCCG'

# With degenerate flanks
seq = "ATGACCCTGAGTTT"
barcode = extract_between_flanks(seq, ("ATGR", "KTGA"))  # Handles R=[AG], K=[GT]

# Quality control
if barcode == "Multiple":
    print("Warning: Flanking sequences appear multiple times")
elif barcode == "None":
    print("Error: One or both flanks not found")
elif barcode == "INVALID_SEQ":
    print("Error: Sequence contains invalid characters")
```

**Use Cases:**
- Amplicon barcode extraction from Illumina/PacBio reads
- Parsing construct libraries with known primer sequences
- Quality filtering for sequences with expected structure

---

## Data Visualization

### `filter_graph(series, num_cutoffs=100, ...)`
Generate a threshold optimization plot showing percentage of values retained at various cutoff levels.

**Parameters:**
- `series`: Numeric pandas Series to analyze
- `num_cutoffs`: Number of threshold values to test (default: 100)
- `max_cutoff`: Maximum cutoff value (default: series maximum)
- `color`: Line color (default: "blue")
- `marker`: Point marker style (default: "o")
- `title`: Plot title
- `xlabel`: X-axis label
- `ylabel`: Y-axis label
- `grid`: Show background grid (default: True)
- `save_path`: Optional path to save figure

**Example:**
```python
import pandas as pd

# Quality score filtering
quality_scores = pd.Series([10, 15, 20, 25, 30, 35, 40, 45, 50])
filter_graph(quality_scores, 
             title="Phred Quality Score Distribution",
             xlabel="Quality Score Cutoff",
             ylabel="% Reads Retained")

# Expression level thresholding
expression = pd.Series(data['TPM'])
filter_graph(expression, 
             num_cutoffs=200,
             max_cutoff=100,
             color='darkred',
             title="TPM Expression Threshold Analysis",
             save_path='tpm_filter.png')
```

**Use Cases:**
- Determining quality score cutoffs for read filtering
- Optimizing expression level thresholds
- Visualizing distribution of continuous metrics

---

### `plot_barcode_hist(series, bins="auto", ...)`
Create histogram showing frequency distribution of categorical values (e.g., barcode abundance).

**Parameters:**
- `series`: Pandas Series containing string values
- `bins`: Histogram binning strategy (default: "auto")
- `title`: Plot title
- `xlabel`: X-axis label (default: "Number of Occurrences")
- `ylabel`: Y-axis label (default: "Number of Unique Strings")
- `color`: Bar color (default: "cornflowerblue")
- `figsize`: Figure dimensions (default: (10, 6))
- `save_path`: Optional path to save figure
- `show_values`: Annotate bars with counts (default: False)

**Example:**
```python
# Barcode abundance distribution
barcodes = pd.Series(['ATGC', 'ATGC', 'GCTA', 'GCTA', 'GCTA', 'TTTT', 'AAAA', 'AAAA'])
plot_barcode_hist(barcodes, 
                  title="Barcode Frequency Distribution",
                  xlabel="Reads per Barcode",
                  show_values=True)

# Identify PCR duplicates
plot_barcode_hist(df['UMI'], 
                  title="UMI Duplication Analysis",
                  color='salmon',
                  bins=range(1, 50),
                  save_path='umi_duplicates.svg')

# Quality control
plot_barcode_hist(sequences, 
                  title="Sequence Diversity Check",
                  figsize=(14, 8))
```

**Features:**
- Automatic binning based on data range
- Colorblind-friendly default palette
- Optional value annotations for exact counts
- Useful for identifying sequencing errors and PCR artifacts

---

## Position-Specific Scoring Matrices (PSSMs)

### `pssm_from_DNA_msa(msa_file, output_json=True)`
Generate DNA PSSM from a multiple sequence alignment for motif scoring and sequence analysis.

**Parameters:**
- `msa_file`: Path to FASTA-format multiple sequence alignment
- `output_json`: Save PSSM as JSON file (default: True)

**Returns:**
DataFrame with nucleotides (A, C, G, T) as rows and positions as columns, containing log2 scores.

**Example:**
```python
# Generate PSSM from alignment
pssm = pssm_from_DNA_msa("promoter_alignment.fasta")

# Score a new sequence
def score_sequence(seq, pssm):
    score = sum(pssm.loc[base, i+1] for i, base in enumerate(seq))
    return score

candidate = "ATGCGATCG"
binding_score = score_sequence(candidate, pssm)
print(f"Binding score: {binding_score:.2f}")

# Identify conserved positions
conservation = pssm.max(axis=0)
highly_conserved = conservation[conservation > 2.0]
print(f"Highly conserved positions: {list(highly_conserved.index)}")
```

**Features:**
- Uses pseudocount of 0.25 to avoid log(0) issues
- Log2-scaled scores for intuitive interpretation
- Automatic JSON export for reuse
- Assumes uniform background frequencies (0.25 per base)

---

### `pssm_from_protein_msa(msa_file, output_json=True)`
Generate protein PSSM from alignment for domain identification and motif analysis.

**Parameters:**
- `msa_file`: Path to FASTA-format protein alignment
- `output_json`: Save PSSM as JSON file (default: True)

**Returns:**
DataFrame with 20 amino acids as rows and positions as columns, containing log2 scores.

**Example:**
```python
# Generate protein PSSM
pssm = pssm_from_protein_msa("kinase_domain.fasta")

# Score protein sequence
protein_seq = "MLKQVEIFTDGSCLGNPHLLSR"
score = sum(pssm.loc[aa, i+1] for i, aa in enumerate(protein_seq))

# Find catalytic residues
catalytic_pos = pssm.idxmax(axis=0)  # Most preferred AA at each position
print(catalytic_pos)

# Identify variable positions
entropy = -(pssm.applymap(lambda x: 2**x * x if x < 0 else 0).sum(axis=0))
variable_sites = entropy[entropy > 1.5]
```

**Features:**
- Uses pseudocount of 0.05 (smaller due to 20-letter alphabet)
- Handles all 20 standard amino acids
- Log2-scaled scoring
- Useful for functional site prediction

---

## Multiple Sequence Alignment

### `a3m_to_fasta(a3m_str)`
Convert ColabFold a3m alignment format to standard FASTA format by removing insertion annotations.

**Parameters:**
- `a3m_str`: Multiple sequence alignment in a3m format

**Returns:**
String containing standard FASTA alignment with insertions removed.

**Example:**
```python
# Read a3m file
with open("alignment.a3m", "r") as f:
    a3m_data = f.read()

# Convert to standard FASTA
fasta_data = a3m_to_fasta(a3m_data)

# Save result
with open("alignment.fasta", "w") as f:
    f.write(fasta_data)
```

**Notes:**
- Lowercase letters represent insertions relative to consensus
- These are removed to create columnar alignment
- Header lines (starting with '>') preserved unchanged

---

### `get_msa_from_sequence(sequence, prefix="msa_output", save_dir=".")`
Generate multiple sequence alignment from a single protein sequence using ColabFold MMseqs2 server.

**Parameters:**
- `sequence`: Input protein sequence (single-letter amino acid codes)
- `prefix`: Filename prefix for outputs (default: "msa_output")
- `save_dir`: Output directory (default: current directory)

**Returns:**
Tuple of (a3m_file_path, fasta_file_path).

**Example:**
```python
# Generate MSA for protein sequence
protein = "MKLLVLSLILSLVLVYIIGFGALIGFGMKQSTKRHHHHHHH"
a3m_path, fasta_path = get_msa_from_sequence(protein, 
                                             prefix="myprotein",
                                             save_dir="alignments")

# Use for downstream analysis
pssm = pssm_from_protein_msa(fasta_path)

# Generate MSA for structure prediction input
seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
get_msa_from_sequence(seq, prefix="ras_protein")
```

**Features:**
- Searches UniRef and environmental databases
- Automatic job submission and monitoring
- Progress bar tracking
- Saves both a3m and FASTA formats
- Requires internet connection

---

### `run_mmseqs2(x, prefix, use_env=True, use_filter=True, ...)`
Low-level interface to ColabFold MMseqs2 server for custom MSA generation workflows.

**Parameters:**
- `x`: Protein sequence(s) as string or list
- `prefix`: Directory prefix for temporary files
- `use_env`: Include environmental sequences (default: True)
- `use_filter`: Apply sequence filtering (default: True)
- `use_templates`: Use template sequences (default: False)
- `host_url`: Server URL (default: "https://a3m.mmseqs.com")

**Returns:**
MSA in a3m format (string for single input, list for multiple).

**Example:**
```python
# Basic MSA generation
msa = run_mmseqs2("MKLLVLSLILSLVLVYII", prefix="test")

# Multiple sequences at once
sequences = ["MKLLVLSLIL", "MTEYKLVVVG", "MALWMRLLPL"]
msas = run_mmseqs2(sequences, prefix="batch", use_env=True)

# Custom search parameters
msa = run_mmseqs2("MKLLVLSLIL", 
                  prefix="filtered",
                  use_env=False,  # UniRef only
                  use_filter=False)  # No filtering
```

**Features:**
- Batch processing for multiple sequences
- Automatic rate limiting and retry logic
- Caches results to avoid redundant calls
- Handles server maintenance and errors gracefully

---

## FASTA/FASTQ File Operations

### `read_fasta(filepath)`
Parse FASTA file into a dictionary of sequences.

**Parameters:**
- `filepath`: Path to FASTA file

**Returns:**
Dictionary with headers (without ">") as keys and uppercase sequences as values.

**Example:**
```python
# Load sequences
sequences = read_fasta("reference.fasta")

# Access specific sequence
ref_seq = sequences["chr1"]

# Iterate through all sequences
for header, seq in sequences.items():
    print(f"{header}: {len(seq)} bp")

# Filter by length
long_seqs = {h: s for h, s in sequences.items() if len(s) > 1000}
```

---

### `write_fasta(sequences, output_file, prefix="seq", wrap=80)`
Write sequences to a FASTA file with customizable formatting.

**Parameters:**
- `sequences`: List of sequences or dict of {header: sequence}
- `output_file`: Output file path
- `prefix`: Header prefix for list input (default: "seq")
- `wrap`: Characters per line (default: 80)

**Example:**
```python
# From list of sequences
seqs = ["ATGCGATCG", "GCTAGCTA", "TTTAGGGCC"]
write_fasta(seqs, "output.fasta", prefix="barcode")

# From dictionary
seq_dict = {
    "construct1": "ATGCGATCGATCG",
    "construct2": "GCTAGCTAGCTA"
}
write_fasta(seq_dict, "constructs.fasta")

# Custom line wrapping
write_fasta(seqs, "nowrap.fasta", wrap=10000)  # Single line per sequence
```

---

### `load_fastq_sequences(filepath)`
Load sequences from FASTQ file into a list (ignores quality scores).

**Parameters:**
- `filepath`: Path to FASTQ file

**Returns:**
List of nucleotide sequences as strings.

**Example:**
```python
# Load all sequences
reads = load_fastq_sequences("sample.fastq")
print(f"Total reads: {len(reads)}")

# Quick QC
avg_length = sum(len(r) for r in reads) / len(reads)
print(f"Average read length: {avg_length:.1f} bp")

# Filter by length
long_reads = [r for r in reads if len(r) >= 1000]
```

---

### `split_fastq_by_indices(fastq_file, index_csv, n=100, min_len=4000, output_dir="output_fastq")`
Demultiplex FASTQ file by index pairs with support for reverse complements and bidirectional matching.

**Parameters:**
- `fastq_file`: Path to input FASTQ file
- `index_csv`: CSV with columns: well, row, column (containing index sequences)
- `n`: Number of bases from each read end to check (default: 100)
- `min_len`: Minimum read length to consider (default: 4000)
- `output_dir`: Directory for demultiplexed FASTQ files (default: "output_fastq")

**Returns:**
DataFrame with read counts per well plus summary statistics.

**Example:**
```python
# Basic demultiplexing
results = split_fastq_by_indices("run1.fastq", 
                                 "barcode_layout.csv")
print(results)

# Custom parameters for PacBio data
results = split_fastq_by_indices("pacbio_reads.fastq",
                                 "96well_indices.csv",
                                 n=150,  # Check 150bp from ends
                                 min_len=5000,  # Only long reads
                                 output_dir="demux_pacbio")

# Review demultiplexing stats
print(f"Total input reads: {results.loc[results['well']=='__TOTAL_INPUT__', 'assigned_reads'].values[0]}")
print(f"Assigned reads: {results.loc[results['well']=='__TOTAL_ASSIGNED__', 'assigned_reads'].values[0]}")
print(f"Unassigned reads: {results.loc[results['well']=='__UNASSIGNED__', 'assigned_reads'].values[0]}")
```

**Features:**
- Matches indices on both forward and reverse complement
- Handles indices on either end of read
- Generates separate FASTQ file for each well
- Provides comprehensive demultiplexing statistics
- Quality control with minimum length filtering




