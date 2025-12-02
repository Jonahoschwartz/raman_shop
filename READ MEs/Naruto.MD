![alt text](../pngs/Naruto.png "Title")

## Protein Structure and Chemical Analysis Utilities

**Tools for Structural Biology and Cheminformatics Workflows**

This package provides utilities for protein structure analysis, chemical similarity calculations, ligand extraction, structure prediction, and database queries. 

---

## Chemical Similarity and Properties

### `tanimoto_calc(smi1, smi2)`
Calculate Tanimoto similarity between two molecules using Morgan fingerprints.

**Parameters:**
- `smi1`: SMILES string of first molecule
- `smi2`: SMILES string of second molecule

**Returns:**
Float representing Tanimoto similarity coefficient (0-1), rounded to 3 decimal places.

**Example:**
```python
similarity = tanimoto_calc("CCO", "CCC")  # Compare ethanol and propane
print(f"Similarity: {similarity}")  # Returns ~0.333
```

---

### `logp(smiles)`
Calculate the octanol-water partition coefficient (LogP) for a molecule.

**Parameters:**
- `smiles`: SMILES string of the molecule

**Returns:**
Float representing LogP value (lipophilicity measure).

**Example:**
```python
logp_value = logp("CCO")  # Ethanol
print(f"LogP: {logp_value}")  # Returns -0.031
```

---

## Protein Sequence Extraction

### `pdb_to_sequences(pdb_file)`
Extract protein sequences from a PDB or mmCIF file, grouped by model (molecule).

**Parameters:**
- `pdb_file`: Path to a .pdb or .cif file

**Returns:**
List where each entry is a molecule (model), represented as a dictionary of chain IDs mapped to sequence strings.

**Example:**
```python
sequences = pdb_to_sequences("protein.pdb")
print(sequences[0])  # {'A': 'MKKLVLSLSLVLAFSSATAAF...', 'B': 'ATGC...'}

# Access specific chain
chain_a_seq = sequences[0]['A']
```

---

## Structural Analysis and Distance Calculations

### `residue_distance_matrix_df(pdb_file)`
Compute residue Cα distance matrix and return as a labeled pandas DataFrame.

**Parameters:**
- `pdb_file`: Path to PDB or CIF file

**Returns:**
Square DataFrame of distances indexed and column-labeled by residues. Format: `{chain_id}_{residue_name}{residue_number}`

**Example:**
```python
dist_matrix = residue_distance_matrix_df("protein.pdb")

# Get distance between two residues
distance = dist_matrix.loc["A_MET1", "A_ALA5"]
print(f"Distance: {distance:.2f} Å")

# Find all residues within 8 Å of a target
close_residues = dist_matrix.loc["A_TRP42", dist_matrix.loc["A_TRP42"] < 8.0]
```

---

### `extract_bfactors_to_df(pdb_file)`
Extract B-factors (temperature factors) for all atoms in a PDB or mmCIF structure.

**Parameters:**
- `pdb_file`: Path to PDB or CIF file

**Returns:**
DataFrame with columns: `['chain', 'residue_name', 'residue_number', 'atom_name', 'b_factor']`

**Example:**
```python
bfactors = extract_bfactors_to_df("protein.pdb")

# Find highly flexible regions
high_b = bfactors[bfactors['b_factor'] > 50]
print(high_b)

# Calculate average B-factor per residue
avg_b = bfactors.groupby(['chain', 'residue_number'])['b_factor'].mean()
```

---

## Structural Validation and Visualization

### `ramachandran_plot_colored(pdb_file, figsize=(8, 8), point_size=20, alpha=0.7, ...)`
Generate a Ramachandran plot colored by residue type with extensive customization.

The Ramachandran plot shows the distribution of phi-psi angles for protein backbone, which is crucial for validating protein structure quality.

**Parameters:**
- `pdb_file`: Path to PDB or mmCIF file
- `figsize`: Figure size in inches (width, height) (default: `(8, 8)`)
- `point_size`: Size of scatter plot points (default: 20)
- `alpha`: Transparency of points, 0-1 (default: 0.7)
- `show_grid`: Whether to display grid lines (default: `True`)
- `add_legend`: Whether to display legend for residue colors (default: `True`)
- `title`: Plot title (default: `"Ramachandran Plot with Residue Coloring"`)
- `xlabel`: X-axis label (default: `"Phi (φ) angle (degrees)"`)
- `ylabel`: Y-axis label (default: `"Psi (ψ) angle (degrees)"`)
- `edge_color`: Color of point edges, `'k'` for black or `None` (default: `'k'`)
- `black_points`: If `True`, plot all points as black (default: `False`)

**Example:**
```python
# Basic Ramachandran plot
ramachandran_plot_colored("protein.pdb")

# Customized plot
ramachandran_plot_colored(
    "protein.pdb",
    figsize=(10, 10),
    point_size=30,
    alpha=0.5,
    black_points=True,
    title="My Protein Structure Validation"
)
```

**Note:**
- Phi angles: N-Cα bond rotation
- Psi angles: Cα-C bond rotation
- Different regions indicate different secondary structures
- Most residues should fall in allowed regions

---

## Ligand Extraction and Conversion

### `ligand_to_smiles(pdb_file, ligand_resname)`
Convert a ligand from PDB/mmCIF to SMILES string with proper bond order detection.

**Parameters:**
- `pdb_file`: Path to PDB or mmCIF file
- `ligand_resname`: Residue name of the ligand (e.g., `'ATP'`, `'HEM'`, `'NAD'`)

**Returns:**
SMILES string if successful, `None` if conversion fails.

**Example:**
```python
# Extract ATP from protein-ligand complex
smiles = ligand_to_smiles("complex.pdb", "ATP")
if smiles:
    print(f"ATP SMILES: {smiles}")
    
    # Use the SMILES for further analysis
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    print(f"Molecular weight: {Chem.Descriptors.MolWt(mol):.2f}")
```

**Note:**
- Automatically tries multiple bond perception methods
- Method 1: RDKit bond perception algorithm (preferred)
- Method 2: Distance-based connectivity as fallback (< 1.8 Å threshold)

---

## Structure Prediction

### `esmfold_predict(sequence, output_path="predicted_structure.pdb")`
Predict a 3D protein structure using the ESMFold API and save it as a PDB file.

ESMFold is a state-of-the-art protein folding prediction model that generates 3D structures from amino acid sequences using language model embeddings.

**Parameters:**
- `sequence`: Protein sequence using single-letter amino acid codes
- `output_path`: Path to save the resulting PDB file (default: `"predicted_structure.pdb"`)

**Returns:**
Path to the saved PDB file if successful, `None` if the request failed.

**Example:**
```python
# Predict structure from sequence
sequence = "MKKLVLSLSLVLAFSSATAAFAAIPQNIRIGTDPTYAPFESKNS"
pdb_file = esmfold_predict(sequence, "my_protein.pdb")

if pdb_file:
    print(f"Structure saved to {pdb_file}")
    
    # Analyze the predicted structure
    dist_matrix = residue_distance_matrix_df(pdb_file)
    ramachandran_plot_colored(pdb_file)
```

**Note:**
- Requires internet connection to access ESM Atlas API
- Sequence length limitations may apply
- Prediction quality varies with sequence characteristics
- Works best for single-domain proteins < 400 residues

---

## Database Queries

### `fetch_uniprot_entry(query, query_type="accession", fields=None, format="dict")`
Fetch an entry from UniProt using the REST API.

**Parameters:**
- `query`: The query string (e.g., UniProt accession or protein name)
- `query_type`: Type of query - `"accession"` or `"name"` (default: `"accession"`)
- `fields`: List of fields to retrieve (e.g., `["accession", "protein_name", "sequence"]`). If `None`, retrieves all fields
- `format`: Output format - `"dict"` or `"dataframe"` (default: `"dict"`)

**Returns:**
Parsed UniProt data as dictionary or DataFrame.

**Example:**
```python
# Fetch by accession
entry = fetch_uniprot_entry("P12345", query_type="accession")
print(entry['primaryAccession'])
print(entry['proteinDescription'])

# Fetch specific fields
entry = fetch_uniprot_entry(
    "P12345",
    fields=["accession", "protein_name", "organism_name", "sequence"],
    format="dataframe"
)
print(entry)

# Fetch by protein name
entry = fetch_uniprot_entry("insulin", query_type="name")
```

---

### `fetch_pdb_structure(pdb_id, download_dir='pdbs')`
Download a PDB structure file from the Protein Data Bank.

**Parameters:**
- `pdb_id`: PDB identifier (e.g., `'1ABC'`)
- `download_dir`: Directory to save the downloaded file (default: `'pdbs'`)

**Returns:**
Path to the downloaded PDB file.

**Example:**
```python
# Download structure
pdb_file = fetch_pdb_structure("1ABC", download_dir="structures")
print(f"Downloaded to {pdb_file}")

# Analyze the downloaded structure
sequences = pdb_to_sequences(pdb_file)
bfactors = extract_bfactors_to_df(pdb_file)
```


