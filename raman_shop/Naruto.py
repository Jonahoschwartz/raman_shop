"""
Protein Structure and Chemical Analysis Utilities

This module provides functions for:
- Chemical similarity and property calculations
- Protein sequence extraction from structures
- Structural analysis and distance calculations
- Ramachandran plot generation
- Ligand extraction and conversion
- Structure prediction via ESMFold
- Database queries (UniProt, PDB)
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Chemical analysis
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDetermineBonds
from rdkit import DataStructs

# Protein structure analysis
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder, PDBList
from Bio.PDB.Polypeptide import is_aa, three_to_one

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# System and web requests
import os
import requests


# ============================================================================
# CHEMICAL SIMILARITY AND PROPERTIES
# ============================================================================

def tanimoto_calc(smi1: str, smi2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules using Morgan fingerprints.

    Parameters
    ----------
    smi1 : str
        SMILES string of first molecule.
    smi2 : str
        SMILES string of second molecule.

    Returns
    -------
    float
        Tanimoto similarity coefficient (0-1), rounded to 3 decimal places.

    Example
    -------
    >>> similarity = tanimoto_calc("CCO", "CCC")
    >>> print(similarity)  # Returns similarity score
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
    return s


def logp(smiles: str) -> float:
    """
    Calculate the octanol-water partition coefficient (LogP) for a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.

    Returns
    -------
    float
        LogP value (lipophilicity measure).

    Example
    -------
    >>> logp_value = logp("CCO")  # ethanol
    >>> print(logp_value)  # Returns LogP value
    """
    mol = Chem.MolFromSmiles(smiles)
    logp_val = Descriptors.MolLogP(mol)
    return logp_val


# ============================================================================
# PROTEIN SEQUENCE EXTRACTION
# ============================================================================

def pdb_to_sequences(pdb_file: str) -> list:
    """
    Extract protein sequences from a PDB or mmCIF file, grouped by model (molecule).

    Parameters
    ----------
    pdb_file : str
        Path to a .pdb or .cif file.

    Returns
    -------
    list
        Each list entry is a molecule (model), represented as a dict of chain IDs
        mapped to sequence strings.

    Raises
    ------
    ValueError
        If file extension is not .pdb or .cif.

    Example
    -------
    >>> sequences = pdb_to_sequences("protein.pdb")
    >>> print(sequences[0])  # {'A': 'MKKLVLSLSLVLAFSSATAAF...'}
    """
    ext = os.path.splitext(pdb_file)[1].lower()

    if ext == ".pdb":
        parser = PDBParser(QUIET=True)
    elif ext == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    structure = parser.get_structure("structure", pdb_file)
    ppb = PPBuilder()
    molecules = []

    for model in structure:
        molecule = {}
        for chain in model:
            peptides = ppb.build_peptides(chain)
            if peptides:
                sequence = ''.join(str(peptide.get_sequence()) for peptide in peptides)
                molecule[chain.id] = sequence
        molecules.append(molecule)

    return molecules


# ============================================================================
# STRUCTURAL ANALYSIS AND DISTANCE CALCULATIONS
# ============================================================================

def residue_distance_matrix_df(pdb_file: str) -> pd.DataFrame:
    """
    Compute residue Cα distance matrix and return as a labeled pandas DataFrame.

    Parameters
    ----------
    pdb_file : str
        Path to PDB or CIF file.

    Returns
    -------
    pd.DataFrame
        Square matrix of distances indexed and column-labeled by residues.
        Format: {chain_id}_{residue_name}{residue_number}

    Note
    ----
    Only considers amino acid residues with Cα atoms present.

    Example
    -------
    >>> dist_matrix = residue_distance_matrix_df("protein.pdb")
    >>> print(dist_matrix.loc["A_MET1", "A_ALA5"])  # Distance between residues
    """
    ext = os.path.splitext(pdb_file)[1].lower()
    parser = PDBParser(QUIET=True) if ext == '.pdb' else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    coords = []
    residue_labels = []

    for chain in model:
        for res in chain:
            if is_aa(res) and 'CA' in res:
                coords.append(res['CA'].get_coord())
                resname = res.get_resname()
                resnum = res.get_id()[1]
                label = f"{chain.id}_{resname}{resnum}"
                residue_labels.append(label)

    coords = np.array(coords)
    n = len(coords)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    df = pd.DataFrame(dist_matrix, index=residue_labels, columns=residue_labels)
    return df


def extract_bfactors_to_df(pdb_file: str) -> pd.DataFrame:
    """
    Extract B-factors (temperature factors) for all atoms in a PDB or mmCIF structure.

    Parameters
    ----------
    pdb_file : str
        Path to PDB or CIF file.

    Returns
    -------
    pd.DataFrame
        Columns = ['chain', 'residue_name', 'residue_number', 'atom_name', 'b_factor']

    Note
    ----
    B-factors indicate atomic displacement/flexibility in the crystal structure.

    Example
    -------
    >>> bfactors = extract_bfactors_to_df("protein.pdb")
    >>> high_b = bfactors[bfactors['b_factor'] > 50]  # Highly flexible regions
    """
    ext = os.path.splitext(pdb_file)[1].lower()
    parser = PDBParser(QUIET=True) if ext == '.pdb' else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    data = []
    for chain in model:
        for res in chain:
            resname = res.get_resname()
            resnum = res.get_id()[1]
            for atom in res:
                atom_name = atom.get_name()
                bfactor = atom.get_bfactor()
                data.append([chain.id, resname, resnum, atom_name, bfactor])

    df = pd.DataFrame(
        data,
        columns=['chain', 'residue_name', 'residue_number', 'atom_name', 'b_factor']
    )
    return df


# ============================================================================
# STRUCTURAL VALIDATION AND VISUALIZATION
# ============================================================================

def ramachandran_plot_colored(
    pdb_file: str,
    figsize: tuple = (8, 8),
    point_size: int = 20,
    alpha: float = 0.7,
    show_grid: bool = True,
    add_legend: bool = True,
    title: str = "Ramachandran Plot with Residue Coloring",
    xlabel: str = "Phi (φ) angle (degrees)",
    ylabel: str = "Psi (ψ) angle (degrees)",
    edge_color: str = 'k',
    black_points: bool = False
) -> None:
    """
    Generate a Ramachandran plot colored by residue type with extensive customization.

    The Ramachandran plot shows the distribution of phi-psi angles for protein backbone,
    which is crucial for validating protein structure quality.

    Parameters
    ----------
    pdb_file : str
        Path to PDB or mmCIF file.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (8, 8).
    point_size : int, optional
        Size of scatter plot points. Default is 20.
    alpha : float, optional
        Transparency of points (0 transparent, 1 opaque). Default is 0.7.
    show_grid : bool, optional
        Whether to display grid lines. Default is True.
    add_legend : bool, optional
        Whether to display a legend for residue colors. Default is True.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    edge_color : str, optional
        Color of point edges ('k' = black, None = no edge). Default is 'k'.
    black_points : bool, optional
        If True, plot all points as black ignoring residue colors. Default is False.

    Returns
    -------
    None
        Displays the plot.

    Note
    ----
    - Phi angles: N-Cα bond rotation
    - Psi angles: Cα-C bond rotation
    - Different regions indicate different secondary structures

    Example
    -------
    >>> ramachandran_plot_colored("protein.pdb", black_points=True)
    """
    ext = os.path.splitext(pdb_file)[1].lower()
    parser = PDBParser(QUIET=True) if ext == ".pdb" else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    ppb = PPBuilder()

    # Color scheme for 20 standard amino acids
    aa_colors = {
        'A': 'red', 'R': 'orange', 'N': 'yellow', 'D': 'green',
        'C': 'cyan', 'Q': 'blue', 'E': 'purple', 'G': 'magenta',
        'H': 'brown', 'I': 'pink', 'L': 'lightgreen', 'K': 'lightblue',
        'M': 'darkred', 'F': 'darkorange', 'P': 'darkgreen', 'S': 'darkcyan',
        'T': 'darkblue', 'W': 'darkmagenta', 'Y': 'gray', 'V': 'black'
    }

    phi_angles = []
    psi_angles = []
    colors = []

    # Extract phi-psi angles for each residue
    for chain in model:
        polypeptides = ppb.build_peptides(chain)
        for poly in polypeptides:
            phi_psi = poly.get_phi_psi_list()
            residues = poly.get_sequence()
            for (phi, psi), resname in zip(phi_psi, residues):
                if phi is not None and psi is not None:
                    phi_angles.append(np.degrees(phi))
                    psi_angles.append(np.degrees(psi))
                    if black_points:
                        colors.append('black')
                    else:
                        colors.append(aa_colors.get(resname, 'black'))

    # Create the plot
    plt.figure(figsize=figsize)
    plt.scatter(
        phi_angles, psi_angles,
        c=colors,
        s=point_size,
        alpha=alpha,
        edgecolors=edge_color,
        linewidth=0.5
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)

    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    # Add reference lines
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)

    # Add legend for residue types
    if add_legend and not black_points:
        legend_handles = [
            mpatches.Patch(color=color, label=res)
            for res, color in sorted(aa_colors.items())
        ]
        plt.legend(
            handles=legend_handles,
            title="Residue",
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )

    plt.tight_layout()
    plt.show()


# ============================================================================
# LIGAND EXTRACTION AND CONVERSION
# ============================================================================

def ligand_to_smiles(pdb_file: str, ligand_resname: str) -> str:
    """
    Convert a ligand from PDB/mmCIF to SMILES string with proper bond order detection.

    This function extracts a specific ligand from a protein structure file and converts
    it to a SMILES representation, attempting multiple methods for bond perception.

    Parameters
    ----------
    pdb_file : str
        Path to PDB or mmCIF file.
    ligand_resname : str
        Residue name of the ligand (e.g., 'ATP', 'HEM').

    Returns
    -------
    str or None
        SMILES string if successful, None if conversion fails.

    Note
    ----
    Methods used:
        1. RDKit bond perception algorithm (preferred)
        2. Distance-based connectivity as fallback

    Example
    -------
    >>> smiles = ligand_to_smiles("complex.pdb", "ATP")
    >>> if smiles:
    ...     print(f"ATP SMILES: {smiles}")
    """
    # Parse structure based on file extension
    ext = os.path.splitext(pdb_file)[1].lower()
    parser = PDBParser(QUIET=True) if ext == '.pdb' else MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    # Extract ligand atoms and create PDB block
    lines = []
    atom_serial = 1
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname() == ligand_resname:
                    for atom in res:
                        coord = atom.get_coord()
                        elem = atom.element.strip() if atom.element else ''
                        if not elem:
                            # Guess element from atom name
                            name = atom.get_name().strip()
                            if len(name) > 1 and name[1].islower():
                                elem = name[:2].capitalize()
                            else:
                                elem = name[0].upper()
                        elem = elem.ljust(2)
                        line = (
                            f"HETATM{atom_serial:5d} {atom.get_name():<4} "
                            f"{ligand_resname} {chain.id}{res.get_id()[1]:4d}    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  "
                            f"1.00  0.00           {elem}"
                        )
                        lines.append(line)
                        atom_serial += 1

    ligand_pdb_block = "\n".join(lines) + "\nEND\n"

    if not ligand_pdb_block.strip():
        print(f"Ligand '{ligand_resname}' not found in structure.")
        return None

    # Create molecule from PDB block
    mol = Chem.MolFromPDBBlock(ligand_pdb_block, removeHs=False, sanitize=False)
    if mol is None:
        print("Failed to create RDKit molecule from ligand PDB block.")
        return None

    # Method 1: Try RDKit bond perception (preferred)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        return smiles

    except Exception as e:
        print(f"Bond perception failed: {e}")

        # Method 2: Fallback to distance-based connectivity
        try:
            mol = Chem.MolFromPDBBlock(
                ligand_pdb_block,
                removeHs=False,
                sanitize=False
            )
            conf = mol.GetConformer()

            # Add bonds based on distance (< 1.8 Å threshold)
            for i in range(mol.GetNumAtoms()):
                for j in range(i + 1, mol.GetNumAtoms()):
                    pos_i = conf.GetAtomPosition(i)
                    pos_j = conf.GetAtomPosition(j)
                    dist = pos_i.Distance(pos_j)

                    if dist < 1.8 and not mol.GetBondBetweenAtoms(i, j):
                        mol.AddBond(i, j, Chem.BondType.SINGLE)

            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True)
            print("Used fallback distance-based method")
            return smiles

        except Exception as e2:
            print(f"All methods failed: {e2}")
            return None


# ============================================================================
# STRUCTURE PREDICTION
# ============================================================================

def esmfold_predict(sequence: str, output_path: str = "predicted_structure.pdb") -> str:
    """
    Predict a 3D protein structure using the ESMFold API and save it as a PDB file.

    ESMFold is a state-of-the-art protein folding prediction model that generates
    3D structures from amino acid sequences using language model embeddings.

    Parameters
    ----------
    sequence : str
        Protein sequence using single-letter amino acid codes.
    output_path : str, optional
        Path to save the resulting PDB file. Default is "predicted_structure.pdb".

    Returns
    -------
    str or None
        Path to the saved PDB file if successful, None if the request failed.

    Note
    ----
    - Requires internet connection to access ESM Atlas API
    - Sequence length limitations may apply
    - Prediction quality varies with sequence characteristics

    Example
    -------
    >>> sequence = "MKKLVLSLSLVLAFSSATAAFAAIPQNIRIGTDPTYAPFESKNS"
    >>> pdb_file = esmfold_predict(sequence, "my_protein.pdb")
    >>> if pdb_file:
    ...     print(f"Structure saved to {pdb_file}")
    """
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    try:
        response = requests.post(url, data=sequence)

        if response.status_code == 200:
            with open(output_path, "w") as f:
                f.write(response.text)
            return output_path
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Failed to connect to ESMFold API: {e}")
        return None


# ============================================================================
# DATABASE QUERIES
# ============================================================================

def fetch_uniprot_entry(
    query: str,
    query_type: str = "accession",
    fields: list = None,
    format: str = "dict"
) -> dict | pd.DataFrame:
    """
    Fetch an entry from UniProt using the REST API.

    Parameters
    ----------
    query : str
        The query string (e.g., UniProt accession or protein name).
    query_type : str, optional
        Type of query - "accession" or "name". Default is "accession".
    fields : list, optional
        List of fields to retrieve (e.g. ["accession", "protein_name", "sequence"]).
        If None, retrieves all fields.
    format : str, optional
        Output format - "dict" or "dataframe". Default is "dict".

    Returns
    -------
    dict or pd.DataFrame
        Parsed UniProt data.

    Raises
    ------
    ValueError
        If no entry is found or format is invalid.

    Example
    -------
    >>> entry = fetch_uniprot_entry("P12345", query_type="accession")
    >>> print(entry['primaryAccession'])
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{query_type}:{query}",
        "format": "json",
    }
    if fields:
        params["fields"] = ",".join(fields)

    response = requests.get(base_url, params=params)
    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"No UniProt entry found for {query} ({query_type})")

    # Return the first result (since search could return multiple hits)
    entry = results[0]

    if format == "dict":
        return entry
    elif format == "dataframe":
        # Flatten nested dict for DataFrame compatibility
        flat_entry = pd.json_normalize(entry, sep='_')
        return flat_entry
    else:
        raise ValueError("Format must be 'dict' or 'dataframe'")


def fetch_pdb_structure(pdb_id: str, download_dir: str = 'pdbs') -> str:
    """
    Download a PDB structure file from the Protein Data Bank.

    Parameters
    ----------
    pdb_id : str
        PDB identifier (e.g., '1ABC').
    download_dir : str, optional
        Directory to save the downloaded file. Default is 'pdbs'.

    Returns
    -------
    str
        Path to the downloaded PDB file.

    Example
    -------
    >>> pdb_file = fetch_pdb_structure("1ABC", download_dir="structures")
    >>> print(f"Downloaded to {pdb_file}")
    """
    pdbl = PDBList()
    return pdbl.retrieve_pdb_file(pdb_id, pdir=download_dir, file_format='pdb')