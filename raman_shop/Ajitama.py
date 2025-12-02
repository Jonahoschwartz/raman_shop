"""
PyMOL Utilities Module
======================
A collection of custom functions for PyMOL to enhance visualization,
coloring, and analysis of protein structures.

Author: Various contributors
Modified: 2025
"""

from pymol import cmd, util
from pymol import stored, CmdException
import csv


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def hex_to_rgb(hex_code):
    """
    Convert hex color code to RGB list scaled between 0 and 1.

    Args:
        hex_code (str): Hex color string (e.g., '#FF5733' or 'FF5733')

    Returns:
        list: RGB components in [0.0, 1.0] format [r, g, b]

    Raises:
        ValueError: If hex code is not 6 characters long
    """
    hex_code = hex_code.lstrip('#')

    if len(hex_code) != 6:
        raise ValueError("Hex code must be 6 characters long (e.g., '#FF5733')")

    r = int(hex_code[0:2], 16) / 255.0
    g = int(hex_code[2:4], 16) / 255.0
    b = int(hex_code[4:6], 16) / 255.0

    return [r, g, b]


def color_by_hex(color_name, hex_code, selection="all"):
    """
    Define a PyMOL color using a hex code and apply it to a selection.

    Args:
        color_name (str): Name to register the new color under
        hex_code (str): Hex color string (e.g., '#4FB9AF')
        selection (str): Atom selection to apply color to (default: 'all')
    """
    rgb = hex_to_rgb(hex_code)
    cmd.set_color(color_name, rgb)
    cmd.color(color_name, selection)


def get_colors_rf_diffusion():
    """
    Register RFdiffusion-style color palette for PyMOL.

    Defines a collection of aesthetically pleasing colors commonly used
    in protein design visualizations.
    """
    colors = {
        "good_gray": [220 / 255.0, 220 / 255.0, 220 / 255.0],  # #DCDCDC
        "good_teal": [0.310, 0.725, 0.686],  # #4FB9AF
        "good_navaho": [1.0, 224 / 255.0, 172 / 255.0],  # #FFE0AC
        "good_melon": [1.0, 198 / 255.0, 178 / 255.0],  # #FFC6B2
        "good_pink": [1.0, 172 / 255.0, 183 / 255.0],  # #FFACB7
        "good_purple": [213 / 255.0, 154 / 255.0, 181 / 255.0],  # #D59AB5
        "good_lightblue": [149 / 255.0, 150 / 255.0, 198 / 255.0],  # #9596C6
        "good_blue": [102 / 255.0, 134 / 255.0, 197 / 255.0],  # #6686C5
        "good_darkblue": [75 / 255.0, 95 / 255.0, 170 / 255.0],  # #4B5FAA
    }

    for name, rgb in colors.items():
        cmd.set_color(name, rgb)


def coloraf(selection="all"):
    """
    Color structure by AlphaFold/pLDDT confidence values.

    Colors residues based on B-factor values representing confidence:
        > 90:    blue   (very high confidence)
        70-90:   cyan   (high confidence)
        50-70:   yellow (low confidence)
        <= 50:   orange (very low confidence)

    Args:
        selection (str): Atom selection to color (default: 'all')
    """
    cmd.color("blue", f"{selection} and b>90")
    cmd.color("cyan", f"{selection} and b<90 and b>70")
    cmd.color("yellow", f"{selection} and b<70 and b>50")
    cmd.color("orange", f"{selection} and b<50")


def color_by_csv(csv_path, selection="all", chain=None, colors="blue_white_red"):
    """
    Color structure using per-residue scalar data from CSV file.

    Reads one float value per line from CSV, assigns to B-factor field
    by residue position, and visualizes with PyMOL color spectrum.

    Args:
        csv_path (str): Path to CSV file (one float per line)
        selection (str): PyMOL selection to apply values to (default: 'all')
        chain (str): Optional chain ID to restrict application
        colors (str): PyMOL color spectrum (e.g., 'rainbow', 'blue_white_red')

    Example:
        color_by_csv("scores.csv", selection="myprot", chain="A", colors="rainbow")
    """
    # Read values from CSV
    with open(csv_path, newline='') as csvfile:
        values = [float(row[0]) for row in csv.reader(csvfile)]

    # Assign values to B-factor field by residue index
    for i, val in enumerate(values, start=1):
        sel = f"({selection}) and resi {i}"
        if chain:
            sel += f" and chain {chain}"
        cmd.alter(sel, f"b = {val}")

    # Apply color spectrum based on B-factor values
    cmd.spectrum("b", colors=colors, selection=selection,
                 minimum=min(values), maximum=max(values))
    cmd.rebuild()


# =============================================================================
# RENDERING SETTINGS
# =============================================================================

def get_lighting_rf_diffusion():
    """
    Set high-quality lighting parameters for ray-traced output.

    Configures PyMOL rendering settings for publication-quality figures,
    matching the aesthetic commonly used in RFdiffusion visualizations.
    """
    settings = {
        "specular": 0,
        "ray_shadow": "off",
        "valence": "off",
        "antialias": 2,
        "ray_trace_mode": 1,
        "ray_trace_disco_factor": 1,
        "ray_trace_gain": 0.1,
        "power": 0.2,
        "ambient": 0.4,
        "ray_trace_color": "gray30",
        "cartoon_ring_mode": 1,
    }

    for key, value in settings.items():
        cmd.set(key, value)


# =============================================================================
# ANIMATION
# =============================================================================

def spin_movie(spins=1, seconds=5, fps=30):
    """
    Create a rotating movie around the Y-axis.

    Args:
        spins (int): Number of full 360-degree rotations (default: 1)
        seconds (int): Duration of movie in seconds (default: 5)
        fps (int): Frames per second (default: 30)

    Example:
        spin_movie(spins=2, seconds=10, fps=30)
        # Then export with: mpng frames/frame_.png
    """
    total_frames = int(fps * seconds)
    degrees_per_frame = 360 * spins / total_frames

    cmd.mstop()
    cmd.mclear()
    cmd.mset(f"1 x{total_frames}")

    for frame in range(1, total_frames + 1):
        cmd.mdo(frame, f"turn y, {degrees_per_frame:.4f}")

    cmd.set("ray_trace_frames", 1)
    cmd.set("movie_fps", fps)

    print(f"Movie created: {spins} spins over {seconds}s at {fps} fps")


# =============================================================================
# MUTATION ANALYSIS
# =============================================================================

# BLOSUM90 substitution matrix
AA_3L = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13,
    'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'B': 20, 'Z': 21, 'X': 22, '*': 23
}

BLOSUM90 = [
    [5, -2, -2, -3, -1, -1, -1, 0, -2, -2, -2, -1, -2, -3, -1, 1, 0, -4, -3, -1, -2, -1, -1, -6],
    [-2, 6, -1, -3, -5, 1, -1, -3, 0, -4, -3, 2, -2, -4, -3, -1, -2, -4, -3, -3, -2, 0, -2, -6],
    [-2, -1, 7, 1, -4, 0, -1, -1, 0, -4, -4, 0, -3, -4, -3, 0, 0, -5, -3, -4, 4, -1, -2, -6],
    [-3, -3, 1, 7, -5, -1, 1, -2, -2, -5, -5, -1, -4, -5, -3, -1, -2, -6, -4, -5, 4, 0, -2, -6],
    [-1, -5, -4, -5, 9, -4, -6, -4, -5, -2, -2, -4, -2, -3, -4, -2, -2, -4, -4, -2, -4, -5, -3, -6],
    [-1, 1, 0, -1, -4, 7, 2, -3, 1, -4, -3, 1, 0, -4, -2, -1, -1, -3, -3, -3, -1, 4, -1, -6],
    [-1, -1, -1, 1, -6, 2, 6, -3, -1, -4, -4, 0, -3, -5, -2, -1, -1, -5, -4, -3, 0, 4, -2, -6],
    [0, -3, -1, -2, -4, -3, -3, 6, -3, -5, -5, -2, -4, -5, -3, -1, -3, -4, -5, -5, -2, -3, -2, -6],
    [-2, 0, 0, -2, -5, 1, -1, -3, 8, -4, -4, -1, -3, -2, -3, -2, -2, -3, 1, -4, -1, 0, -2, -6],
    [-2, -4, -4, -5, -2, -4, -4, -5, -4, 5, 1, -4, 1, -1, -4, -3, -1, -4, -2, 3, -5, -4, -2, -6],
    [-2, -3, -4, -5, -2, -3, -4, -5, -4, 1, 5, -3, 2, 0, -4, -3, -2, -3, -2, 0, -5, -4, -2, -6],
    [-1, 2, 0, -1, -4, 1, 0, -2, -1, -4, -3, 6, -2, -4, -2, -1, -1, -5, -3, -3, -1, 1, -1, -6],
    [-2, -2, -3, -4, -2, 0, -3, -4, -3, 1, 2, -2, 7, -1, -3, -2, -1, -2, -2, 0, -4, -2, -1, -6],
    [-3, -4, -4, -5, -3, -4, -5, -5, -2, -1, 0, -4, -1, 7, -4, -3, -3, 0, 3, -2, -4, -4, -2, -6],
    [-1, -3, -3, -3, -4, -2, -2, -3, -3, -4, -4, -2, -3, -4, 8, -2, -2, -5, -4, -3, -3, -2, -2, -6],
    [1, -1, 0, -1, -2, -1, -1, -1, -2, -3, -3, -1, -2, -3, -2, 5, 1, -4, -3, -2, 0, -1, -1, -6],
    [0, -2, 0, -2, -2, -1, -1, -3, -2, -1, -2, -1, -1, -3, -2, 1, 6, -4, -2, -1, -1, -1, -1, -6],
    [-4, -4, -5, -6, -4, -3, -5, -4, -3, -4, -3, -5, -2, 0, -5, -4, -4, 11, 2, -3, -6, -4, -3, -6],
    [-3, -3, -3, -4, -4, -3, -4, -5, 1, -2, -2, -3, -2, 3, -4, -3, -2, 2, 8, -3, -4, -3, -2, -6],
    [-1, -3, -4, -5, -2, -3, -3, -5, -4, 3, 0, -3, 0, -2, -3, -2, -1, -3, -3, 5, -4, -3, -2, -6],
    [-2, -2, 4, 4, -4, -1, 0, -2, -1, -5, -5, -1, -4, -4, -3, 0, -1, -6, -4, -4, 4, 0, -2, -6],
    [-1, 0, -1, 0, -5, 4, 4, -3, 0, -4, -4, 1, -2, -4, -2, -1, -1, -4, -3, -3, 0, 4, -1, -6],
    [-1, -2, -2, -2, -3, -1, -2, -2, -2, -2, -2, -1, -1, -2, -2, -1, -1, -3, -2, -2, -2, -1, -2, -6],
    [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, 1]
]


def get_blosum90_color_name(aa1, aa2):
    """
    Get RGB color representing similarity between two amino acids.

    Uses BLOSUM90 matrix to map amino acid similarity to a color spectrum
    from blue (similar) to red (dissimilar).

    Args:
        aa1 (str): Three-letter code for first amino acid
        aa2 (str): Three-letter code for second amino acid

    Returns:
        str: Hex color name (e.g., '0xff0000')
    """
    # Return red for non-standard residues
    if aa1 not in AA_3L or aa2 not in AA_3L:
        return 'red'

    # Return blue for identical residues
    if aa1 == aa2:
        return 'blue'

    # Get BLOSUM90 score
    i1 = AA_3L[aa1]
    i2 = AA_3L[aa2]
    score = BLOSUM90[i1][i2]

    # Map score to color: 3 is highest for non-identical, subtract 4 to get [-10, -1]
    score = abs(score - 4)

    # Normalize to [0, 1] range
    normalized = 1.0 - (score / 10.0)

    # Interpolate between red and blue
    r = int((1.0 - normalized) * 255)
    g = 0
    b = int(normalized * 255)

    return f'0x{r:02x}{g:02x}{b:02x}'


def color_by_mutation(obj1, obj2, waters=0, labels=0):
    """
    Align and color two proteins to highlight mutations.

    Creates alignment, superimposes structures, and colors mutated residues
    by their BLOSUM90 similarity score (blue=similar, red=dissimilar).

    Args:
        obj1 (str): First protein object or selection
        obj2 (str): Second protein object or selection
        waters (int): If 1, show waters colored by structure (default: 0)
        labels (int): If 1, label mutated sidechains (default: 0)

    Example:
        color_by_mutation protein1, protein2, waters=1, labels=1

    Note:
        - Mutations: blue/red by similarity
        - Conserved: wheat
        - Unaligned: gray
    """
    # Validate inputs
    if cmd.count_atoms(obj1) == 0:
        print(f"{obj1} is empty")
        return
    if cmd.count_atoms(obj2) == 0:
        print(f"{obj2} is empty")
        return

    waters = int(waters)
    labels = int(labels)

    # Create alignment
    aln = '__aln'
    cmd.super(obj2, obj1, object=aln, cycles=0)  # Sequence alignment
    cmd.super(obj2, obj1)  # Structural alignment

    # Store residue information
    stored.resn1, stored.resn2 = [], []
    stored.resi1, stored.resi2 = [], []
    stored.chain1, stored.chain2 = [], []

    cmd.iterate(f"{obj1} and name CA and {aln}", 'stored.resn1.append(resn)')
    cmd.iterate(f"{obj2} and name CA and {aln}", 'stored.resn2.append(resn)')
    cmd.iterate(f"{obj1} and name CA and {aln}", 'stored.resi1.append(resi)')
    cmd.iterate(f"{obj2} and name CA and {aln}", 'stored.resi2.append(resi)')
    cmd.iterate(f"{obj1} and name CA and {aln}", 'stored.chain1.append(chain)')
    cmd.iterate(f"{obj2} and name CA and {aln}", 'stored.chain2.append(chain)')

    # Build selections
    mutant_selection = ''
    non_mutant_selection = 'none or '
    colors = []

    for n1, n2, i1, i2, c1, c2 in zip(stored.resn1, stored.resn2,
                                      stored.resi1, stored.resi2,
                                      stored.chain1, stored.chain2):
        # Handle empty chain names
        c1 = '""' if c1 == '' else c1
        c2 = '""' if c2 == '' else c2

        sel1 = f"{obj1} and resi {i1} and chain {c1}"
        sel2 = f"{obj2} and resi {i2} and chain {c2}"

        if n1 == n2:
            non_mutant_selection += f"(({sel1}) or ({sel2})) or "
        else:
            mutant_selection += f"(({sel1}) or ({sel2})) or "
            color = get_blosum90_color_name(n1, n2)
            colors.append((color, f"{obj2} and resi {i2} and chain {c2} and elem C"))

    if mutant_selection == '':
        print("Error: No mutations found")
        raise CmdException

    # Create selections
    cmd.select('mutations', mutant_selection[:-4])
    cmd.select('non_mutations', non_mutant_selection[:-4])
    cmd.select('not_aligned', f"({obj1} or {obj2}) and not mutations and not non_mutations")

    # Setup visualization
    cmd.hide('everything', f"{obj1} or {obj2}")
    cmd.show('cartoon', f"{obj1} or {obj2}")
    cmd.show('lines', f"({obj1} or {obj2}) and ((non_mutations or not_aligned) and not name c+o+n)")
    cmd.show('sticks', f"({obj1} or {obj2}) and mutations and not name c+o+n")

    # Color scheme
    cmd.color('gray', 'elem C and not_aligned')
    cmd.color('wheat', 'elem C and non_mutations')
    cmd.color('cyan', f"elem C and mutations and {obj1}")

    for (col, sel) in colors:
        cmd.color(col, sel)

    # Hide hydrogens
    cmd.hide('everything', f"(hydro) and ({obj1} or {obj2})")
    cmd.center(f"{obj1} or {obj2}")

    # Optional features
    if labels:
        cmd.label('mutations and name CA', '"(%s-%s-%s)"%(chain, resi, resn)')

    if waters:
        cmd.set('sphere_scale', '0.1')
        cmd.show('spheres', f"resn HOH and ({obj1} or {obj2})")
        cmd.color('red', f"resn HOH and {obj1}")
        cmd.color('salmon', f"resn HOH and {obj2}")

    # Cleanup
    cmd.delete(aln)
    cmd.deselect()

    print(f"""
    Mutations highlighted:
    - {obj1} mutated sidechains: cyan
    - {obj2} mutated sidechains: blue→red by BLOSUM90 similarity
    - Conserved regions: wheat
    - Unaligned regions: gray

    Note: Mutations in unaligned regions may not be detected.
    """)


# =============================================================================
# DNA ANALYSIS
# =============================================================================

def color_by_chains():
    """Color each chain uniquely across all objects."""
    for obj in cmd.get_names('objects'):
        util.color_chains(f'{obj} and e. c')


def dna_selections(display='all'):
    """
    Create useful selections for protein-DNA interfaces.

    Args:
        display (str): Display mode - 'all', 'labels', or 'none'

    Selections created:
        - DNA: all DNA residues
        - DNAbases: DNA bases only
        - DNAbb: DNA backbone
        - sc_base: protein sidechains near DNA bases
        - dna_h2o: waters near DNA
    """
    bb_atoms = ('name C2\*+C3\*+C4\*+C5\*+P+O3\*+O4\*+O5\*+O1P+O2P+'
                'H1\*+1H2\*+2H2\*+H3\*+H4\*+1H5\*+2H5\*+'
                'c2\'+c3\'+c4\'+c5\'+o3\'+o4\'+o5\'+op2+op1+'
                'h1\'+1h2\'+2h2\'+h3\'+h4\'+1h5\'+2h5\'')

    waters = 'n. wo6+wn7+wn6+wn4+wo4 or r. hoh'

    # DNA selections
    cmd.select('DNA', 'r. g+a+c+t+gua+ade+cyt+thy+da+dc+dg+dt+5mc', enable=0)
    cmd.select('notDNA', 'not DNA', enable=0)
    cmd.select('DNAbases', f'DNA and not {bb_atoms}', enable=0)
    cmd.select('DNAbb', f'DNA and {bb_atoms}', enable=0)

    # Interface selections
    cmd.select('sc_base', 'byres notDNA w. 7 of DNAbases', enable=0)
    cmd.select('sc_base', 'sc_base and not n. c+n+o', enable=0)
    cmd.select('dna_h2o', f'{waters} w. 3.6 of DNAbases', enable=0)

    # Water styling
    cmd.set('sphere_transparency', '0.5')
    cmd.color('marine', 'dna_h2o')

    # Color chains
    cmd.color('gray', 'e. c')

    # Protein backbone
    cmd.select('pbb', 'notDNA and n. c+n+ca', enable=0)

    # Polar protons
    cmd.do('selectPolarProtons')

    # Labels
    if display != 'none':
        cmd.label("n. c1\*+c1\' and DNA", "'%s%s(%s)' % (chain,resi,resn)")
        cmd.set('label_color', 'white')

    # Display
    if display == 'all':
        cmd.show('sticks', 'DNAbases or sc_base')
        cmd.show('ribbon', 'DNAbb')
        cmd.show('cartoon', 'notDNA')
        cmd.show('spheres', 'dna_h2o')
        cmd.hide('everything', 'e. h and not polar_protons')


# =============================================================================
# PYMOL COMMAND REGISTRATION
# =============================================================================

# Color utilities
cmd.extend("color_by_hex", color_by_hex)
cmd.extend("coloraf", coloraf)
cmd.extend("color_by_csv", color_by_csv)
cmd.extend("get_colors_rf_diffusion", get_colors_rf_diffusion)

# Rendering
cmd.extend("get_lighting_rf_diffusion", get_lighting_rf_diffusion)

# Animation
cmd.extend("spin_movie", spin_movie)

# Mutation analysis
cmd.extend("color_by_mutation", color_by_mutation)

# DNA analysis
cmd.extend("cbce", color_by_chains)
cmd.extend("DNAselections", dna_selections)
cmd.extend("DNAselections_nodisplay", lambda: dna_selections('none'))
cmd.extend("DNAselections_labelsonly", lambda: dna_selections('labels'))

# Auto-completion
cmd.auto_arg[0]["coloraf"] = [cmd.object_sc, "object", ""]
cmd.auto_arg[0]["color_by_mutation"] = [cmd.object_sc, "object", ""]
cmd.auto_arg[1]["color_by_mutation"] = [cmd.object_sc, "object", ""]