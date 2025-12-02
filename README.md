<div align="center">

![Raman Shop Logo](pngs/Raman_Shop_Logo.png)


### A collection of useful scripts from the Raman lab, organized by use case. Hosted here to make life easier

---

</div>

## Packages

###  Nori → Plate Reader
**Tools for dealing with plate reader data and manipulating it**

###  Chashu → Molecular Biology Tools
**Tools for dealing with sequences of biological molecules**

###  Tonkotsu → Sequencing Analysis
**Tools for analyzing sequencing data**

###  Ajitama → PyMOL Functions
**Meant to be called by a pymolRC to add useful functions to your PyMOL!**

### Naruto → Molecular Visualization
**Tools for manipulating PDB structures as well as analysis of small molecules**

---

## Documentation

Each package has its own README in the `readme` folder.

## Installation

```bash
conda activate MyEnvironmentName
conda install git pip
pip install git+git://github.com/YourUsername/RepositoryName.git
```

## Basic use

```
from raman_shop import {the package you want to use} as {whatever}

{package}.function(input)

example:

from raman_shop import Nori as nori

data = nori.parse_plate_reader_data("experiment_001.txt")
```

---

<div align="center">

Contributions from members of the Raman Lab. Mostly written and maintained by Jonah O'Mara Schwartz


</div>