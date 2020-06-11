# import urllib.request
# import requests
# def ss():
#     a=[]
#     req1=requests.request("get",'https://www.materialsproject.org/rest/v2/materials/CO2/vasp?API_KEY=1aEMcIrOQCo7OnTmeR')
#     a.append(req1.text)
#     req2=requests.request("get",'https://materialsproject.org/#search/materials/{"nelements"%3A1%2C"elements"%3A"C"}/vasp?API_KEY=1aEMcIrOQCo7OnTmeR')
#     a2=[]
#     a2.append(req2.text)
#     # print(req.text)
#     print(a2)
# if __name__=='__main__':
#     ss()

# import pymatgen as mg
# >>>
# >>> si = mg.Element("Si")
# >>> si.atomic_mass
# 28.0855
# >>> print(si.melting_point)
# 1687.0 K
# >>>
# >>> comp = mg.Composition("Fe2O3")
# >>> comp.weight
# 159.6882
# >>> # Note that Composition conveniently allows strings to be treated just
# >>> # like an Element object.
# >>> comp["Fe"]
# 2.0
# >>> comp.get_atomic_fraction("Fe")
# 0.4
# >>> lattice = mg.Lattice.cubic(4.2)
# >>> structure = mg.Structure(lattice, ["Cs", "Cl"],
# ...                          [[0, 0, 0], [0.5, 0.5, 0.5]])
# >>> structure.volume
# 74.088000000000008
# >>> structure[0]
# PeriodicSite: Cs (0.0000, 0.0000, 0.0000) [0.0000, 0.0000, 0.0000]
# >>> # You can create a Structure using spacegroup symmetry as well.
# >>> li2o = mg.Structure.from_spacegroup("Fm-3m", mg.Lattice.cubic(3),
#                                         ["Li", "O"],
#                                         [[0.25, 0.25, 0.25], [0, 0, 0]])
# >>>
# >>> # Integrated symmetry analysis tools from spglib.
# >>> from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# >>> finder = SpacegroupAnalyzer(structure)
# >>> finder.get_spacegroup_symbol()
# 'Pm-3m'
# >>>
# >>> # Convenient IO to various formats. You can specify various formats.
# >>> # Without a filename, a string is returned. Otherwise,
# >>> # the output is written to the file. If only the filenmae is provided,
# >>> # the format is intelligently determined from a file.
# >>> structure.to(fmt="poscar")
# >>> structure.to(filename="POSCAR")
# >>> structure.to(filename="CsCl.cif")
# >>>
# >>> # Reading a structure is similarly easy.
# >>> structure = mg.Structure.from_str(open("CsCl.cif").read(), fmt="cif")
# >>> structure = mg.Structure.from_file("CsCl.cif")
# >>>
# >>> # Reading and writing a molecule from a file. Supports XYZ and
# >>> # Gaussian input and output by default. Support for many other
# >>> # formats via the optional openbabel dependency (if installed).
# >>> methane = mg.Molecule.from_file("methane.xyz")
# >>> mol.to("methane.gjf")
# >>>
# >>> # Pythonic API for editing Structures and Molecules (v2.9.1 onwards)
# >>> # Changing the specie of a site.
# >>> structure[1] = "F"
# >>> print(structure)
# Structure Summary (Cs1 F1)
# Reduced Formula: CsF
# abc   :   4.200000   4.200000   4.200000
# angles:  90.000000  90.000000  90.000000
# Sites (2)
# 1 Cs     0.000000     0.000000     0.000000
# 2 F     0.500000     0.500000     0.500000
# >>>
# >>> # Changes species and coordinates (fractional assumed for structures)
# >>> structure[1] = "Cl", [0.51, 0.51, 0.51]
# >>> print(structure)
# Structure Summary (Cs1 Cl1)
# Reduced Formula: CsCl
# abc   :   4.200000   4.200000   4.200000
# angles:  90.000000  90.000000  90.000000
# Sites (2)
# 1 Cs     0.000000     0.000000     0.000000
# 2 Cl     0.510000     0.510000     0.510000
# >>>
# >>> # Replaces all Cs in the structure with K
# >>> structure["Cs"] = "K"
# >>> print(structure)
# Structure Summary (K1 Cl1)
# Reduced Formula: KCl
# abc   :   4.200000   4.200000   4.200000
# angles:  90.000000  90.000000  90.000000
# Sites (2)
# 1 K     0.000000     0.000000     0.000000
# 2 Cl     0.510000     0.510000     0.510000
# >>>
# >>> # Replaces all K in the structure with K: 0.5, Na: 0.5, i.e.,
# >>> # a disordered structure is created.
# >>> structure["K"] = "K0.5Na0.5"
# >>> print(structure)
# Full Formula (K0.5 Na0.5 Cl1)
# Reduced Formula: K0.5Na0.5Cl1
# abc   :   4.209000   4.209000   4.209000
# angles:  90.000000  90.000000  90.000000
# Sites (2)
#   #  SP                   a    b    c
# ---  -----------------  ---  ---  ---
#   0  K:0.500, Na:0.500  0    0    0
#   1  Cl                 0.5  0.5  0.5
# >>>
# >>> # Because structure is like a list, it supports most list-like methods
# >>> # such as sort, reverse, etc.
# >>> structure.reverse()
# >>> print(structure)
# Structure Summary (Cs1 Cl1)
# Reduced Formula: CsCl
# abc   :   4.200000   4.200000   4.200000
# angles:  90.000000  90.000000  90.000000
# Sites (2)
# 1 Cl     0.510000     0.510000     0.510000
# 2 Cs     0.000000     0.000000     0.000000
# >>>
# >>> # Molecules function similarly, but with Site and cartesian coords.
# >>> # The following changes the C in CH4 to an N and displaces it by 0.01A
# >>> # in the x-direction.
# >>> methane[0] = "N", [0.01, 0, 0]
# >>>
# >>> # If you set up your .pmgrc.yaml with your Materials Project API key
# >>> # You can now easily grab structures from the Materials Project.
# >>> lifepo4 = mg.get_structure_from_mp("LiFePO4")