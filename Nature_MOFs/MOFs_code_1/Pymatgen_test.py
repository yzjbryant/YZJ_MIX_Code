import pymatgen as mg
# c=mg.Element("O")
# print(c.melting_point)

structure1=mg.Structure.from_file("ABW_1.cif")
print(structure1)

print("++++++++++++++++++++++++")

structure2=mg.Structure.from_file("ABW_2.cif")
print(structure2)

print(structure1!=structure2)