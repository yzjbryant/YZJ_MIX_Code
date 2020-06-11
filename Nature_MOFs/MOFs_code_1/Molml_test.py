from molml.features import CoulombMatrix
feat=CoulombMatrix(input_type='list',n_jobs=1,sort=False,eigen=False,drop_values=False,only_lower_triangle=False)
H2=(['H','H'],
    [
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
    ])
HCN=(
    ['H','C','N'],
    [
        [-1.0,0.0,0.0],
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
    ]
)
feat.fit([H2,HCN])
print(feat.transform([H2]))
print(feat.transform([H2,HCN]))
feat2 = CoulombMatrix(input_type='filename')
paths = ['data/qm7/qm-%04d.out' % i for i in range(2)]
print(feat2.fit_transform(paths))
