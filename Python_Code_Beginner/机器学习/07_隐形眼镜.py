fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readline()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)
print(treePlotter.createPlot(lensesTree))
#trees、treePlotter为05_决策树的代码




