import matplotlib.pyplot as plt
import StringIO
from random import gauss
from numpy import asarray
from sklearn import tree
 
from sklearn.externals.six import StringIO
import pydot

data = [[gauss(0.5, 0.2), 1] for i in range(10)] + [[gauss(0.3, 0.2), 0] for i in range(10)]+[[gauss(0.8, 0.2), 0] for i in range(10)]
labels = asarray(['Smurf']*10 + ['Fairy']*10 + ['Troll']*10)
 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, labels)
 
from sklearn.externals.six import StringIO
with open("data.dot",'w') as f:
	f = tree.export_graphviz(clf,feature_names=["Size", "Color"], out_file=f)
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("tree.pdf")
# To actually get the tree picture, you must type in the console the following: dot -T png dot_data.dot -o dot_data.png

print clf.predict([[1,0],[0.5,1],[0.2,0]])
 
['Troll' 'Smurf' 'Fairy'] # Which is what we expect!



