import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
import sys
import csv
#===================================================
#reading all pattern to pattern variable
#===================================================
pattern=list()
fhd=csv.reader(open('pattern.csv','r', encoding='utf-8-sig'))
for line in fhd:
    pattern.append(line)

#===================================================
#conducting pca
#===================================================
n=20
pattern= np.array(pattern, dtype='float64')
pca = PCA(n_components=n)
pca.fit(pattern)
pattern=pca.transform(pattern)
pattern=pattern.tolist()
para=pca.components_
v=pca.explained_variance_ratio_
s=para.tolist()
m=v.tolist()
#===================================================
#get the final predict result and save it to label_predict.csv
#===================================================
output=csv.writer(open('PCA_para.csv','a',newline=''),dialect='excel')
output.writerows(s)
output=csv.writer(open('v.csv','a',newline=''),dialect='excel')
output.writerows(map(lambda x:[x],m))
