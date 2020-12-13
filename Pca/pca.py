import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm        
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

linestyles = [(0, ()),
              (0, (5, 10)),(0, (5, 5)),(0, (5, 1)), 
              (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)), 
              (0, (3, 10, 1, 10, 1, 10)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1)), 
              (0, (1, 10)),(0, (1, 5)),(0, (1, 1))]
couleurs = cm.Dark2.colors




# Section 1: get data 
print("notre dataset")
df = pd.read_csv('precipitations.csv',sep=';',index_col=0)
print(df)
#end section 1 




# Section 2 :  Clean data => uniquement les variable qui nous interesse : 12 variables correspondent aux mois de l'annee 
print("\n\non prend que les mois ")
data = df._get_numeric_data().values
nomDesVilles = list(df.index)
nomDesVariables_ = list(df)
indice_individus = np.array(range(34))
indice_variables = np.array(range(12))
X = data[indice_individus[:,None],indice_variables] 
print(X)
#end Section 2  


# Section 3 : centré réduire les variables 
print("\n\ndonnes centres reduits")
moyennes = X.mean(axis=0)
ecartTypes = X.std(axis=0,ddof=1)
Xc = X - moyennes
Xcr = Xc / ecartTypes
nomDesVariables = [nomDesVariables_[i] for i in indice_variables]
nomDesIndividus = [nomDesVilles[i] for i in indice_individus]
donnees = pd.DataFrame(data=Xcr, index=nomDesIndividus, columns=nomDesVariables)
donnees.columns = [str(col) + 'CR' for col in donnees.columns]
print(donnees)
# end section 3 


# Section 4 : precipitation en fonction des mois de l'annee
plt.figure(1)
for i, (ligne,label) in enumerate(zip(Xcr, nomDesIndividus)):
    plt.plot(ligne, label=label,
             color = couleurs[i%len(couleurs)],
             linestyle=linestyles[(i//len(couleurs))%len(linestyles)])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True,fontsize=15)
plt.xlabel("Mois de l'année")
plt.ylabel("précipitations (centre reduit)")
plt.grid(True)

# end section 4



# Section 5: matrice des correlations entre les variable
plt.figure(2)
ax = sns.heatmap(df.iloc[indice_individus,indice_variables].corr(), annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)

# end section 5 



# Section 6: example de diperssion entre 2 variable pour estime la lisaison
df_jan_dec = pd.DataFrame(donnees, columns=['DECEMBREpCR','JANVIERpCR'])
ax1 = df_jan_dec.plot.scatter(x='DECEMBREpCR',
                      y='JANVIERpCR',
                      c='DarkBlue')

# end section 6 


# Section 7 : faire de l'Acp sur les donnes centres reduit
print("\n\nles composantes ")
acp = PCA()
CP = acp.fit(Xcr)
lesNouvellesCoordonnees = acp.fit_transform(Xcr)
plesNouvellesCoordonnees = pd.DataFrame(data=lesNouvellesCoordonnees, index=nomDesIndividus, columns=list(range(1,acp.n_features_+1)))
plesNouvellesCoordonnees.columns = ['Cmp' + str(col) for col in plesNouvellesCoordonnees.columns]
print(plesNouvellesCoordonnees)
#end section 7



# Section 8 : inertie portee par chaque axe 
plt.figure(4)
plt.bar(np.arange(len(acp.explained_variance_ratio_))+1,acp.explained_variance_ratio_*100)
plt.plot(np.arange(len(acp.explained_variance_ratio_))+1,np.cumsum(acp.explained_variance_ratio_*100),'r--o')
plt.xlabel("Dimensions",fontsize=12)
plt.ylabel("pourcentage inertie ",fontsize=12)
plt.title("inertie en fonction des dimensions",fontsize=12)
plt.grid(True)
#end section 8 




# section 9 : affichage des nouvelles coordonnes sur 2 axes
plt.subplots(figsize=(18,12))
ax = plt.subplot('{}{}{}'.format(1,1,1))
plt.plot(lesNouvellesCoordonnees[:,0],lesNouvellesCoordonnees[:,1],'o')
plt.title('axes {} et {}'.format(1,2))
if len(nomDesIndividus) != 0 :
    for k in  range(len(nomDesIndividus)):
        plt.text(lesNouvellesCoordonnees[k,0], lesNouvellesCoordonnees[k,1], nomDesIndividus[k])
plt.grid(color='lightgray',linestyle='--')
x_lim = plt.xlim()
ax.arrow(x_lim[0], 0, x_lim[1]-x_lim[0], 0,length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.plot(plt.xlim(), np.zeros(2),'k-')
plt.text(x_lim[1], 0, "axe {}".format(1))
y_lim = plt.ylim()
ax.arrow(0,y_lim[0], 0, y_lim[1]-y_lim[0],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.plot(np.zeros(2),plt.ylim(),'k-')
plt.text(0,y_lim[1], "axe {}".format(2))
plt.show()
#end section 9 