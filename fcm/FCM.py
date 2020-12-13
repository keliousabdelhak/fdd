import pandas as pd
import numpy as np
import random
import operator
import math
import csv
import matplotlib.pyplot as plt

                                                                                                                        #Netoyage du dataset car il contenait des valeurs tel que les noms de ville et les direction on a donc numeriser ces informations pour garder un maximum de donnée
fichier=open("precipitations.csv", "rt")
nb=0
lecteurCSV=csv.reader(fichier,delimiter=";")
out=open("precipitationsUsed.csv", "w")
outw=csv.writer(out)
listecsv=[]
for ligne in lecteurCSV:
    ligne[-1]=ligne[-1].replace("Sud", "1.00")
    ligne[-1]=ligne[-1].replace("Nord", "2.00")
    ligne[-1]=ligne[-1].replace("Est", "3.00")
    ligne[-1]=ligne[-1].replace("Ouest", "4.00")
    if nb==0:
        ligne[0]="Ville"
    else :
        ligne[0]=str(nb)
    nb=nb+1

    listecsv.append(ligne)
outw.writerows(listecsv)
out.close()
fichier.close()


df_full=pd.read_csv("precipitationsUsed.csv")
columns=list(df_full.columns)
df=df_full


k=4                                                                                                                   # nombre de cluster
m=2.00                                                                                                                # le paramtre de fuzzy
n=len(df)                                                                                                             # nombre de ligne de df
nb_iteration=100                                                                                                      # nombre d iteration a realiser




def calculvaleurs(init_m, centers):                                                                                     #mise a jours des elements de notre matrice
    p=float(2/(m-1))
    for i in range(n):
        x=list(df.iloc[i])
        dist=[np.linalg.norm(list(map(operator.sub, x, centers[j]))) for j in range(k)]
        for j in range(k):
            somme=sum([math.pow(float(dist[j]/dist[c]), p) for c in range(k)])
            init_m[i][j]=float(1/somme)
    return init_m

def calculeCenters(init_mat):                                                                                           #calcule les centres des clusters
    centers=[]
    vals=list(zip(*init_mat))

    for j in range(k):
        x=list(vals[j])
        b=[]
        x1=[e**m for e in x]
        somme=sum(x1)


        for i in range(n):
            ligne=list(df.iloc[i])
            produit=[x1[i]*int(val) for val in ligne]
            b.append(produit)
        somme1=map(sum, zip(*b))
        c=[x2/somme for x2 in somme1]
        centers.append(c)
    return centers








def clustering():                                                                                                       #c'est la methode qui est le squelette principale du fuzzy cluster
    nb=0                                                                                                              #c'est le compteur de notre boucle while pour iterer sur le nombre d'iteration

                                                                                                                        #initialisation de notre matrice de départ avec random

    mat_init=[]
    for i in range(n):                                                                                                  # on boucle sur le nombre de ligne
        l1=[random.random() for i in range(k)]                                                                        # creation d'une liste de nombre avec random suivant le nombre de cluster qu'on choisi comme parametre
        somme=sum(l1)                                                                                                 # c'est la somme des nombre créer auparavant
        l2=[x/somme for x in l1]                                                                                    # on devise chaqque elements de la liste sur leur somme
        mat_init.append(l2)

    while nb<=nb_iteration:                                                                                           #notre boucle while qui vas iterer et executer chacune de notre fonction pour appliqué l'algorithme fuzzy c means
        centers=calculeCenters(mat_init)                                                                              #on  calcule d'abords les centres
        mat_init=calculvaleurs(mat_init,centers)                                                                     #mise a jour des elements de la matrice                                                                                                                        #mise a jour de la liste des labels
        labels=[]                                                                                                     #mise a jour de notre liste de label
        for i in range(n):
            a,b=max((a,b) for (b,a) in enumerate(mat_init[i]))
            labels.append(b)
        nb+=1                                                                                                         #on augmente nb pour passer a l'iteration suivante

    return labels, centers


labels, centers=clustering()

                                                                                                                        #affichage dans un plot les points qui ici seront différenticier avec trois couleurs
colormap=np.array(['Red','green','blue','yellow'])

plt.scatter(df_full.loc[:,['Ville']], df_full.loc[:,['JANVIERp']], c=colormap[labels])
plt.title('Classification selon le mois de janvier')
plt.show()

plt.scatter(df_full.loc[:,['Ville']], df_full.loc[:,['Temperature moyenne annuelle']], c=colormap[labels])
plt.title('Classification selon la temperature moyenne annuelle')
plt.show()

plt.scatter(df_full.loc[:,['Ville']], df_full.loc[:,['Nombre annuel de jours de pluie']], c=colormap[labels])
plt.title('Classification le nombre annuel de jours de pluie')
plt.show()

