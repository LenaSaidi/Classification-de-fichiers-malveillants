# Classification-de-fichiers-malveillants

Le challenge: 

    Theme : Classification de fichiers malveillants
    travail demandé :
    - Analyser les données fournies dans le fichier "ai_test_dataset.csv" 
    - Entraîner et évaluer au moins un modèle d'apprentissage (Machine ou Deep Learning) sur ces données en Python.
    - Fournir le code et tout autre élément jugé utile.
    - Rédiger un texte cours pour expliquer votre démarche et commenter vos résultats.



Démarche :

    Dans ce test, j'ai utilisé un modèle KNN (K-Nearest Neighbors) pour prédire une variable cible à partir 
    des données. J'ai d'abord vérifié les valeurs manquantes et constaté qu'elles étaient très peu nombreuses,
    donc j'ai choisi de les supprimer, car leur impact était faible. Ensuite, j'ai divisé les données en deux
    parties : une pour l'entraînement du modèle et une autre pour tester sa performance. J'ai testé plusieurs 
    valeurs pour le nombre de voisins (de 1 à 12), mais j'ai trouvé que le meilleur k pour mon modèle était 12.

    Cependant, après avoir utilisé GridSearchCV pour tester différents paramètres (comme la distance et les poids),
    j'ai découvert qu'avec les autres paramètres, un k de 13 donnait des résultats légèrement meilleurs.

Commentaire des résultats :

    Les résultats montrent que la précision du modèle varie selon le nombre de voisins. Avec k = 12,
    l'accuracy était de 88.5%, mais après avoir ajusté d'autres paramètres comme la distance (manhattan)
    et les poids (distance), j'ai obtenu un meilleur modèle avec k = 13, avec une précision de 89.5%.

    Cela dit, je suis bien consciente que, dans ce modèle, ce qui importe le plus est la capacité à détecter
    les fonctions malveillantes, surtout dans un contexte de sécurité ou de détection d'anomalies. 
    Le modèle montre de bonnes performances pour la classe majoritaire, mais il reste
    encore à améliorer la détection de la classe minoritaire, qui représente les fonctions malveillantes.

    En résumé, le modèle avec k = 13 et les autres paramètres optimisés a montré de meilleures performances
    globales.