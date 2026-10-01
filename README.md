# Applicazione di tecniche interpretabili e neuro-simboliche sul dataset KANDY-Easy

## Obiettivi dell'elaborato

L'obbiettivo dell'elaborato è quello di andare a osservare ed analizzare l'utilizzo il comportamento di modelli interpretabili e neuro-simbolici sul dataset di KANDY-Easy [link1]; per realizzare tale obiettivo, il lavoro è stato diviso in quattro notebook, uno per ciascuna fase del progetto:

1. Inizialmente andiamo ad approfondire il dataset in esame.
2. Andiamo poi a confrontare i comportamenti di diverse tecniche interpretabili su tale dataset.
3. Dopodiché, prepariamo la strada per le tecniche neurosimboliche, esplorando diversi tipi di encoder visivi.
4. Infine, utilizziamo la tecnica neuro-simbolica dei Concept Bottleneck Models (CBM)

Per quanto riguarda le tecniche interpretabili, estrarremo alcune feature percettive direttamente dalle immagini del dataset tramite l'utilizzo di moduli di computer vision, tali feature verrano poi utilizzate nell'addestramento dei seguenti modelli:

- Regressione Logistica
- Alberi di decisione
- Explainable Boosting Machine

Come modelli candidati per l'encoder visivo del CBM sono stati poi messe a confronto le seguenti architetture:

- CNN (da zero)
- Resnet-18 (pre-addestrata su imagenet e da 0)
- Vit-Tiny/16 (pre-addestrata su imagenet e da 0)

Oltre a misurarne le prestazioni, i modelli sono stati anche analizzati con tecniche post-hoc basate sul gradiente.

Per concludere, sono state testate due versioni di CBM che si differenziano nella testa di classificazione usata:

- Nella prima si utilizza una testa di classificazione lineare
- Nella seconda viene utilizzato un piccolo MLP

Oltre la fine, DA CONCLUDERE

---

### Organizzazione

All'interno della repository si trovano le seguenti cartelle:

- data: contenente tutto ciò che concerne il dataset KANDY-easy
- notebooks: contenente, appunto, i notebooks jupyter che compongono l'elaborato
- results:  contenente i risultati degli esperimenti e i grafici qui mostrati
- configs: contenente i file .yml per l'impostazione dei parametri utilizzati negli esperimenti
- prompt: contenente

---

### Fonti Utilizzate DA SISTEMARE

- Paper 1
- Paper 2
- Paper 3
- Documentazione 1
- Documentazione 2

#### Large Language Models DA SPECIFICARE

- Chatgpt
- Claude

---

## Notebook 1 - Analisi del Dataset

L'obbiettivo del primo notebook è quello di andare a familiarizzare ed esplorare il dataset utilizzato.
Ho scelto il dataset KANDY perché mi sembrava quello che tra quelli proposti più si prestava per iniziare a lavorare su questo tipo di problemi, data la disponibilità di descrizioni simboliche ricche per ciascun compito esaminato; inoltre, i test sono stati svolti interamente sulla variante "Easy" del dataset, questo in quanto, per iniziare, ho reputato fosse meglio concentrarsi su compiti "semplici" per capire meglio dove i modelli ecellono, dove faticano e di cosa hanno bisogno per migliorare le loro prestazioni; si riserva dunque la variante "Hard" per futuri approfondimenti.

Prima di osservare le analisi svolte sui dati e i loro risultati, introduciamo brevemente il dataset.

KANDY-Easy è un dataset che si ispira alle forme geometriche presenti nei quadri di Kandinsky ed è composto da 2000 immagini a sfondo grigio su cui campeggiano una o più forme geometriche legate tra di loro da una certa relazione; le 2000 immagini sono divise in 1000 di addestramento, 500 di valutazione e 500 di test, ciascuno di questi set si divide a sua volta in 20 task, in cui: i primi 10 riguardano singoli oggetti e le loro posizioni all'interno delle immagini mentre i secondi 10 si concentrano maggiormente sulle relazioni tra le forme geometriche. I task sono binari, data cioè una domanda del tipo "vi è un triangolo rosso in questa immagine?", il modello deve rispondere "s1" oppure "n0". Il dataset è inoltre generato,tramite un sistema basato su rappresentazioni simboliche ad albero e regole interpretabili, a partire dalla descrizione simbolica per ciascun task contenuta in un file yaml; tali descrizioni risulteranno molto utili non solo per l'analisi dei modelli ma anche per capire quali feature e concetti utilizzare.

Adesso procediamo con l'analisi del dataset.

Comcinciamo con l'osservare quelli che sono i compiti presenti:

<details>
<summary><strong>Task di KANDY-Easy</strong></summary>

| ID | Task |
|---:|---|
| 0 | triangle vs any |
| 1 | square vs any |
| 2 | circle vs any |
| 3 | red vs any |
| 4 | green vs any |
| 5 | blue vs any |
| 6 | cyan vs any |
| 7 | magenta vs any |
| 8 | yellow vs any |
| 9 | red triangle on the right |
| 10 | red triangle on the right and arbitrary objects |
| 11 | red triangle on the right and at least one circle and arbitrary objects |
| 12 | red triangle on the right and at least one blue object and arbitrary objects |
| 13 | triangle and square, same color |
| 14 | palindrome ABA |
| 15 | house |
| 16 | car |
| 17 | tower |
| 18 | wagon |
| 19 | traffic light |

</details>

Come avevamo già fatto notare, la difficoltà sale gradualmente: partendo da compiti con un solo oggetto e quasi nessuna relazione sino a compiti con molti oggetti e relazioni.



Sono andato poi a vedere il numero di esempi per set e per task, per vedere quale fosse lo split dei dati e se vi fossero compiti sovra- o sotto-rappresentati;
per quanto riguarda gli split abbiamo che:

- Train samples: 1000
- Validation samples: 500
- Test samples: 500

Mentre per la distribuzione dei task abbiamo che:

<details>
<summary><strong>Distribuzione esempi nel dataset</strong></summary>

### Training

![Training](results/dataset_exploration/task_distribution_train.png)

### Validation

![Validation](results/dataset_exploration/task_distribution_val.png)

### Test

![Test](results/dataset_exploration/task_distribution_test.png)

</details>

Come si nota dai grafici, i compiti sono bilanciati: ciascun task nel training set ha 50 esempi, mentre nel validation e nel test ha 25 esempi.

Successivamente ho analizzato le distribuzioni globali e non, delle label degli esempi: 

<details>
<summary><strong>Distribuzione labels nel dataset</strong></summary>

### Training

![Training](results/dataset_exploration/label_distribution_train.png)
![Training](results/dataset_exploration/positive_ratio_train.png)

### Validation

![Validation](results/dataset_exploration/label_distribution_val.png)
![Validation](results/dataset_exploration/positive_ratio_val.png)

### Test

![Test](results/dataset_exploration/label_distribution_test.png)
![Test](results/dataset_exploration/positive_ratio_test.png)

</details>

Si osserva che, globalmente, gli esempi negativi sono maggiormente rappresentati degli esempi positivi (60%-40%) su ciascuno split; questo è dovuto al modo in cui i task vengono generati: si cerca infatti di mantenere il 50%-50% di esempi, ma se lo spazio di configurazioni simboliche non è sufficiente allora la distribuzione delle label può riflettere la diversa disponibilità di configurazioni. Inoltre, questo ci dice che il modello baseline che semplicemente assegna la label "0" a tutti gli esempi che gli vengono proposti ha un'accuracy prossima al 60%, quello che ci aspettiamo è dunque che i nostri modelli superino, più o meno ampiamente tale risultato. A livello locale lo scenario risulta più variegato tra i compiti, ad esempio: nel training set il compito 2 ha 20 positivi e 30 negativi mentre il compito 19 ha 5 positivi e 45 negativi; questo comportamento risulta essere però simile tra tutti gli split del dataset. 

Queste osservazioni ci portano alla seguente conclusione: per valutare i nostri modelli non potremo fare affidamento soltanto sull'accuracy, in quanto una buona accuratezza può essere ottenuta anche semplicemente sfruttando lo sbilanciamento tra le classi; sarà quindi necessario considerare anche metriche come precision, recall e F1-score, le quali ci permettono di valutare in modo più completo la capacità del modello di riconoscere correttamente entrambe le classi.

Dopodiché ho analizzato la distribuzione dei concetti presenti all'interno dei vari compiti, questo per questo per completare l'analisi della struttura del dataset e capire quali proprietà degli oggetti e quali relazioni risultano maggiormente rappresentate nei diversi task.

<details>
<summary><strong>Distribuzione concetti nel dataset</strong></summary>

### Training

![Training](results/dataset_exploration/object_properties_train.png)

### Validation

![Validation](results/dataset_exploration/object_properties_val.png)

### Test

![Test](results/dataset_exploration/object_properties_test.png)

</details>

Sebbene le distribuzioni siano comunque molto simili, possiamo notare che:

- triangle è la forma più rappresentata
- red è il colore più rappresentato
- side_by_side è la posizione più rappresentata
- large è la dimensione più rappresentata

Ho visualizzato poi la distribuzione di oggetti presenti all'interno delle immagini

<details>
<summary><strong>Distribuzione oggetti nel dataset</strong></summary>

### Training

![Training](results/dataset_exploration/object_count_train.png)

### Validation

![Validation](results/dataset_exploration/object_count_val.png)
### Test

![Test](results/dataset_exploration/object_count_test.png)

</details>

Si nota che la maggior parte degli esempi ha un solo oggetto.

Ho poi estratto dalle descrzioni simboliche alcuni concetti interpretabili (ex: circle, yellow, qudrant_lr, small, 1 oggetto) che ci serviranno successivamente per andare a costruire i modelli interpretabili e neuro-simbolici.


<details>
<summary><strong>Distribuzione labels nel dataset</strong></summary>

### Training

![Training](results/dataset_exploration/GRAFICO_TRAIN.png)

</details>

Infine sono andato a plottare alcuni esempi appartenenti al dataset.

<details>
<summary><strong>Esempi</strong></summary>

### Esempi

![Training](results/dataset_exploration/task_0_examples.png)
![Test](results/dataset_exploration/task_9_examples.png)
![Test](results/dataset_exploration/task_11_examples.png)
![Test](results/dataset_exploration/task_14_examples.png)
![Test](results/dataset_exploration/task_19_examples.png)

</details>

### Conclusioni: 

L'analisi del primo notebook è dunque servita per comprendere meglio la struttura di KANDY-Easy e la distribuzione dei dati nei diversi task; in particolare, sono emerse una maggiore presenza di esempi negativi e un livello di difficoltà crescente tra i vari compiti, con quelli più strutturati che richiedono informazioni più complesse rispetto ai semplici attributi di forma o colore. Questa analisi è stata utile anche per individuare quali proprietà e relazioni risultano più rappresentate nel dataset e, soprattutto, per capire quali informazioni possono essere utilizzate nella costruzione di una rappresentazione più interpretabile.

---

## Notebook 2 - Modelli Interpretabili

Nel secondo notebook, l'obiettivo è quello di andare a provare diversi modelli interpretabili come regressione logistica, alberi interpretabili ed Explainable Boosting Machine, tuttavia, per farlo dobbiamo estrarre delle feature rilevanti dagli esempi del dataset, motivo per cui questo modulo del progetto segue questo flusso: 

immagine -> percezione oggetti -> feature interpretabili -> classificatore -> predizione task

Procediamo dunque a mostrare il processo che dall'immagine ci porta alle feature interpretabili.

### Percezione

Per la percezione, vi era il bisogno che le feature fossero interpretabili, motivo per cui esse sono state estratte utilizzando alcuni metodi di computer vision provenienti dalla libreria di OpenCV, mantenendo così il processo completamente deterministico; per costruire il sistema di percezione, ho dunque cominciato seguendo quella che è la difficoltà del dataset: mi sono quindi inizialmente concentrato sui concetti di "colore" e di "forma".

Colore

In questo caso il sistema procede a indiviudare i colori secondo tale procedura:

1. Si crea un array contenente i colori presenti nel dataset espressi in RGB
2. Si converte l'immagine in RGB
3. Dopodiché per ogni pixel nell'immagine RGB si crea un array della dimensione del numero di colori presenti nel dataset (un cubo HxWxn_colors)
4. Andiamo a riempire gli array con la norma delle distanze tra i valori del pixel in esame e i colori presenti nel dataset
5. "Appiattiamo" la rappresentazione estrando per ogni array il colore alla distanza minima
6. "Appiattiamo" la rappresentazione estrando per ogni array la distanza minima
7. Si inizializza un template in cui ogni cella è impostata a -1
8. Iteriamo poi sui colori e modifichiamo i -1 nel template con l'indice del colore più vicino; per fare questo verifichiamo due condizioni e cioè l'indice è il colore più vicino e la distanza tra i due è minore di una certa soglia. 

Forma&Dimensioni

Per le forme il processo è leggermente più complesso:

1. Utilizzando alcuni metodi dalla libreria di OpenCV andiamo a "dipingere" le tre forme fondamentali su di una tela (cerchio, triangolo, quadrato).
2. Per ciascuna di esse, andiamo a estrarne i contorni.
3. Una volta fatto ciò selezioniamo il contorno con l'aria maggiore, questo sarà il nostro template.
4. Una volta avuti i riferimenti astratti ci occupiamo di ritrovarli nell'immagine concreta, per farlo definiamo 3 funzioni:
5. Una funzione che data una maschera (matrice di 0/1) ritorna il contorno con l'area maggiore
6. Una funzione che data una componente (e cioè l'output della funzione al punto .5), va a confrontarle con tutte le forme predisposte all'inizio e ritorna la distanza minima e il nome della shape a distanza minima
7. Una funzione che presa un'immagine, utilizza prima la funzione utilizzata per i colori e ottenere una matrice con gli indici di colori, dopodiché iteriamo sui colori e, segmentando, andiamo a ottenere una maschera per ciascun colore. Ottenuta questa maschera andiamo a rifinirla e, successivamente, cerchiamo le componenti connesse, le numeriamo e dopo aver controllato che l'area coperta è abbastanza grande andiamo a lanciare la funzizone 6. sulla componente connessa, ottenento l'oggetto.
8. Siccome vogliamo anche sapere se l'oggetto è small o large, prima di ritornare la lista degli oggetti definitiva, utilizziamo una funzione che stabilisce in automatico la soglia che divide correttamente small e large: identifica le aree di tutti gli oggetti nel train split (usa la 7.) e poi utilizziamo k-means per ottenere due gruppi distinti, ne otteniamo due centri e ne facciamo la media, quella sarà la nostra soglia.

Una volta fatto ciò ho testato testato il sistema di percezione elaborato sin qui, ottenendo i seguenti risultati:

| Split | Mean truth | Mean detected | MAE count | Count exact | Shape accuracy | Color accuracy | Size accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2.065 | 2.027 | 0.038 | 0.963 | 0.950 | 1.000 | 0.873 |
| Validation | 2.086 | 2.036 | 0.050 | 0.954 | 0.966 | 1.000 | 0.867 |
| Test | 2.068 | 2.022 | 0.046 | 0.954 | 0.947 | 1.000 | 0.871 |

![Perception](results/interpretable_models/plots/perception_visual_check.png)

Il sistema risulta performante tuttavia, per arrivare a questa performance è stato necessario tunare i valori delle soglie in particolare la soglia del metodo color_label_image risulta essere quella più critica, in quanto una soglia sbagliata "acceca" il sistema: una soglia troppo bassa infatti non fa cambiare il valore nella matrice utilizzata per andare a fare la detection del colore, la stessa che poi viene utilizzata per andare a fare detection dell'oggetto, ciò importa che se, ad esempio, il bordeaux è "troppo distante" dal rosso, l'oggetto non viene rilevato dal sistema.

Arrivati a questo punto, andando a individuare gli oggetti nei vari esempi nello split e andando ad aggiungere anche alcuni riferimenti riguardanti la posizione di essi (posizioni medie e variazioni standard), otteniamo le seguenti feature interpretabili:

Feature percettive: 'n_objects', 'count_shape_triangle', 'mean_x_triangle', 'mean_y_triangle', 'count_shape_square', 'mean_x_square', 'mean_y_square', 'count_shape_circle', 'mean_x_circle', 'mean_y_circle', 'count_color_red', 'count_color_green', 'count_color_blue', 'count_color_cyan', 'count_color_magenta', 'count_color_yellow', 'count_size_small', 'count_size_large', 'mean_x_all', 'mean_y_all', 'std_x_all', 'std_y_all'.

Tuttavia provando i modelli interpretabili con solo queste feature, i risultati su alcuni task più difficili (quelli cioè che includono delle relazioni) non mi soddisfacevano, ho quindi cercato di utilizzare più informazioni presenti all'interno dei dati andando a definire alcune feature relazionali.

Relazioni

Per le feature che riguardano le relazioni spaziali tra gli oggetti, sono andato a definire un insieme ristretto di feature riguardanti il legame tra coppie di oggetti; fatto ciò sono andato a iterare su tutte le coppie di oggetti presenti in tutte le immagini del train set e, per ciascuna di queste coppie, applicando alcuni controlli su distanza e colore, e ho ottenuto dei dizionari con valori binari 1/0 che segnalavano la presenza o meno della relazione tra le coppie di oggetti. Ho dunque ottenuto:

Feature relazionali: 'pair_left_right', 'pair_above_below', 'pair_diagonal', 'pair_same_shape', 'pair_same_color', 'pair_same_size', 'pair_horiz_aligned', 'pair_vert_aligned', 'pair_close'.

Con questo si conclude la fase di percezione, andiamo dunque a vedere quali sono le perfomance dei modelli, utilizzando queste feature.

### Addestramento e Valutazione

In questa fase del progetto sono a andato ad addestrare e valutare sia la Regressione Logistica sia i Decision Tree, utilizzando le seguenti metriche:

- Precision: quante previsioni corrette sulle previsioni totali
- Recall: quanti positivi sono stati individuati tra i positivi totali
- F1: media armonica tra le precedenti

Per ognuno dei 20 task eseguiamo lo stesso procedimento: alleniamo sia il modello di Logistic Regression sia il Decision Tree sul training set, valutiamo le performance dei due modelli tramite la metrica F1-score sul validation set e, una volta individuato il modello migliore, lo mettiamo alla prova sul test set. La scelta viene effettuata considerando la metrica F1, in quanto permette di ottenere un modello bilanciato tra precision e recall. Si vuole dunque evitare sia un modello con precision alta ma recall bassa, ovvero un modello che effettua poche predizioni errate tra quelle positive ma che riesce a individuare solo una parte dei casi realmente positivi, sia un modello con precision bassa ma recall alta, cioè un modello che individua una grande quantità di casi positivi ma produce anche numerosi falsi positivi.


Partiamo dunque dalle configurazioni dei modelli:

<details>
<summary><strong>Parametri Logistic Regression</strong></summary>

```python
"logistic": lambda: Pipeline([
    (
        "scale",
        StandardScaler(),
    ),
    (
        "clf",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
    ),
]),
```
</details>


Utilizziamo lo standard scaler perché, abbiamo feature con scale diverse: conteggi, coordinate, distanze, feature binarie, ecc. standardizzarle quindi evita che quelle numericamente più grandi abbiano un'influenza maggiore solamente per la loro scala. Utilizziamo poi i pesi di classe "balanced" perché, come visto nel notebook 1, gli esempi delle due classi sono spesso sbilanciate.


<details>
<summary><strong>Parametri Decision Tree</strong></summary>

```python
"tree": lambda: DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=SEED,
),
```
</details>

Nel Decision Tree ritroviamo i pesi di classe "balanced" per lo stesso motivo della regressione logistica; abbiamo poi max_depth=5 per mantenere l'albero corto e dunque altamente interpretabile, e min_samples_leaf=4 impedisce di creare foglie basate su pochissimi esempi, riducendo il rischio di overfitting.

Osserviamo dunque i risultati del test, in queste due tabelle troviamo le performance utilizzando solo le feature percettive (tabella a sinistra) e quelle utilizzando le feature percettive e quelle relazionali:

```html
<table>
<tr>
<td valign="top">

### Feature Percettive

| task_id | task | model | val_f1 | accuracy | precision | recall | f1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | triangle vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 1 | square vs any | logistic | 0.962963 | 0.96 | 0.900000 | 1.000000 | 0.947368 |
| 2 | circle vs any | logistic | 0.933333 | 0.92 | 1.000000 | 0.857143 | 0.923077 |
| 3 | red vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 4 | green vs any | logistic | 0.947368 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 5 | blue vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 6 | cyan vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 7 | magenta vs any | logistic | 0.909091 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 8 | yellow vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 9 | red triangle on the right | logistic | 0.800000 | 0.96 | 0.888889 | 1.000000 | 0.941176 |
| 10 | red triangle on the right and arbitrary objects | logistic | 0.814815 | 0.84 | 0.933333 | 0.823529 | 0.875000 |
| 11 | red triangle on the right and at least one circle | logistic | 0.864865 | 0.80 | 0.761905 | 1.000000 | 0.864865 |
| 12 | red triangle on the right and at least one blue object | logistic | 0.777778 | 0.72 | 0.722222 | 0.866667 | 0.787879 |
| 13 | triangle and square, same color | tree | 0.838710 | 0.80 | 0.833333 | 0.882353 | 0.857143 |
| 14 | palindrome aba | tree | 0.695652 | 0.76 | 0.875000 | 0.583333 | 0.700000 |
| 15 | house | logistic | 0.888889 | 0.60 | 0.416667 | 0.625000 | 0.500000 |
| 16 | car | tree | 0.818182 | 0.92 | 0.900000 | 0.900000 | 0.900000 |
| 17 | tower | logistic | 0.967742 | 0.96 | 0.888889 | 1.000000 | 0.941176 |
| 18 | wagon | logistic | 0.965517 | 0.84 | 0.875000 | 0.875000 | 0.875000 |
| 19 | traffic light | tree | 0.666667 | 0.56 | 0.166667 | 0.666667 | 0.266667 |

</td>
<td valign="top">

### Feature Percettive + Relazionali

| task_id | task | model | val_f1 | accuracy | precision | recall | f1 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | triangle vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 1 | square vs any | logistic | 0.962963 | 0.96 | 0.900000 | 1.000000 | 0.947368 |
| 2 | circle vs any | logistic | 0.933333 | 0.92 | 1.000000 | 0.857143 | 0.923077 |
| 3 | red vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 4 | green vs any | logistic | 0.947368 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 5 | blue vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 6 | cyan vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 7 | magenta vs any | logistic | 0.909091 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 8 | yellow vs any | logistic | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 9 | red triangle on the right | logistic | 0.800000 | 0.96 | 0.888889 | 1.000000 | 0.941176 |
| 10 | red triangle on the right and arbitrary objects | logistic | 0.838710 | 0.84 | 0.933333 | 0.823529 | 0.875000 |
| 11 | red triangle on the right and at least one circle | logistic | 0.888889 | 0.76 | 0.750000 | 0.937500 | 0.833333 |
| 12 | red triangle on the right and at least one blue object | tree | 0.733333 | 0.80 | 0.777778 | 0.933333 | 0.848485 |
| 13 | triangle and square, same color | logistic | 1.000000 | 0.96 | 0.944444 | 1.000000 | 0.971429 |
| 14 | palindrome aba | tree | 0.769231 | 0.68 | 0.666667 | 0.666667 | 0.666667 |
| 15 | house | logistic | 0.833333 | 0.48 | 0.307692 | 0.500000 | 0.380952 |
| 16 | car | tree | 0.941176 | 0.72 | 0.666667 | 0.600000 | 0.631579 |
| 17 | tower | logistic | 0.967742 | 0.96 | 0.888889 | 1.000000 | 0.941176 |
| 18 | wagon | logistic | 0.965517 | 0.96 | 1.000000 | 0.937500 | 0.967742 |
| 19 | traffic light | logistic | 0.800000 | 0.80 | 0.333333 | 0.666667 | 0.444444 |

</td>
</tr>
</table>
```

CCome si può vedere dai risultati, introdurre le feature relazionali lascia completamente invariata la performance sui primi task, 0-8, che dipendono principalmente dalle proprietà dei singoli oggetti, come forma e colore. Nei task successivi, invece, l'effetto delle feature relazionali è meno uniforme: in alcuni casi portano a un miglioramento, mentre in altri non portano benefici o addirittura peggiorano la performance. In particolare, si osservano miglioramenti nei task 12, 13, 18 e 19. Il miglioramento più marcato riguarda il task 13, relativo alla presenza di un triangolo e di un quadrato dello stesso colore, dove l'F1 passa da 0.857 a 0.971, mentre nel task 18, relativo al wagon, l'F1 passa da 0.875 a 0.968, nel task 12 l'F1 passa invece da 0.788 a 0.848, e infine, nel task 19 passa da 0.267 a 0.444.

Questo suggerisce che introdurre informazioni sulle relazioni tra gli oggetti può aiutare i modelli a rappresentare meglio alcuni task in cui la sola descrizione dei singoli elementi non è sufficiente. Tuttavia, il comportamento non è uniforme: le feature relazionali non sono sempre utili e in alcuni casi possono introdurre informazione poco utile o addirittura dannosa per il classificatore, ad esempio: il task 16 passa da un F1 di 0.900 a 0.632, mentre il task 15 passa da 0.500 a 0.381.

Nei task 9, 10 e 17, invece, la performance sul test set rimane sostanzialmente invariata. Questo indica che in questi casi le feature relazionali aggiuntive non portano un'informazione sufficiente a migliorare ulteriormente la capacità del modello.

Notiamo inoltre che con le feature percettive estratte precedentemente, nella maggior parte dei casi la Logistic Regression performa meglio del Decision Tree. Anche dopo l'introduzione delle feature relazionali la Logistic Regression rimane il modello più frequentemente selezionato. Questo suggerisce che le feature estratte siano già sufficientemente informative e che non sia sempre necessario utilizzare un modello in grado di rappresentare relazioni non lineari più complesse.

Nel complesso, quindi, le feature relazionali sembrano essere utili soprattutto nei task in cui è importante combinare le proprietà degli oggetti con la loro disposizione o con le relazioni che intercorrono tra essi, ma il loro effetto dipende dalla struttura specifica del problema. L'aggiunta di queste feature non porta quindi a un miglioramento generale delle performance, ma può risultare vantaggiosa in alcuni task specifici.


<details>
<summary><strong>Risultati su alcuni Task</strong></summary>

#### Task 0

![interp](results/interpretable_models/task_0_selected_logistic.png)

Nel task 0 la feature più importante è count_shape_triangle, che presenta un coefficiente fortemente positivo. Coerentemente con la richiesta del task quindi, la presenza di triangoli è l'elemento principale utilizzato per riconoscere la classe positiva e, al contrario, count_shape_circle e count_shape_square hanno coefficienti fortemente negativi, indicando che la presenza di forme diverse è associata alla classe negativa. Le informazioni spaziali, come mean_x_triangle e mean_y_square, hanno invece un contributo più contenuto. Il modello si basa quindi soprattutto sulle forme.

#### Task 2

![interp](results/interpretable_models/task_2_selected_logistic.png)

Anche nel task 2 la forma è l'informazione principale: count_shape_circle ha il coefficiente positivo più elevato, mentre count_shape_triangle è fortemente negativo. È interessante anche il contributo positivo di count_color_cyan e count_size_small, che forniscono informazione aggiuntiva senza essere strettamenti legate alla richiesta del task. Le feature count_shape_square, count_size_large e mean_y_circle contribuiscono invece negativamente. Il modello sembra quindi riconoscere soprattutto la presenza dei cerchi, utilizzando le altre caratteristiche per distinguere meglio le due classi.

#### Task 9

![interp](results/interpretable_models/task_9_selected_logistic.png)

Nel task 9 il contributo maggiore è dato da count_color_red, seguito da mean_x_triangle e count_shape_triangle. La presenza di un triangolo rosso e, soprattutto, una maggiore coordinata x del triangolo sono quindi fortemente associate alla classe positiva come ci si aspetterebbe. È interessante anche pair_same_color, che presenta un coefficiente negativo: in questo caso il fatto che esistano coppie con lo stesso colore sembra essere associato maggiormente alla classe negativa. Il modello sembra quindi riuscre a combinare colore, forma, posizione e relazioni tra oggetti.

#### Task 12 

![interp](results/interpretable_models/task_12_selected_tree.png)

Nel task 12 il Decision Tree utilizza innanzitutto count_color per separare le immagini senza oggetti del colore richiesto da quelle che ne contengono. Nel ramo positivo, count_color_blue diventa particolarmente importante: valori superiori a 0.5 portano verso la classificazione positiva. Successivamente vengono utilizzate std_x_all e mean_x_circle per distinguere ulteriormente i casi. L'albero mostra quindi una struttura gerarchica in cui prima vengono considerate le informazioni sul colore e successivamente alcune caratteristiche relative alle posizioni degli oggetti.

#### Task 14

![interp](results/interpretable_models/task_14_selected_tree.png)

Nel task 14 la prima suddivisione avviene sulla posizione media dei cerchi (mean_x_circle), mostrando che la posizione degli oggetti è importante per questo task. Nei rami successivi compaiono pair_same_shape, count_shape_circle, mean_x_triangle e std_y_all. In particolare, pair_same_shape permette di separare alcuni casi positivi, mentre le feature spaziali vengono utilizzate per raffinare ulteriormente la decisione. 

#### Task 16

![interp](results/interpretable_models/task_16_selected_tree.png)

Il task 16 è interessante perché la prima suddivisione del tree avviene direttamente su pair_same_color. Quando questa relazione è presente, il ramo porta direttamente alla classe negativa; quando invece non è presente, il modello considera pair_same_size e successivamente count_shape_circle e mean_x_all. Questo mostra che per il riconoscimento della "car" le relazioni tra gli oggetti hanno un ruolo centrale: il modello utilizza infatti già alla radice una feature relazionale, per poi combinare informazioni su forma e posizione.
 
</details>

### Explainable Boosting Machine

Come ultimo modello interpretabile introduciamo l'Explainable Boosting Machine (EBM) che, a differenza della regressione logistica e dell'albero di decisione, non si limita a mostrare quali feature sono più importanti, ma permette di vedere come ogni feature contribuisce alla decisione al variare del suo valore; questa caratteristica è particolarmente utile perché possiamo visualizzare delle curve che mostrano il contributo di una singola feature alla predizione, per esempio, possiamo osservare come cambia il contributo della posizione media di una forma oppure del numero di oggetti di un certo colore o di una certa forma.

In questo modo possiamo andare oltre una semplice classifica delle feature e capire meglio quale relazione il modello ha effettivamente appreso; prima di scendere nel dettaglio con le curve, osserviamo gli iperparametri del modello e i suoi risultati.

<details>
<summary><strong>Iperparametri EBM</strong></summary>

```python
model = ExplainableBoostingClassifier(
        interactions=5,
        max_bins=32,
        max_rounds=500,
        learning_rate=0.03,
        random_state=SEED,
    )
```
Dove "interactions" setta il numero di interazioni che permettiamo di usare al modello, "max_bins" setta in quanti intervalli dividere ciascuna feature, "max_rounds" è il numero massimo di round di boosting eseguiti dal modello.

</details>

| task_id | task | val_f1 | accuracy | precision | recall | f1 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | triangle vs any | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 1 | square vs any | 0.962963 | 0.96 | 0.900000 | 1.000000 | 0.947368 |
| 2 | circle vs any | 0.933333 | 0.88 | 0.923077 | 0.857143 | 0.888889 |
| 3 | red vs any | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 4 | green vs any | 0.947368 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 5 | blue vs any | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 6 | cyan vs any | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 7 | magenta vs any | 0.909091 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 8 | yellow vs any | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 9 | red triangle on the right | 0.631579 | 0.96 | 1.000000 | 0.875000 | 0.933333 |
| 10 | red triangle on the right and arbitrary objects | 0.787879 | 0.80 | 0.833333 | 0.882353 | 0.857143 |
| 11 | red triangle on the right and at least one circle | 0.923077 | 0.88 | 0.842105 | 1.000000 | 0.914286 |
| 12 | red triangle on the right and at least one blue object | 0.823529 | 0.80 | 0.750000 | 1.000000 | 0.857143 |
| 13 | triangle and square, same color | 1.000000 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 14 | palindrome aba | 0.846154 | 0.84 | 0.900000 | 0.750000 | 0.818182 |
| 15 | house | 0.882353 | 0.64 | 0.466667 | 0.875000 | 0.608696 |
| 16 | car | 0.941176 | 0.84 | 0.875000 | 0.700000 | 0.777778 |
| 17 | tower | 0.965517 | 1.00 | 1.000000 | 1.000000 | 1.000000 |
| 18 | wagon | 0.965517 | 0.92 | 1.000000 | 0.875000 | 0.933333 |
| 19 | traffic light | 1.000000 | 0.84 | 0.400000 | 0.666667 | 0.500000 |

| representation | mean F1 | median F1 | mean accuracy |
|---|---:|---:|---:|
| perceptual | 0.869 | 0.932 | 0.882 |
| perceptual + generic relations | 0.869 | 0.944 | 0.890 |
| EBM | 0.902 | 0.940 | 0.918 |

L'Explainable Boosting Machine (EBM) raggiunge un F1 medio di 0.902 e un'accuracy media di 0.918 sui 20 task, risultando il modello con la performance media più alta tra quelle considerate, infatti con le sole feature percettive si ottiene un F1 medio di 0.869 e un'accuracy media di 0.882, mentre aggiungendo le feature relazionali generiche l'F1 medio rimane sostanzialmente invariato e l'accuracy sale a 0.890. L'EBM mostra quindi un miglioramento medio più evidente soprattutto per quanto riguarda l'F1.
La mediana dell'F1 mostra un comportamento interessante: le feature percettive raggiungono 0.932, le feature percettive più relazionali 0.944, mentre l'EBM raggiunge 0.940 e questo ci dice che il vantaggio dell'EBM non è uniforme su tutti i task, ma deriva soprattutto da miglioramenti ottenuti in alcuni problemi più difficili, questo perché se il valore medio dell'F1 aumenta senza che aumenti la mediana, significa che l'EBM riesce a migliorare soprattutto le prestazioni in quei casi in cui i modelli precedenti incontrano più difficoltà.

Nei task più semplici, dallo 0 all'8, l'EBM mantiene prestazioni elevate: raggiunge un F1 pari a 1.000 nei task 0, 3, 4, 5, 6, 7 e 8, mentre nei task 1 e 2 ottiene rispettivamente 0.947 e 0.889; in questi casi le feature percettive sono già sufficientemente informative e l'utilizzo di un modello più flessibile non produce un miglioramento significativo.
Il vantaggio dell'EBM diventa invece più evidente in alcuni task strutturati. Nel task 11 raggiunge un F1 di 0.914, nel task 13 arriva a 1.000, nel task 14 a 0.818, nel task 15 a 0.609, nel task 16 a 0.778 e nel task 17 raggiunge nuovamente 1.000. In particolare, rispetto alla combinazione di feature percettive e relazionali generiche, l'EBM migliora sensibilmente i task 14, 15, 16 e 17. Ciò significa che la flessibilità dell'EBM nell'apprendere i contributi di feature e interazioni è utile soprattutto quando il problema richiede una combinazione più complessa di informazioni.

Il comportamento non è però sempre migliore: in alcuni task l'EBM ottiene risultati leggermente inferiori rispetto alle feature percettive più relazionali: ad esempio nel task 2 l'F1 è 0.889 contro 0.923, nel task 9 è 0.933 contro 0.941, nel task 10 è 0.857 contro 0.875 e nel task 18 è 0.933 contro 0.968. Questo conferma che la maggiore flessibilità del modello non implica una performance migliore in ogni singolo task. Un caso interessante è il task 19 (traffic light), dove l'EBM raggiunge un F1 di 0.500, superiore sia al valore ottenuto con le sole feature percettive (0.267) sia a quello ottenuto con le feature percettive e relazionali (0.444). La prestazione rimane comunque relativamente bassa e deve essere interpretata con cautela, dato il numero ridotto di esempi positivi.

Dal punto di vista interpretativo, riportiamo sotto alcune curve dell'EBM relative al task 12 che mostrano i contributi di alcune feature (nella cartella results/interpretable_model_plots, sono presenti tutte le curve e le heatmap di interazione per alcuni task scelti 0-2-9-12-14-16).

<details>
<summary><strong>Task 12: red traingle on the right and at least one blue object and arbitrary objects</strong></summary>

![ebm](results/interpretable_models/task_12_ebm_count_color_red.png)
![ebm](results/interpretable_models/task_12_ebm_count_shape_triangle.png)
![ebm](results/interpretable_models/task_12_ebm_count_color_blue.png)
![ebm](results/interpretable_models/task_12_ebm_mean_x_triangle.png)

Interessante notare come, in questo task, la presenza del colore rosso sembri avere un contributo maggiore alla decisione rispetto alla presenza del triangolo.

</details>

---

## Notebook 3 - Encoder Visivi

In questo notebook passiamo a un confronto tra più moduli visivi, così da capire quale tipo di rappresentazione funziona meglio sui task KANDY, in vista del loro utilizzo all'interno del modello neuro-simbolico che utilizzeremo in seguito, ovvero il Concept Bottleneck Model.

In particolare andremo a utilizzare classificatore a n_task teste con uno dei seguenti backbone:

- una **CNN semplice**, addestrata da zero;
- una **ResNet-18 pretrained**;
- una **ResNet-18 da zero**;
- una **ViT-B/16 Tiny pretrained**;
- una **ViT-B/16 Tiny da zero**.

Il confronto viene fatto sui task KANDY e le prestazioni vengono misurate separatamente su validation e test: la scelta del modello da portare nella fase successiva non viene fatta guardando il test set, ma utilizzando la media dell'F1-score sul validation set, così da individuare l'architettura e il tipo di inizializzazione più adatti. Una volta scelta questa configurazione, analizziamo alcune decisioni con Saliency Maps e SHAP. L'obiettivo è capire che cosa sta utilizzando il modello per classificare le immagini e avere quindi un punto di partenza più motivato per il Concept Bottleneck Model.

Prima di mostrare i risultati dei modelli, diamo uno sguardo a ciò che li precede.

### Iperparametri e Data Augmentation

<details>
<summary><strong>Iperparametri</strong></summary>
    
```python
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
```

Tali iperparametri, ottimizzatore e scheduler sono stati utilizzati per tutti e cinque i modelli; è possibile dunque che tunando i parametri per il singolo modello si possa aver un miglioramento delle prestazioni. 
Per l'ottimizzazione è stato utilizzato AdamW con weight decay per limitare l'overfitting. Il learning rate viene poi adattato dinamicamente tramite ReduceLROnPlateau: quando la loss di validazione non migliora per 3 epoche consecutive, il learning rate viene dimezzato; in questo modo manteniamo un apprendimento più rapido nelle prime fasi e aggiornamenti più piccoli quando il modello raggiunge una fase di stabilizzazione. Inoltre, è stato utilizzato l'early stopping per fermarsi quando il modello non mostra più alcun accenno di miglioramento.

</details>


<details>
<summary><strong>Data Augmentation</strong></summary>
    
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

Le immagini vengono ridimensionate a 224×224 pixel e normalizzate utilizzando media e deviazione standard di ImageNet; durante il training viene inoltre applicata una piccola data augmentation tramite RandomAffine, introducendo leggere variazioni di rotazione, traslazione e scala; questo permette al modello di vedere versioni leggermente diverse delle stesse immagini e riduce la dipendenza dalla posizione esatta degli oggetti, migliorando la capacità di generalizzazione.
</details>

### I modelli

I modelli utilizzati come backbone, come accennato prima sono stati una CNN, una Resnet e un ViT; la prima architettura è stata scritta a mano nella forma che si trova riportata sotto, mentre gli altri modelli sono stati importati da alcune librerie, in particolare: la ResNet-18 è stata importata dai modelli di TorchVision, così come i pesi della sua versione pre-trainata su ImageNet, il ViT-Tiny così come i suoi pesi, invece, sono stati importati dalla libreria Timm; ho importato questo modello e non uno di quelli di torchvision perché ho lavorato senza gpu, e avevo dunque bisogno di un architettura più leggera (ViT-Tiny ha 5-6 milioni di parametri mentre il ViT-B16 di Torch ne ha circa 86).

<details>
<summary><strong>CNN</strong></summary>
    
```python

class SimpleCNN(nn.Module):
    
    def __init__(self, feature_dim=256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.output_dim = feature_dim

    def forward(self, x):
        x = self.features(x)
        return self.projection(x)

```
</details>

Una volta instaziato, il backbone viene utilizzato come encoder visivo, vengono quindi aggiunte n_task teste di classificazione. 

### Loss e metriche

Per ogni immagine viene utilizzato il logit prodotto dalla testa associata al relativo task_id: il logit viene estratto, gli viene applicata la sigmoide per trasformarlo in una probabilità tra 0 e 1 e, tramite una soglia, si decide se il risultato è più vicino alla classe negativa o alla classe positiva (nel nostro caso usiamo semplicemente "se x>=0.5 allora l'esempio è positivo").

Inoltre, abbiamo visto che i task possono avere un numero diverso di esempi positivi e negativi, per tenere conto di questo sbilanciamento usiamo un pos_weight specifico per ogni task, calcolato come:

$$
\text{pos\_weight} = \frac{N_{\text{negativi}}}{N_{\text{positivi}}}
$$

Questo peso viene applicato alla parte della binary cross-entropy relativa alla classe positiva: quando i positivi sono pochi, le loro predizioni sbagliate incidono maggiormente sulla loss; in questo modo evitiamo che un task con pochi esempi positivi venga ottimizzato solo sulla classe negativa.

Come metriche usiamo le stesse utilizzate in precedenza e cioè:

- Accuracy
- Precision
- Recall
- F1

Esse vengono calcolate separatamente per ciascuno dei 20 task e infine viene calcolata la loro media, così ogni task contribuisce allo stesso modo alla valutazione complessiva.

### Confronto tra Encoders

![encoders](results/visual_models/validation_model_comparison.png)

Il confronto mostra una differenza abbastanza netta tra i modelli pretrained e quelli addestrati da zero: la ResNet-18 pretrained ottiene il risultato migliore in validation, seguita dalla ViT-Tiny pretrained; i modelli scratch, al contrario, raggiungono valori più bassi, con ResNet-18 e CNN, mentre la ViT-Tiny scratch emerge come il modello che viene maggiormente penalizzato dal numero limitato di esempi. Tali risultati indicano che, essendo appunto il numero di campioni limitato, l'inizializzazione pretrained aiuta a costruire rapidamente una buona rappresentazione visiva.

| variant | val_accuracy | val_precision | val_recall | val_f1 | best_epoch | training_time_seconds |
|---|---:|---:|---:|---:|---:|---:|
| resnet18_pretrained | 0.912 | 0.907988 | 0.942004 | 0.922913 | 27 | 2829.805990 |
| vit_tiny_pretrained | 0.750 | 0.765402 | 0.740898 | 0.739335 | 40 | 2286.768455 |
| resnet18_scratch | 0.750 | 0.707219 | 0.769846 | 0.721676 | 28 | 2890.512607 |
| cnn_scratch | 0.588 | 0.542044 | 0.797690 | 0.611612 | 8 | 578.585122 |
| vit_tiny_scratch | 0.498 | 0.378000 | 0.800000 | 0.497657 | 1 | 511.743909 |


<details>
<summary><strong>Training log — CNN, ResNet e ViT</strong></summary>

```text
================================================================================
EXPERIMENT: cnn_scratch
================================================================================
Epoch 01/40 | train loss=0.8261 F1=0.268 | val loss=0.8621 F1=0.265
Epoch 02/40 | train loss=0.8248 F1=0.311 | val loss=0.8617 F1=0.265
Epoch 03/40 | train loss=0.8254 F1=0.327 | val loss=0.8615 F1=0.350
Epoch 04/40 | train loss=0.8240 F1=0.348 | val loss=0.8595 F1=0.326
Epoch 05/40 | train loss=0.8232 F1=0.345 | val loss=0.8559 F1=0.325
Epoch 06/40 | train loss=0.8102 F1=0.391 | val loss=0.8391 F1=0.411
Epoch 07/40 | train loss=0.7839 F1=0.475 | val loss=0.8054 F1=0.524
Epoch 08/40 | train loss=0.7558 F1=0.555 | val loss=0.7787 F1=0.612
Epoch 09/40 | train loss=0.7265 F1=0.554 | val loss=0.7643 F1=0.571
Epoch 10/40 | train loss=0.7099 F1=0.557 | val loss=0.7363 F1=0.577
Epoch 11/40 | train loss=0.7096 F1=0.560 | val loss=0.7398 F1=0.518
Epoch 12/40 | train loss=0.7053 F1=0.559 | val loss=0.7283 F1=0.585
Epoch 13/40 | train loss=0.6830 F1=0.588 | val loss=0.7226 F1=0.603
Epoch 14/40 | train loss=0.6808 F1=0.575 | val loss=0.7197 F1=0.557
Epoch 15/40 | train loss=0.6710 F1=0.584 | val loss=0.7120 F1=0.588
Epoch 16/40 | train loss=0.6618 F1=0.623 | val loss=0.7000 F1=0.584
Early stopping.

================================================================================
EXPERIMENT: resnet18_pretrained
================================================================================
Epoch 01/40 | train loss=0.7473 F1=0.549 | val loss=0.6651 F1=0.645
Epoch 02/40 | train loss=0.5788 F1=0.661 | val loss=0.5508 F1=0.696
Epoch 03/40 | train loss=0.4900 F1=0.726 | val loss=0.6176 F1=0.632
Epoch 04/40 | train loss=0.4381 F1=0.759 | val loss=0.5344 F1=0.752
Epoch 05/40 | train loss=0.4679 F1=0.758 | val loss=0.4823 F1=0.759
Epoch 06/40 | train loss=0.4605 F1=0.753 | val loss=0.6080 F1=0.718
Epoch 07/40 | train loss=0.4038 F1=0.787 | val loss=0.4165 F1=0.825
Epoch 08/40 | train loss=0.3530 F1=0.813 | val loss=0.4102 F1=0.820
Epoch 09/40 | train loss=0.4478 F1=0.783 | val loss=0.3695 F1=0.816
Epoch 10/40 | train loss=0.3223 F1=0.848 | val loss=0.3506 F1=0.811
Epoch 11/40 | train loss=0.3157 F1=0.853 | val loss=0.3503 F1=0.820
Epoch 12/40 | train loss=0.3444 F1=0.823 | val loss=0.4783 F1=0.770
Epoch 13/40 | train loss=0.2653 F1=0.877 | val loss=0.3226 F1=0.854
Epoch 14/40 | train loss=0.2511 F1=0.896 | val loss=0.3895 F1=0.833
Epoch 15/40 | train loss=0.1883 F1=0.914 | val loss=0.2793 F1=0.877
Epoch 16/40 | train loss=0.1885 F1=0.927 | val loss=0.3021 F1=0.897
Epoch 17/40 | train loss=0.1569 F1=0.936 | val loss=0.2633 F1=0.891
Epoch 18/40 | train loss=0.1564 F1=0.949 | val loss=0.3697 F1=0.838
Epoch 19/40 | train loss=0.1598 F1=0.933 | val loss=0.3839 F1=0.851
Epoch 20/40 | train loss=0.1566 F1=0.927 | val loss=0.3172 F1=0.905
Epoch 21/40 | train loss=0.1073 F1=0.964 | val loss=0.2630 F1=0.888
Epoch 22/40 | train loss=0.0902 F1=0.958 | val loss=0.3111 F1=0.912
Epoch 23/40 | train loss=0.1352 F1=0.935 | val loss=0.3030 F1=0.885
Epoch 24/40 | train loss=0.1138 F1=0.950 | val loss=0.2690 F1=0.892
Epoch 25/40 | train loss=0.2252 F1=0.899 | val loss=0.3595 F1=0.859
Epoch 26/40 | train loss=0.1460 F1=0.937 | val loss=0.2218 F1=0.916
Epoch 27/40 | train loss=0.0927 F1=0.965 | val loss=0.2245 F1=0.923
Epoch 28/40 | train loss=0.0705 F1=0.972 | val loss=0.2288 F1=0.915
Epoch 29/40 | train loss=0.0616 F1=0.981 | val loss=0.2337 F1=0.913
Epoch 30/40 | train loss=0.0654 F1=0.972 | val loss=0.2378 F1=0.916
Epoch 31/40 | train loss=0.0462 F1=0.987 | val loss=0.2200 F1=0.923
Epoch 32/40 | train loss=0.0406 F1=0.989 | val loss=0.2058 F1=0.913
Epoch 33/40 | train loss=0.0387 F1=0.991 | val loss=0.2260 F1=0.922
Epoch 34/40 | train loss=0.0294 F1=0.994 | val loss=0.2179 F1=0.920
Epoch 35/40 | train loss=0.0339 F1=0.996 | val loss=0.2162 F1=0.911
Early stopping.

================================================================================
EXPERIMENT: resnet18_scratch
================================================================================
Epoch 01/40 | train loss=0.9523 F1=0.389 | val loss=0.9136 F1=0.345
Epoch 02/40 | train loss=0.8822 F1=0.410 | val loss=0.8910 F1=0.460
Epoch 03/40 | train loss=0.7938 F1=0.500 | val loss=0.9602 F1=0.387
Epoch 04/40 | train loss=0.7625 F1=0.543 | val loss=0.7353 F1=0.584
Epoch 05/40 | train loss=0.7407 F1=0.528 | val loss=0.7416 F1=0.592
Epoch 06/40 | train loss=0.7034 F1=0.583 | val loss=0.7274 F1=0.574
Epoch 07/40 | train loss=0.6928 F1=0.587 | val loss=0.6985 F1=0.592
Epoch 08/40 | train loss=0.6609 F1=0.583 | val loss=0.7075 F1=0.612
Epoch 09/40 | train loss=0.6470 F1=0.629 | val loss=0.8157 F1=0.633
Epoch 10/40 | train loss=0.6452 F1=0.631 | val loss=0.6613 F1=0.617
Epoch 11/40 | train loss=0.6313 F1=0.622 | val loss=0.6787 F1=0.591
Epoch 12/40 | train loss=0.6345 F1=0.599 | val loss=0.7130 F1=0.633
Epoch 13/40 | train loss=0.5954 F1=0.658 | val loss=0.6381 F1=0.628
Epoch 14/40 | train loss=0.5701 F1=0.672 | val loss=0.7075 F1=0.679
Epoch 15/40 | train loss=0.6105 F1=0.652 | val loss=0.5935 F1=0.667
Epoch 16/40 | train loss=0.5446 F1=0.687 | val loss=0.7061 F1=0.639
Epoch 17/40 | train loss=0.5835 F1=0.660 | val loss=0.6302 F1=0.653
Epoch 18/40 | train loss=0.5432 F1=0.699 | val loss=0.6648 F1=0.664
Epoch 19/40 | train loss=0.5152 F1=0.701 | val loss=1.0045 F1=0.526
Epoch 20/40 | train loss=0.4695 F1=0.741 | val loss=0.5921 F1=0.700
Epoch 21/40 | train loss=0.4470 F1=0.765 | val loss=0.6152 F1=0.668
Epoch 22/40 | train loss=0.4547 F1=0.764 | val loss=0.5753 F1=0.693
Epoch 23/40 | train loss=0.4611 F1=0.725 | val loss=1.0918 F1=0.540
Epoch 24/40 | train loss=0.4451 F1=0.756 | val loss=2.1071 F1=0.361
Epoch 25/40 | train loss=0.4402 F1=0.756 | val loss=0.5425 F1=0.709
Epoch 26/40 | train loss=0.4409 F1=0.765 | val loss=0.5386 F1=0.718
Epoch 27/40 | train loss=0.4110 F1=0.777 | val loss=0.5211 F1=0.711
Epoch 28/40 | train loss=0.3909 F1=0.818 | val loss=0.5407 F1=0.722
Epoch 29/40 | train loss=0.3783 F1=0.822 | val loss=0.5696 F1=0.684
Epoch 30/40 | train loss=0.3784 F1=0.795 | val loss=0.5519 F1=0.704
Epoch 31/40 | train loss=0.3940 F1=0.800 | val loss=0.5662 F1=0.716
Epoch 32/40 | train loss=0.3782 F1=0.804 | val loss=0.5466 F1=0.719
Epoch 33/40 | train loss=0.3424 F1=0.827 | val loss=1.1658 F1=0.493
Epoch 34/40 | train loss=0.3529 F1=0.814 | val loss=0.8261 F1=0.624
Epoch 35/40 | train loss=0.3348 F1=0.835 | val loss=0.5525 F1=0.714
Epoch 36/40 | train loss=0.3205 F1=0.846 | val loss=0.5588 F1=0.712
Early stopping.

================================================================================
EXPERIMENT: vit_tiny_pretrained
================================================================================
Epoch 01/40 | train loss=1.2206 F1=0.429 | val loss=1.0632 F1=0.275
Epoch 02/40 | train loss=0.9805 F1=0.419 | val loss=0.9742 F1=0.478
Epoch 03/40 | train loss=0.8296 F1=0.529 | val loss=0.8608 F1=0.402
Epoch 04/40 | train loss=0.8487 F1=0.476 | val loss=0.8853 F1=0.393
Epoch 05/40 | train loss=0.8507 F1=0.376 | val loss=0.8172 F1=0.458
Epoch 06/40 | train loss=0.7586 F1=0.528 | val loss=0.8015 F1=0.444
Epoch 07/40 | train loss=0.7314 F1=0.524 | val loss=0.7018 F1=0.542
Epoch 08/40 | train loss=0.6177 F1=0.612 | val loss=0.6539 F1=0.588
Epoch 09/40 | train loss=0.6087 F1=0.626 | val loss=0.6008 F1=0.633
Epoch 10/40 | train loss=0.5399 F1=0.663 | val loss=0.5791 F1=0.624
Epoch 11/40 | train loss=0.5332 F1=0.676 | val loss=0.6026 F1=0.679
Epoch 12/40 | train loss=0.5045 F1=0.697 | val loss=0.5341 F1=0.633
Epoch 13/40 | train loss=0.4770 F1=0.701 | val loss=0.5869 F1=0.655
Epoch 14/40 | train loss=0.4574 F1=0.750 | val loss=0.5480 F1=0.678
Epoch 15/40 | train loss=0.4431 F1=0.740 | val loss=0.5614 F1=0.628
Epoch 16/40 | train loss=0.4772 F1=0.715 | val loss=0.5700 F1=0.598
Epoch 17/40 | train loss=0.4107 F1=0.752 | val loss=0.5363 F1=0.694
Epoch 18/40 | train loss=0.3897 F1=0.772 | val loss=0.5142 F1=0.717
Epoch 19/40 | train loss=0.3716 F1=0.792 | val loss=0.5022 F1=0.703
Epoch 20/40 | train loss=0.3728 F1=0.792 | val loss=0.5378 F1=0.695
Epoch 21/40 | train loss=0.3607 F1=0.785 | val loss=0.5527 F1=0.647
Epoch 22/40 | train loss=0.3547 F1=0.792 | val loss=0.5681 F1=0.706
Epoch 23/40 | train loss=0.3654 F1=0.788 | val loss=0.5371 F1=0.695
Epoch 24/40 | train loss=0.3324 F1=0.810 | val loss=0.5373 F1=0.684
Epoch 25/40 | train loss=0.3090 F1=0.835 | val loss=0.5433 F1=0.720
Epoch 26/40 | train loss=0.3037 F1=0.827 | val loss=0.5524 F1=0.738
Epoch 27/40 | train loss=0.2974 F1=0.841 | val loss=0.5443 F1=0.712
Epoch 28/40 | train loss=0.2631 F1=0.854 | val loss=0.5577 F1=0.711
Epoch 29/40 | train loss=0.2665 F1=0.864 | val loss=0.5832 F1=0.723
Epoch 30/40 | train loss=0.2637 F1=0.861 | val loss=0.6067 F1=0.722
Epoch 31/40 | train loss=0.2474 F1=0.866 | val loss=0.6189 F1=0.710
Epoch 32/40 | train loss=0.2345 F1=0.880 | val loss=0.6306 F1=0.719
Epoch 33/40 | train loss=0.2493 F1=0.869 | val loss=0.6512 F1=0.731
Epoch 34/40 | train loss=0.2201 F1=0.892 | val loss=0.6630 F1=0.738
Epoch 35/40 | train loss=0.2204 F1=0.886 | val loss=0.6833 F1=0.719
Epoch 36/40 | train loss=0.2375 F1=0.878 | val loss=0.6711 F1=0.729
Epoch 37/40 | train loss=0.2132 F1=0.887 | val loss=0.6814 F1=0.735
Epoch 38/40 | train loss=0.2069 F1=0.903 | val loss=0.6823 F1=0.731
Epoch 39/40 | train loss=0.2038 F1=0.890 | val loss=0.6867 F1=0.738
Epoch 40/40 | train loss=0.1911 F1=0.906 | val loss=0.6917 F1=0.739

================================================================================
EXPERIMENT: vit_tiny_scratch
================================================================================
Epoch 01/40 | train loss=0.9815 F1=0.362 | val loss=0.9024 F1=0.498
Epoch 02/40 | train loss=0.8890 F1=0.399 | val loss=0.8545 F1=0.385
Epoch 03/40 | train loss=0.8788 F1=0.363 | val loss=0.8547 F1=0.413
Epoch 04/40 | train loss=0.8642 F1=0.361 | val loss=0.8760 F1=0.344
Epoch 05/40 | train loss=0.8665 F1=0.367 | val loss=0.8715 F1=0.345
Epoch 06/40 | train loss=0.8666 F1=0.374 | val loss=0.8698 F1=0.265
Epoch 07/40 | train loss=0.8457 F1=0.281 | val loss=0.8652 F1=0.245
Epoch 08/40 | train loss=0.8415 F1=0.342 | val loss=0.8685 F1=0.308
Epoch 09/40 | train loss=0.8463 F1=0.324 | val loss=0.8603 F1=0.361
Early stopping.
Tempo totale: 151.6 minuti
```

</details>
```

Una volta selezionata la ResNet-18 come modello più performante sono andato a testarla sullo split di test e ad analizzarla tramite alcune tecniche basate sul gradiente.

### ResNet su Test Set

![encoders](results/visual_models/resnet18_pretrained_test_f1_by_task.png)

Sul test la ResNet-18 pretrained mantiene una buona performance generale, mostrando inoltre un calo contenuto rispetto alla validation: il modello mostra quindi una buona generalizzazione. I task basati su attributi semplici (0-8) risultano molto facili da riconoscere, e anche su molti task su proprietà posizionali il modello ottiene una buona prestazione. Le difficoltà aumentano invece in quei task che richiedono di combinare più proprietà o riconoscere strutture più complesse, in particolare: "palindrome aba" risulta essere il task più critico assieme a "triangle and square, same color" e "house".

| task_id | task | accuracy | precision | recall | f1 | n_samples |
|---:|---|---:|---:|---:|---:|---:|
| 0 | triangle vs any | 0.96 | 1.000000 | 0.857143 | 0.923077 | 25 |
| 1 | square vs any | 0.96 | 1.000000 | 0.888889 | 0.941176 | 25 |
| 2 | circle vs any | 0.92 | 1.000000 | 0.857143 | 0.923077 | 25 |
| 3 | red vs any | 1.00 | 1.000000 | 1.000000 | 1.000000 | 25 |
| 4 | green vs any | 0.92 | 0.777778 | 1.000000 | 0.875000 | 25 |
| 5 | blue vs any | 1.00 | 1.000000 | 1.000000 | 1.000000 | 25 |
| 6 | cyan vs any | 0.92 | 0.777778 | 1.000000 | 0.875000 | 25 |
| 7 | magenta vs any | 1.00 | 1.000000 | 1.000000 | 1.000000 | 25 |
| 8 | yellow vs any | 1.00 | 1.000000 | 1.000000 | 1.000000 | 25 |
| 9 | red triangle on the right | 1.00 | 1.000000 | 1.000000 | 1.000000 | 25 |
| 10 | red triangle on the right and arbitrary objects | 0.96 | 1.000000 | 0.941176 | 0.969697 | 25 |
| 11 | red triangle on the right and at least one circle | 0.88 | 0.882353 | 0.937500 | 0.909091 | 25 |
| 12 | red triangle on the right and at least one blue object | 0.84 | 0.823529 | 0.933333 | 0.875000 | 25 |
| 13 | triangle and square, same color | 0.68 | 0.846154 | 0.647059 | 0.733333 | 25 |
| 14 | palindrome aba | 0.60 | 0.583333 | 0.583333 | 0.583333 | 25 |
| 15 | house | 0.76 | 0.571429 | 1.000000 | 0.727273 | 25 |
| 16 | car | 0.84 | 0.800000 | 0.800000 | 0.800000 | 25 |
| 17 | tower | 0.92 | 0.800000 | 1.000000 | 0.888889 | 25 |
| 18 | wagon | 0.92 | 0.888889 | 1.000000 | 0.941176 | 25 |
| 19 | traffic light | 0.96 | 0.750000 | 1.000000 | 0.857143 | 25 |

Il caso "house" è interessante perché il recall è alto ma la precisione è più bassa: il modello tende quindi a riconoscere molte delle immagini positive, ma produce anche diversi falsi positivi. Nel complesso, i risultati confermano che il modello gestisce bene gli attributi visivi semplici, mentre le configurazioni più strutturate rimangono la parte più difficile.

<details>
<summary><strong>Alcuni Esempi di Predizione</strong></summary>

![encoders](results/visual_models/esempio1.png)
Qui il modello sembra trovarsi in difficoltà a percepire il triangolo.

![encoders](results/visual_models/esempio2.png)
Qui mostro un caso in cui il modello non sbaglia mai.

![encoders](results/visual_models/esempio3.png)
Qui il modello sembra avere incertezza sul colore in quegli esempi che hanno le forme geometriche richieste.

![encoders](results/visual_models/esempio4.png)
Qui il modello dimostra non aver compreso appieno concetti più astratti come "palindromo".

![encoders](results/visual_models/esempio5.png)
Qui il modello sembra non riuscire a vedere capire che rapporto ci deve essere tra le dimensioni degli oggetti presenti.

![encoders](results/visual_models/esempio6.png)
Qui sembra essere messo in diffcoltà dalle variazioni di colore, come nel notebook 2 il sistema percettivo si ritrovava in difficoltà quando impostavamo una soglia di distanza tra colori troppo bassa.

</details>

### Mappe di Salienza

Per quanto riguarda le mappe di salienza ho utilizzato principalmete due tecniche ovvero Vanilla Gradient e GradCam, ho poi utilizzato un'altra tecnica sempre basata sul gradiente ma che utilizza la filosofia di SHAP, andando dunque a calcolare il contributo del singolo pixel a partire dalla variazione che esso apporta sui pixel di background; background che viene creato a partire da alcuni esempi di riferimento.

#### Vanilla Gradient

Partiamo dunque da alcuni esempi su Vanilla Gradient, in questo caso sono andato a raccogliere un esempio per ciascun tipo di esempio TP-TN e, quando disponibili FP-FN su alcuni task scelti (gli stessi task visualizzati nei task sopra). Riporto qui solo alcuni plot selezionati, i plot completi sono disponibili in "results/visual_models/xai/saliency".

![encoders](results/visual_models/xai/saliency/task_00_FN.png)
![encoders](results/visual_models/xai/saliency/task_09_TN.png)
![encoders](results/visual_models/xai/saliency/task_13_FN.png)
![encoders](results/visual_models/xai/saliency/task_14_TP.png)

Le mappe mostrano che il modello tende a concentrarsi sugli oggetti rilevanti dell'immagine anche in quei casi in cui la predizione è errata, segno che il modello riesce spesso a individuare le parti più importanti dell'immagine; nei task più semplici la salienza è molto localizzata, mentre nei task strutturati si distribuisce su più oggetti, lungo la loro configurazionee a volte anche in altre aree dell'immagine: nei casi FP e FN questo comportamento evidenzia un limite, e cioè il modello riconosce spesso gli elementi visivi corretti, ma fatica a distinguere la relazione o la combinazione precisa richiesta dalla task.

Nelle mappe si nota inoltre una maggiore intensità dei gradienti lungo i contorni degli oggetti, mentre la parte interna rimane relativamente poco attiva; questo comportamento è coerente con la natura di Vanilla Gradient: la misura evidenzia i pixel per i quali una piccola perturbazione modifica maggiormente l'output e siccome gli oggetti in esami hanno regioni interne quasi uniformi e bordi molto netti, le variazioni lungo il contorno risultano particolarmente rilevanti.


#### Grad-CAM

![encoders](results/visual_models/xai/gradcam_task_00_FN.png)
![encoders](results/visual_models/xai/gradcam_task_09_TN.png)
![encoders](results/visual_models/xai/gradcam_task_13_FN.png)
![encoders](results/visual_models/xai/gradcam_task_14_TP.png)

Sugli stessi esempi, visualizziamo adesso le mappe ottenute con Grad-CAM le quali, per via del loro funzionamento basato sulle feature più profonde del modello, risultano essere meno precise ma anche meno rumorose. Tra i risultati più interessanti:
- Nel task 9 “red triangle on the right”, la zona più rilevante è quella che si trova nella parte destra dell'immagine.
- Nel task 14 “palindrome aba”, in particolare nell'esempio positivo, la Grad-CAM mostra invece una maggiore attivazione nella regione dell'oggetto centrale rispetto agli elementi laterali (come accadeva anche in Vanilla Gradient). Questo risultato è interessante perché, data la struttura del task, ci si aspetterebbe che la relazione tra i due elementi laterali fosse particolarmente rilevante. La mappa potrebbe quindi indicare che il modello utilizza una rappresentazione diversa da quella intuitivamente attesa. Tuttavia, Grad-CAM non permette di concludere direttamente che gli elementi laterali vengano ignorati, né che questo comportamento sia la causa delle prestazioni non ottimali sul task; tali ipotesi richiederebbero un'analisi sistematica di più esempi.
- Nel task 0, diversamente da quanto osservato con Vanilla Gradient, il modello sembra avere maggiori difficoltà nell'individuare il triangolo. La mappa appare infatti piuttosto diffusa sull'immagine, quasi come se il modello "stesse ricercando l'oggetto" senza riuscire a trovarlo. Un'ipotesi è che Grad-CAM basandosi sulle rappresentazioni più profonde della rete, che quindi risultano più compresse e quindi meno precise, nel caso di un triangolo di piccole dimensioni, potrebbe avere più difficoltà a preservare e localizzare efficacemente l'informazione relativa alla sua presenza.

Anche qui ne riporto solo alcune, le altre si trovano nel percorso "results/visual_models/xai"

#### SHAP

![encoders](results/visual_models/shap/task_00_FN.png)
![encoders](results/visual_models/shap/task_09_TN.png)
![encoders](results/visual_models/shap/task_13_FN.png)
![encoders](results/visual_models/shap/task_14_TP.png)

Sugli stessi esempi considerati per Vanilla Gradient e GradCAM, riportiamo infine le mappe ottenute con SHAP che, rispetto alle due tecniche precedenti, risulta particolarmente interessante perché la mappa non mostra solamente quali regioni siano rilevanti, ma anche la direzione del contributo alla predizione: le aree con valori positivi e negativi permettono infatti di distinguere tra caratteristiche che spingono l'output verso una classe e caratteristiche che lo contrastano; questo è possibile grazie all'utilizzo del GradientExplainer, il cui funzionamento si basa sui gradienti del modello rispetto agli input e sull'integrazione di tali gradienti rispetto a una distribuzione di riferimento (background), ottenendo così un'approssimazione dei valori SHAP per modelli differenziabili.

Nel complesso, le mappe SHAP risultano abbastanza localizzate sugli oggetti presenti nell'immagine e, rispetto a Vanilla Gradient, mostrano una risposta meno concentrata esclusivamente sui bordi, evidenziando maggiormente il contributo degli oggetti nel loro insieme. Nei casi più semplici, come il task 0, l'attivazione è quasi interamente concentrata sull'unico oggetto presente e risulta quindi immediato individuare quale elemento stia contribuendo maggiormente alla decisione del modello. Anche il task 14, “palindrome aba” risulta interessante, infatti in questo caso la mappa SHAP evidenzia chiaramente i due oggetti laterali, mentre l'oggetto centrale presenta un contributo di segno opposto. Rispetto a quanto osservato con GradCAM e Vanilla Gradient, che sembravano concentrarsi soprattutto sull'elemento centrale, SHAP mette quindi in evidenza la relazione tra gli elementi ai lati, risultando più coerente con la struttura del task e suggerisce che almeno in questo esempio il modello abbia appreso una rappresentazione più vicina alla relazione che si vorrebbe verificare.

Per quello che riguarda il task 14, le due famiglie di metodi Gradient (Vanilla Gradient e Grad-CAM) e SHAP evidenziano aspetti differenti della decisione del modello. Grad-CAM mostra quali regioni delle rappresentazioni profonde risultano maggiormente legate allo score del target, mentre SHAP attribuisce alle caratteristiche dell'input un contributo positivo o negativo rispetto a una distribuzione di riferimento (il background). Per questo le due mappe non devono necessariamente coincidere: una regione può essere fortemente rappresentata nelle feature profonde senza costituire, in termini di attribuzione all'input, il principale elemento che determina l'output.

---

## Notebook 4 - Concept Bottleneck Model

In quest'ultimo notebook introduciamo un modello neurosimbolico il cui nome è Concept Bottleneck Model (CBM), nel nostro caso il modello sarà composto da una ResNet-18, il cui compito è quello di estrarre le feature visive dagli esempi di KANDY, utilizzare poi un concept layer per predirre i concetti semantici contenuti in essi e, come classicatore finale, costruiremo due varianti:

- nella prima il classificatore sarà costituito da un layer lineare
- nella seconda il classificatore sarà costituito da un piccolo MLP

Possiamo dunque esprimere il processo nel seguente modo:

```text
immagine -> ResNet-18 -> concept layer -> task head -> 20 task
```

Il layer dei concetti è costituito a partire dalle descrizioni simboliche contenute in "symbol"; è importante ricordare che quest'ultimo non viene mai utilizzato attivamente all'interno dell'architettura, tuttavia, viene comunque utilizzato durante il preprocessing per estrarre le label dei concetti che forniscono la supervisione al concept layer durante l'addestramento. Il vocabolario dei concetti così ottenuto contiene proprietà come forma, colore, dimensione, posizione nella sequenza e relazioni semplici tra gli oggetti.

Per quello che riguarda l'addestramento del modello, esso avviene utilizzando una loss combinata, costituita come una media pesata dalla loss sui concetti e dalla loss sulla predizione del task, in questo modo la rete viene allenata sia a costruire una rappresentazione intermedia interpretabile sia a utilizzare tale rappresentazione per la classificazione finale.

Partiamo dunque dall'osservare come è stato formato il vocabolario dei concetti.

### Vocabolario dei Concetti

Ora, nelle prime versioni del CBM avevo costruito un vocabolario di oltre 100 concetti, includendo molte combinazioni tra forma, colore, dimensione, posizione e strutture degli oggetti; questa scelta aumentava le performance del modello in quanto permetteva di descrivere in modo molto dettagliato le immagini, tuttavia, introduceva anche diversi problemi: molti concetti erano ridondanti o simili tra loro mentre altri risultavano poco frequenti, rendendo più difficile capire quali informazioni fossero realmente utilizzate dal modello e diminuendo dunque l'interpretabilità delle soluzioni. Ho quindi ridotto progressivamente il vocabolario, cercando di mantenere solo i concetti che hanno un significato semantico chiaro, che sono riutilizzabili tra più task e che aiutano a descrivere le regole di KANDY in modo relativamente compatto e leggibile; nonostante questo, vedremo che anche alla fine alcuni concetti risulteranno comunque ridondanti.

Il vocabolario finale contiene 37 concetti, organizzati in cinque gruppi:

- attributi di base: forma, colore e dimensione;
- proprietà dell'ultimo elemento e confronti tra primo e ultimo elemento della sequenza
- relazioni strutturali come stack e side_by_side
- strutture semplici, ad esempio un triangolo associato a un quadrato o due cerchi affiancati
- proprietà di sequenza, utilizzate direttamente per descrivere alcune task, come la presenza di un cerchio o di un oggetto blu prima dell'ultimo elemento.

Per fare ciò che abbiamo detto ho inizialmente definito i concetti manualmente. 

<details>
<summary><strong>Concetti</strong></summary>
    
```python

CONCEPT_NAMES = [

    "has_triangle",
    "has_square",
    "has_circle",
    "has_red",
    "has_green",
    "has_blue",
    "has_cyan",
    "has_magenta",
    "has_yellow",
    "has_small",
    "has_large",
    #attributi di base

    "last_is_triangle",
    "last_is_square",
    "last_is_circle",
    "last_is_red",
    "last_is_green",
    "last_is_blue",
    "last_is_cyan",
    "last_is_magenta",
    "last_is_yellow",
    "last_is_small",
    "last_is_large",
    #ultimo elemento

    "first_last_same_shape",
    "first_last_same_color",
    "first_last_same_size",
    "triangle_square_share_color",
    #confronti semplici

    "has_stack",
    "has_side_by_side",
    "stack_same_size",
    "side_by_side_same_size",
    #relazioni generali

    "triangle_square_stack",
    "two_circles_side_by_side",
    "two_or_three_squares_stack",
    "two_or_three_squares_side_by_side",
    "traffic_light_color_order",
     #struttura semplice

    "exists_circle_before_last",
    "exists_blue_before_last",
    #relazioni sulla sequenza
]

```
</details>

Dopodiché sono andato a estrarre automaticamente tali concetti dalle descrizioni simboliche degli oggetti.La procedura parte dai singoli oggetti presenti nelle descrizioni, estraendone forma, colore e dimensione, e individuando poi le principali relazioni tra gruppi di oggetti, come stack e side_by_side; sulle informazioni ottenute vengono poi costruiti concetti di diverso livello, tra cui la presenza di specifiche forme, colori o dimensioni, le proprietà del primo e dell'ultimo oggetto, il confronto tra primo e ultimo elemento, la presenza di particolari combinazioni di oggetti e alcune configurazioni strutturali, come stack di oggetti della stessa dimensione, triangolo e quadrato sovrapposti o la sequenza di colori tipica di un semaforo.

Per ogni immagine viene quindi generato un vettore di 37 concetti, con valore 1 quando il concetto è presente e 0 altrimenti; tali informazioni vengono poi aggiunte ai diversi split del dataset e verranno successivamente utilizzati come target del concept layer.


<details>
<summary><strong>Concetti</strong></summary>
    
```python

CONCEPT_NAMES = [

    "has_triangle",
    "has_square",
    "has_circle",
    "has_red",
    "has_green",
    "has_blue",
    "has_cyan",
    "has_magenta",
    "has_yellow",
    "has_small",
    "has_large",
    #attributi di base

    "last_is_triangle",
    "last_is_square",
    "last_is_circle",
    "last_is_red",
    "last_is_green",
    "last_is_blue",
    "last_is_cyan",
    "last_is_magenta",
    "last_is_yellow",
    "last_is_small",
    "last_is_large",
    #ultimo elemento

    "first_last_same_shape",
    "first_last_same_color",
    "first_last_same_size",
    "triangle_square_share_color",
    #confronti semplici

    "has_stack",
    "has_side_by_side",
    "stack_same_size",
    "side_by_side_same_size",
    #relazioni generali

    "triangle_square_stack",
    "two_circles_side_by_side",
    "two_or_three_squares_stack",
    "two_or_three_squares_side_by_side",
    "traffic_light_color_order",
     #struttura semplice

    "exists_circle_before_last",
    "exists_blue_before_last",
    #relazioni sulla sequenza
]

```
</details>


| feature | positive_rate_train |
|---|---:|
| has_large | 0.670 |
| has_small | 0.641 |
| last_is_large | 0.523 |
| has_square | 0.499 |
| has_triangle | 0.477 |
| last_is_small | 0.477 |
| has_circle | 0.472 |
| has_side_by_side | 0.359 |
| has_red | 0.350 |
| last_is_square | 0.347 |
| last_is_triangle | 0.331 |
| first_last_same_size | 0.328 |
| last_is_circle | 0.322 |
| has_green | 0.286 |
| has_blue | 0.282 |
| exists_circle_before_last | 0.274 |
| has_cyan | 0.267 |
| last_is_red | 0.259 |
| has_yellow | 0.256 |
| first_last_same_shape | 0.249 |
| has_magenta | 0.245 |
| exists_blue_before_last | 0.169 |
| first_last_same_color | 0.162 |
| last_is_green | 0.161 |
| has_stack | 0.156 |
| last_is_blue | 0.154 |
| last_is_cyan | 0.145 |
| last_is_yellow | 0.144 |
| last_is_magenta | 0.137 |
| side_by_side_same_size | 0.134 |
| stack_same_size | 0.091 |
| triangle_square_share_color | 0.090 |
| triangle_square_stack | 0.039 |
| two_or_three_squares_stack | 0.031 |
| two_circles_side_by_side | 0.031 |
| two_or_three_squares_side_by_side | 0.030 |
| traffic_light_color_order | 0.005 |

Una volta raccolte le percentuali di presenza dei concetti, vado a rimuovere quelle la cui percentuale è sotto una certa soglia, in questo caso ho scelto di rimuovere solo i concetti che sono presenti in meno dell'1% degli esempi: dunque non viene rimosso alcun concetto: anche "traffic_light_color_order", pur essendo molto raro, viene mantenuto nel vocabolario utilizzato dal modello.

Prima di mostrare i risultati del CBM, diamo uno sguardo a ciò che li precede.

### Parametri

Per quanto riguarda i parametri, ho utilizzato i seguenti:

<details>
<summary><strong>Parametri</strong></summary>
    
```python

SEED = 12345
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 1e-4
TASK_LOSS_WEIGHT = 0.25
EARLY_STOPPING_PATIENCE = 6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
    ),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

```

</details>

Gli unici parametri che cambiano sono gli iperparamtri della ResNet questo perché adesso stiamo cercando di ottimizzare la sua performance specifica. In particolare, quello che andiamo a fare è:

- aumentare la batch size da 8 a 16 per rendere più stabile l'ottimizzazione della loss combinata
- diminuire leggermente il weight decay a 10^-4 per applicare una regolarizzazione meno forte a una rete già vincolata dalla supervisione sui concetti.
- ridurre il numero di epoche a 20, questo perché il modello mostrava buone capacità già intorno a tale numero di epoche, rendendo meno necessario un training lungo.
- riduciamo a 6 per interrompere prima il training in caso di stagnazione della validation, limitando il rischio di continuare ad aggiornare inutilmente la rappresentazione appresa.

Per il resto anche il pre-processing sui dati resta lo stesso.

Passiamo dunque alla loss utilizzata.

### Loss & Addestramento

Durante l'addestramento andiamo a utilizzare una loss composta da due contributi: una BCEWithLogitsLoss sui 37 concetti e una binary cross-entropy sulla predizione del task: la concept loss utilizza un pos_weight specifico per ogni concetto, maggiore per i concetti più rari, mentre la task loss usa un peso diverso per ciascun task in funzione del rapporto tra esempi negativi e positivi. La loss finale è quindi:

[
\mathcal{L} = \mathcal{L}{concept} + 0.25 \cdot \mathcal{L}{task}
]

in modo da mantenere come obiettivo principale l'apprendimento di una rappresentazione interpretabile senza trascurare la classificazione dei task.

La valutazione viene effettuata separatamente per ciascuno dei 20 task: per ogni immagine viene considerato solamente il logit associato al task_id corretto e viene applicata una soglia corrispondente a una probabilità di 0.5 dopo la sigmoide. Per ogni task vengono quindi calcolati Accuracy e F1-score, mentre come metrica complessiva viene utilizzato il macro-F1, ottenuto facendo la media degli F1 dei 20 task. Il modello migliore viene selezionato sulla base del macro-F1 di validation e il training viene interrotto tramite early stopping quando la metrica non migliora più per il numero di epoche stabilito.

<details>
<summary><strong>Addestramento Head Lineare</strong></summary>
    
| epoch | train_loss | val_loss | val_macro_f1 |
|---:|---:|---:|---:|
| 1 | 0.902565 | 0.664226 | 0.428409 |
| 2 | 0.616467 | 0.554593 | 0.474202 |
| 3 | 0.532530 | 0.499412 | 0.501515 |
| 4 | 0.476296 | 0.455170 | 0.507493 |
| 5 | 0.428281 | 0.415461 | 0.507299 |
| 6 | 0.402443 | 0.401249 | 0.551052 |
| 7 | 0.377699 | 0.390545 | 0.564178 |
| 8 | 0.371107 | 0.424185 | 0.587851 |
| 9 | 0.358081 | 0.393837 | 0.589448 |
| 10 | 0.340145 | 0.357783 | 0.617481 |
| 11 | 0.321034 | 0.357708 | 0.626838 |
| 12 | 0.310565 | 0.352097 | 0.645328 |
| 13 | 0.297560 | 0.361883 | 0.683312 |
| 14 | 0.281513 | 0.354508 | 0.708546 |
| 15 | 0.278445 | 0.334332 | 0.720776 |
| 16 | 0.266776 | 0.342440 | 0.740910 |
| 17 | 0.261333 | 0.384363 | 0.764560 |
| 18 | 0.260032 | 0.358359 | 0.771963 |
| 19 | 0.243181 | 0.329985 | 0.814203 |
| 20 | 0.233764 | 0.326121 | 0.810617 |

</details>

<details>
<summary><strong>Addestramento Head MLP</strong></summary>
    
| epoch | train_loss | val_loss | val_macro_f1 |
|---:|---:|---:|---:|
| 1 | 0.8607 | 0.6791 | 0.2775 |
| 2 | 0.5956 | 0.5455 | 0.3919 |
| 3 | 0.5182 | 0.5141 | 0.4548 |
| 4 | 0.4619 | 0.4499 | 0.6070 |
| 5 | 0.4090 | 0.3833 | 0.7245 |
| 6 | 0.3897 | 0.4414 | 0.7594 |
| 7 | 0.3825 | 0.3639 | 0.8251 |
| 8 | 0.3412 | 0.3348 | 0.8553 |
| 9 | 0.3183 | 0.3383 | 0.8273 |
| 10 | 0.2871 | 0.3002 | 0.8802 |
| 11 | 0.2824 | 0.2898 | 0.8810 |
| 12 | 0.2598 | 0.2736 | 0.9083 |
| 13 | 0.2425 | 0.2580 | 0.8812 |
| 14 | 0.2326 | 0.2896 | 0.8864 |
| 15 | 0.2759 | 0.3149 | 0.9055 |
| 16 | 0.2571 | 0.2637 | 0.9120 |
| 17 | 0.2093 | 0.2365 | 0.9269 |
| 18 | 0.1982 | 0.2357 | 0.9204 |
| 19 | 0.1991 | 0.2477 | 0.9322 |
| 20 | 0.1866 | 0.2699 | 0.9286 |

</details>

### Risultati & Controlli Head Lineare

| task_id | task_name | accuracy | f1 |
|---:|---|---:|---:|
| 0 | triangle vs any | 0.88 | 0.823529 |
| 1 | square vs any | 0.92 | 0.900000 |
| 2 | circle vs any | 1.00 | 1.000000 |
| 3 | red vs any | 0.92 | 0.888889 |
| 4 | green vs any | 0.92 | 0.875000 |
| 5 | blue vs any | 1.00 | 1.000000 |
| 6 | cyan vs any | 0.76 | 0.500000 |
| 7 | magenta vs any | 0.84 | 0.777778 |
| 8 | yellow vs any | 1.00 | 1.000000 |
| 9 | red triangle on the right | 0.80 | 0.761905 |
| 10 | red triangle on the right and arbitrary objects | 0.92 | 0.944444 |
| 11 | red triangle on the right and at least one circle | 0.52 | 0.500000 |
| 12 | red triangle on the right and at least one blue object | 0.76 | 0.823529 |
| 13 | triangle and square, same color | 0.68 | 0.800000 |
| 14 | palindrome aba | 0.32 | 0.413793 |
| 15 | house | 0.52 | 0.500000 |
| 16 | car | 0.76 | 0.769231 |
| 17 | tower | 0.88 | 0.800000 |
| 18 | wagon | 1.00 | 1.000000 |
| 19 | traffic light | 0.76 | 0.500000 |

Osservando le prestazioni del CBM con testa lineare, esse risultano inferiori rispetto a quelle ottenute dalla ResNet-18 utilizzata direttamente per la classificazione dei task, questo è spiegabile considerando il concept bottleneck del CBM: mentre l'encoder visuale può utilizzare direttamente l'intera rappresentazione estratta dalla ResNet, il CBM deve prima comprimere l'informazione in un insieme limitato di 37 concetti interpretabili; questa scelta migliora la trasparenza del modello, ma può eliminare informazioni visive utili per alcuni task. Otteniamo infatti "Accuracy media: 0.808" e "F1 medio: 0.779".

Sono andato poi a eseguire una serie di controlli, relativi ai concetti utilizzati dal modello.


### Analisi dei concetti Head Lineare

#### Qualità del Bottleneck

Qui andiamo a misurare la qualità del bottleneck concettuale confrontando i concetti predetti dal modello con le relative label corrette presenti nel test set. Per ogni concetto, le probabilità predette vengono trasformate in valori binari usando una soglia di 0.5 e viene calcolato l'F1 tra predizioni e valori reali. Questo permette di valutare quanto accuratamente il bottleneck riesca a riconoscere ciascun concetto.

Viene riportato il positive_rate, cioè la percentuale di esempi del test set in cui il concetto è effettivamente presente, questo valore ci permette di interpretare meglio l'F1 dei concetti, soprattutto nel caso questi risultino essere poco frequenti. Infine, viene calcolato l'F1 medio sui 37 concetti, ottenendo una misura complessiva della qualità della rappresentazione concettuale appresa dal modello.

| concept | f1 | positive_rate |
|---|---:|---:|
| 18 | last_is_magenta | 1.000000 | 0.082 |
| 15 | last_is_green | 1.000000 | 0.072 |
| 20 | last_is_small | 0.996337 | 0.272 |
| 11 | last_is_triangle | 0.990385 | 0.206 |
| 27 | has_side_by_side | 0.988764 | 0.354 |
| 4 | has_green | 0.986111 | 0.292 |
| 5 | has_blue | 0.986111 | 0.290 |
| 14 | last_is_red | 0.984456 | 0.190 |
| 19 | last_is_yellow | 0.984127 | 0.062 |
| 12 | last_is_square | 0.983957 | 0.184 |
| 21 | last_is_large | 0.982332 | 0.278 |
| 13 | last_is_circle | 0.981366 | 0.160 |
| 10 | has_large | 0.979528 | 0.636 |
| 2 | has_circle | 0.979424 | 0.496 |
| 0 | has_triangle | 0.979339 | 0.484 |
| 8 | has_yellow | 0.978166 | 0.232 |
| 6 | has_cyan | 0.974729 | 0.270 |
| 35 | exists_circle_before_last | 0.972028 | 0.282 |
| 7 | has_magenta | 0.967509 | 0.284 |
| 9 | has_small | 0.963526 | 0.664 |
| 16 | last_is_blue | 0.960000 | 0.072 |
| 17 | last_is_cyan | 0.960000 | 0.072 |
| 26 | has_stack | 0.957576 | 0.158 |
| 1 | has_square | 0.931452 | 0.466 |
| 3 | has_red | 0.929348 | 0.342 |
| 22 | first_last_same_shape | 0.899390 | 0.690 |
| 33 | two_or_three_squares_side_by_side | 0.893617 | 0.042 |
| 36 | exists_blue_before_last | 0.886364 | 0.162 |
| 23 | first_last_same_color | 0.883072 | 0.606 |
| 24 | first_last_same_size | 0.834254 | 0.800 |
| 32 | two_or_three_squares_stack | 0.769231 | 0.020 |
| 29 | side_by_side_same_size | 0.754491 | 0.140 |
| 34 | traffic_light_color_order | 0.750000 | 0.006 |
| 31 | two_circles_side_by_side | 0.702703 | 0.026 |
| 28 | stack_same_size | 0.661017 | 0.084 |
| 30 | triangle_square_stack | 0.653846 | 0.034 |
| 25 | triangle_square_share_color | 0.452830 | 0.080 |

Possiamo notare che il bottleneck concettuale riesce a rappresentare bene soprattutto gli attributi visivi e le relazioni più semplici, mentre, le principali difficoltà si ritrovano nei concetti più complessi e composizionali, i quali richiedono di combinare più proprietà o considerare contemporaneamente più oggetti. Questo mette ancora in evidenza la difficoltà del modello su task più complessi.

#### Importanza sui Concetti

In questa analisi ho valutato l'influenza dei singoli concetti sulla predizione del modello. Per ogni concetto, il valore predetto viene impostato a zero per tutti gli esempi e si misura la variazione della probabilità associata al task corretto rispetto alla configurazione originale.

| concept | mean_score_drop | mean_abs_change |
|---|---:|---:|
| has_small | -0.002463 | 0.016721 |
| first_last_same_size | -0.003402 | 0.014694 |
| has_large | 0.003437 | 0.014585 |
| has_red | 0.000962 | 0.013541 |
| first_last_same_color | 0.002880 | 0.013050 |
| has_square | 0.003716 | 0.012704 |
| first_last_same_shape | 0.005242 | 0.011608 |
| has_triangle | -0.002826 | 0.011357 |
| has_circle | -0.000505 | 0.010253 |
| last_is_red | 0.007193 | 0.009953 |
| has_green | -0.005765 | 0.008977 |
| has_blue | -0.001332 | 0.008805 |
| exists_circle_before_last | 0.003314 | 0.008617 |
| has_magenta | -0.002073 | 0.008145 |
| has_side_by_side | -0.007192 | 0.007514 |
| last_is_small | 0.001822 | 0.007418 |
| has_cyan | -0.005162 | 0.007242 |
| last_is_square | 0.004555 | 0.006778 |
| last_is_triangle | 0.002117 | 0.006435 |
| last_is_large | 0.000351 | 0.006238 |
| exists_blue_before_last | 0.000919 | 0.006122 |
| has_yellow | -0.000716 | 0.005696 |
| triangle_square_share_color | -0.001092 | 0.004394 |
| last_is_circle | -0.000991 | 0.003832 |
| last_is_blue | -0.000688 | 0.003168 |
| side_by_side_same_size | -0.000503 | 0.003158 |
| has_stack | -0.002613 | 0.002657 |
| two_or_three_squares_side_by_side | 0.001227 | 0.002625 |
| last_is_cyan | -0.000246 | 0.002549 |
| stack_same_size | 0.001862 | 0.001953 |
| last_is_magenta | -0.001368 | 0.001790 |
| last_is_green | -0.001023 | 0.001558 |
| last_is_yellow | -0.000531 | 0.001316 |
| two_or_three_squares_stack | 0.000401 | 0.001092 |
| two_circles_side_by_side | 0.000783 | 0.000894 |
| triangle_square_stack | 0.000158 | 0.000504 |
| traffic_light_color_order | 0.000058 | 0.000112 |

<details>
<summary><strong>Analisi specifica per task</strong></summary>

#### Task 0 — triangle vs any

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 0 | triangle vs any | first_last_same_shape | 0.029789 | 0.029789 |
| 0 | triangle vs any | has_large | 0.018527 | -0.018527 |
| 0 | triangle vs any | first_last_same_color | 0.016564 | -0.016564 |
| 0 | triangle vs any | first_last_same_size | 0.016497 | 0.016497 |
| 0 | triangle vs any | has_circle | 0.013460 | -0.013460 |
| 0 | triangle vs any | has_triangle | 0.012326 | 0.012326 |
| 0 | triangle vs any | has_yellow | 0.008880 | -0.008880 |
| 0 | triangle vs any | has_small | 0.007411 | -0.007411 |
| 0 | triangle vs any | has_red | 0.006073 | -0.006073 |
| 0 | triangle vs any | has_square | 0.004453 | -0.004453 |

#### Task 9 — red triangle on the right

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 9 | red triangle on the right | last_is_triangle | 0.044381 | 0.044381 |
| 9 | red triangle on the right | has_large | 0.036903 | 0.036903 |
| 9 | red triangle on the right | has_red | 0.025677 | 0.025677 |
| 9 | red triangle on the right | last_is_red | 0.020726 | 0.020726 |
| 9 | red triangle on the right | has_green | 0.018474 | -0.018474 |
| 9 | red triangle on the right | first_last_same_size | 0.015583 | -0.015583 |
| 9 | red triangle on the right | has_small | 0.013892 | 0.013892 |
| 9 | red triangle on the right | has_square | 0.013795 | -0.013795 |
| 9 | red triangle on the right | has_side_by_side | 0.013644 | -0.013644 |
| 9 | red triangle on the right | has_blue | 0.011828 | -0.011828 |

#### Task 13 — triangle and square, same color

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 13 | triangle and square, same color | has_square | 0.047535 | 0.047535 |
| 13 | triangle and square, same color | has_red | 0.026415 | -0.026415 |
| 13 | triangle and square, same color | has_small | 0.025333 | -0.025333 |
| 13 | triangle and square, same color | has_triangle | 0.024947 | 0.024947 |
| 13 | triangle and square, same color | stack_same_size | 0.022398 | 0.022398 |
| 13 | triangle and square, same color | last_is_square | 0.021458 | 0.021458 |
| 13 | triangle and square, same color | side_by_side_same_size | 0.014933 | -0.014933 |
| 13 | triangle and square, same color | has_large | 0.013747 | 0.013747 |
| 13 | triangle and square, same color | first_last_same_size | 0.012812 | 0.012812 |
| 13 | triangle and square, same color | first_last_same_color | 0.012364 | 0.012364 |

#### Task 14 — palindrome aba

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 14 | palindrome aba | has_triangle | 0.028261 | -0.028261 |
| 14 | palindrome aba | last_is_red | 0.016902 | 0.016902 |
| 14 | palindrome aba | last_is_small | 0.015523 | -0.015523 |
| 14 | palindrome aba | has_small | 0.014848 | 0.014848 |
| 14 | palindrome aba | has_side_by_side | 0.014578 | -0.014578 |
| 14 | palindrome aba | first_last_same_size | 0.013941 | -0.013941 |
| 14 | palindrome aba | has_cyan | 0.013773 | -0.013773 |
| 14 | palindrome aba | last_is_circle | 0.013434 | 0.013434 |
| 14 | palindrome aba | exists_circle_before_last | 0.010780 | 0.010780 |
| 14 | palindrome aba | last_is_square | 0.010015 | 0.010015 |

#### Task 19 — traffic light

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 19 | traffic light | has_small | 0.032042 | 0.032042 |
| 19 | traffic light | exists_circle_before_last | 0.029555 | 0.029555 |
| 19 | traffic light | exists_blue_before_last | 0.026901 | -0.026901 |
| 19 | traffic light | first_last_same_shape | 0.026447 | -0.026447 |
| 19 | traffic light | has_blue | 0.025862 | -0.025862 |
| 19 | traffic light | has_large | 0.017891 | -0.017891 |
| 19 | traffic light | last_is_blue | 0.017767 | -0.017767 |
| 19 | traffic light | last_is_circle | 0.014652 | 0.014652 |
| 19 | traffic light | last_is_small | 0.014504 | 0.014504 |
| 19 | traffic light | last_is_large | 0.012187 | -0.012187 |

</details>


A livello globale, l'intervento su un singolo concetto produce variazioni generalmente contenute nella probabilità del task, con i concetti più influenti che raggiungono variazioni nell'ordine di circa 0.5–1.7%. Questo suggerisce che la decisione della task head non dipende generalmente da un singolo concetto, ma combina l'informazione proveniente da più concetti.

A livello locale, per task, l'analisi mostra invece che l'importanza dei concetti varia in funzione del compito. Ad esempio:

- nel task 13, risultano particolarmente influenti "has_square", "has_triangle" e "first_last_same_color"
- nel task 19, il concetto specifico "traffic_light_color_order" produce una variazione molto ridotta, mentre la predizione risulta maggiormente influenzata da concetti più generici.

Nel complesso, il bottleneck concettuale viene quindi effettivamente utilizzato dalla task head, ma la predizione appare distribuita su più concetti. L'importanza dei singoli concetti non è inoltre sempre perfettamente allineata alla struttura attesa dei task e, in alcuni casi, l'intervento su un concetto può persino migliorare lo score del task.

Successivamente valuto l'importanza dei singoli concetti sulla predizione finale della task head: calcolo la macro-F1 di riferimento utilizzando i concetti predetti dal modello, poi per ogni concetto, lo imposto a zero per tutti gli esempi e ricalcolo la macro-F1, così da misurare quanto la sua rimozione modifica la performance del modello. Infine, ordino i concetti in base alla variazione di macro-F1 e li rimuovo progressivamente, uno alla volta, per osservare come cambia la performance complessiva al crescere del numero di concetti rimossi.

![ablation](results/neuro_symbolic_concept_cbm/plots/cumulative_linear.png)

Notiamo che la diminuzione è evidente nelle prime fasi, indicando che i concetti classificati come più importanti contribuiscono in modo significativo alla decisione della task head; proseguendo poi con la rimozione la performance continua a calare anche se ci sono alcune oscillazioni, questo può essere dovuto a dipendenze tra le variabili rimosse e alla soglia utilizzata per decidere se la predizione è 0 o 1.

#### Robustezza & Concetti Rumorosi

Qui sono andato a valutare la robustezza del modello rispetto a errori nelle predizioni dei concetti. I concetti predetti vengono prima trasformati in valori binari (0/1) usando una soglia di 0.5 e, successivamente, una percentuale crescente di essi viene modificata casualmente, invertendo il valore (da 0 a 1 e da 1 a 0). Sono andato a considerate percentuali di rumore pari a 0%, 5%, 10%, 20% e 30%: il caso con 0% di rumore rappresenta il comportamento di riferimento, mentre gli altri casi permettono di valutare quanto le prestazioni degradino quando le informazioni concettuali diventano progressivamente meno affidabili.

| noise_rate | task_accuracy |
|---:|---:|
| 0.00 | 0.826 |
| 0.05 | 0.760 |
| 0.10 | 0.734 |
| 0.20 | 0.644 |
| 0.30 | 0.586 |

Qui il risultato mostra che la performance diminuisce progressivamente all'aumentare del rumore introdotto nei concetti predetti: infatti, già con una piccola quantità di rumore si osserva una riduzione dell'accuratezza, mentre aumentando ulteriormente il rumore la degradazione diventa più critica. Questo significa che la qualità dei concetti forniti alla task head è importante per la predizione finale e che gli errori nel bottleneck possono propagarsi alla fase di classificazione.

#### Decomposizioni Errori

Qui analizziamo, per ciascuno dei 20 task, quanto gli errori nella predizione dei concetti incidano sulla performance finale della task head. A questo scopo confrontiamo la performance ottenuta quando la task head riceve i concetti corretti del test set con quella ottenuta utilizzando i concetti predetti dal bottleneck.

La differenza tra le due accuracy viene utilizzata come misura della perdita di performance associata agli errori nella rappresentazione concettuale:

accuracy_drop = oracle_concept_accuracy - predicted_concept_accuracy

Un accuracy_drop elevato indica quindi che gli errori del bottleneck hanno un impatto maggiore sulla classificazione del task, mentre un valore ridotto indica che la task head riesce a mantenere prestazioni simili anche quando i concetti forniti non sono perfettamente corretti.

| task_id | task_name | oracle_concept_accuracy | predicted_concept_accuracy | accuracy_drop |
|---:|---|---:|---:|---:|
| 15 | house | 0.68 | 0.60 | 0.08 |
| 16 | car | 0.92 | 0.84 | 0.08 |
| 13 | triangle and square, same color | 0.76 | 0.68 | 0.08 |
| 0 | triangle vs any | 0.92 | 0.88 | 0.04 |
| 18 | wagon | 0.96 | 0.92 | 0.04 |
| 17 | tower | 0.96 | 0.92 | 0.04 |
| 3 | red vs any | 0.96 | 0.92 | 0.04 |
| 1 | square vs any | 0.96 | 0.96 | 0.00 |
| 2 | circle vs any | 1.00 | 1.00 | 0.00 |
| 4 | green vs any | 0.88 | 0.88 | 0.00 |
| 9 | red triangle on the right | 0.92 | 0.92 | 0.00 |
| 6 | cyan vs any | 0.84 | 0.84 | 0.00 |
| 8 | yellow vs any | 0.96 | 0.96 | 0.00 |
| 12 | red triangle on the right and at least one blue object | 0.88 | 0.88 | 0.00 |
| 11 | red triangle on the right and at least one circle | 0.88 | 0.88 | 0.00 |
| 5 | blue vs any | 0.92 | 0.96 | -0.04 |
| 19 | traffic light | 0.84 | 0.88 | -0.04 |
| 10 | red triangle on the right and arbitrary objects | 0.88 | 0.92 | -0.04 |
| 14 | palindrome aba | 0.20 | 0.28 | -0.08 |
| 7 | magenta vs any | 0.68 | 0.80 | -0.12 |

Osserviamo che per diversi task, la task head ottiene la stessa accuracy sia utilizzando i concetti corretti sia quelli predetti dal bottleneck, suggerendo che il modello sia accurato. Interessanti sono però i valori negativi di accuracy_drop: in quei task la performance con i concetti predetti è superiore a quella ottenuta con i concetti corretti, come accade per magenta vs any e palindrome aba. Questo potrebbe essere dovuto al fatto che le predizioni del bottleneck modificano alcuni esempi in modo da produrre casualmente una classificazione finale più favorevole.

#### Visualizzazioni

Un punto di forza di questo approccio è il poter visualizzare alcune metriche interessanti sui concetti, in particolare in questi grafici troviamo la probabilità assegnata dal bottleneck ai concetti più attivi per specifiche immagini e i pesi della task head lineare, che permettono di osservare rispettivamente quali concetti il modello ritiene più presenti in un esempio e quanto ciascun concetto contribuisce alla predizione dei diversi task.

![inter](results/neuro_symbolic_concept_cbm/plots/task_head_weights.png)
![inter](results/neuro_symbolic_concept_cbm/plots/tasks_concepts_test_example_0.png)

### Risultati & Controlli Head MLP

Sono andato poi a testare l'architettura cambiando la testa di classificazione e, al posto di un layer lineare, ho introdotto un MLP.

| task_id | task_name | accuracy | f1 |
|---:|---|---:|---:|
| 0 | triangle vs any | 0.96 | 0.933333 |
| 1 | square vs any | 0.88 | 0.842105 |
| 2 | circle vs any | 1.00 | 1.000000 |
| 3 | red vs any | 1.00 | 1.000000 |
| 4 | green vs any | 1.00 | 1.000000 |
| 5 | blue vs any | 1.00 | 1.000000 |
| 6 | cyan vs any | 0.72 | 0.666667 |
| 7 | magenta vs any | 1.00 | 1.000000 |
| 8 | yellow vs any | 1.00 | 1.000000 |
| 9 | red triangle on the right | 1.00 | 1.000000 |
| 10 | red triangle on the right and arbitrary objects | 0.96 | 0.971429 |
| 11 | red triangle on the right and at least one circle | 0.92 | 0.941176 |
| 12 | red triangle on the right and at least one blue object | 0.88 | 0.909091 |
| 13 | triangle and square, same color | 0.88 | 0.918919 |
| 14 | palindrome aba | 0.40 | 0.482759 |
| 15 | house | 0.72 | 0.666667 |
| 16 | car | 0.84 | 0.833333 |
| 17 | tower | 0.96 | 0.941176 |
| 18 | wagon | 0.96 | 0.969697 |
| 19 | traffic light | 0.96 | 0.857143 |

Rispetto alla task head lineare, la task head MLP raggiunge un'accuracy media di 0.902 e una F1 media di 0.897. Dai risultati per task si osserva una buona performance nella maggior parte dei compiti, con prestazioni particolarmente elevate nei task basati su attributi visivi semplici e nelle relazioni più dirette, mentre rimangono difficoltà nei task più astratti o che coinvolgono relazioni complesse tra oggetti. L'introduzione della non linearità porta quindi a un miglioramento marcato della classificazione finale, anche se su task complessi come "palindrome aba" la performance rimane bassa.

#### Qualità del Bottleneck

| concept | f1 | positive_rate |
|---|---:|---:|
| last_is_green | 1.000000 | 0.072 |
| traffic_light_color_order | 1.000000 | 0.006 |
| has_green | 0.996564 | 0.292 |
| last_is_red | 0.994709 | 0.190 |
| has_yellow | 0.991304 | 0.232 |
| has_side_by_side | 0.985994 | 0.354 |
| has_circle | 0.982036 | 0.496 |
| last_is_magenta | 0.976190 | 0.082 |
| has_square | 0.975824 | 0.466 |
| has_cyan | 0.973783 | 0.270 |
| has_red | 0.973134 | 0.342 |
| has_magenta | 0.972222 | 0.284 |
| last_is_triangle | 0.970000 | 0.206 |
| has_stack | 0.969325 | 0.158 |
| has_blue | 0.963455 | 0.290 |
| has_small | 0.958084 | 0.664 |
| has_triangle | 0.957082 | 0.484 |
| last_is_yellow | 0.953846 | 0.062 |
| exists_circle_before_last | 0.951724 | 0.282 |
| last_is_large | 0.950000 | 0.278 |
| last_is_cyan | 0.947368 | 0.072 |
| last_is_blue | 0.935065 | 0.072 |
| last_is_square | 0.933333 | 0.184 |
| last_is_circle | 0.930233 | 0.160 |
| last_is_small | 0.925795 | 0.272 |
| first_last_same_color | 0.897033 | 0.606 |
| first_last_same_shape | 0.887538 | 0.690 |
| has_large | 0.884211 | 0.636 |
| first_last_same_size | 0.852713 | 0.800 |
| exists_blue_before_last | 0.852632 | 0.162 |
| two_or_three_squares_side_by_side | 0.823529 | 0.042 |
| triangle_square_stack | 0.693878 | 0.034 |
| stack_same_size | 0.682927 | 0.084 |
| two_circles_side_by_side | 0.650000 | 0.026 |
| two_or_three_squares_stack | 0.625000 | 0.020 |
| side_by_side_same_size | 0.623256 | 0.140 |
| triangle_square_share_color | 0.556522 | 0.080 |

Il bottleneck della MLP risulta complessivamente molto accurato, con un F1 medio di 0.897, leggermente inferiore allo 0.906 ottenuto dalla testa lineare. Il miglioramento della MLP si manifesta quindi soprattutto nella classificazione finale, mentre la qualità della rappresentazione concettuale rimane comparabile tra le due configurazioni. 

Rimangono comunque alcune difficoltà sui concetti più composizionali, come triangle_square_share_color, stack_same_size e triangle_square_stack.

#### Importanza dei Concetti

| concept | mean_score_drop | mean_abs_change |
|---|---:|---:|
| has_red | 0.027147 | 0.039738 |
| has_circle | -0.011899 | 0.033629 |
| last_is_red | 0.027882 | 0.030929 |
| has_triangle | 0.006064 | 0.025909 |
| has_square | 0.007378 | 0.024212 |
| last_is_triangle | 0.021994 | 0.024188 |
| first_last_same_size | -0.003923 | 0.022343 |
| has_large | -0.014633 | 0.019095 |
| has_yellow | -0.003825 | 0.016946 |
| has_magenta | -0.005542 | 0.016504 |
| has_blue | 0.006498 | 0.015388 |
| exists_circle_before_last | -0.000781 | 0.015239 |
| has_cyan | -0.005625 | 0.014924 |
| has_green | 0.002598 | 0.014150 |
| first_last_same_color | -0.003213 | 0.013698 |
| has_small | -0.002228 | 0.012389 |
| last_is_circle | -0.003341 | 0.011910 |
| last_is_square | 0.006214 | 0.011659 |
| first_last_same_shape | -0.002634 | 0.011586 |
| triangle_square_share_color | -0.003639 | 0.010795 |
| has_side_by_side | -0.006685 | 0.007853 |
| exists_blue_before_last | 0.001387 | 0.006840 |
| two_or_three_squares_side_by_side | 0.005771 | 0.006720 |
| last_is_small | -0.003682 | 0.006555 |
| last_is_large | -0.003682 | 0.004995 |
| two_circles_side_by_side | 0.003365 | 0.004807 |
| last_is_magenta | -0.003631 | 0.004034 |
| last_is_green | -0.001096 | 0.003552 |
| triangle_square_stack | 0.002784 | 0.003503 |
| two_or_three_squares_stack | 0.003308 | 0.003466 |
| last_is_blue | -0.001939 | 0.003214 |
| side_by_side_same_size | -0.002622 | 0.003145 |
| has_stack | -0.002238 | 0.002365 |
| last_is_cyan | -0.002049 | 0.002332 |
| traffic_light_color_order | 0.001630 | 0.001638 |
| last_is_yellow | 0.000274 | 0.001624 |
| stack_same_size | -0.000275 | 0.001342 |

<details>
<summary><strong>Analisi specifica per task</strong></summary>

#### Task 9 — red triangle on the right

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 9 | red triangle on the right | last_is_triangle | 0.128857 | 0.128857 |
| 9 | red triangle on the right | last_is_red | 0.085641 | 0.085641 |
| 9 | red triangle on the right | has_red | 0.075394 | 0.075394 |
| 9 | red triangle on the right | has_large | 0.055235 | -0.055235 |
| 9 | red triangle on the right | has_triangle | 0.040023 | 0.040023 |
| 9 | red triangle on the right | side_by_side_same_size | 0.030898 | -0.030898 |
| 9 | red triangle on the right | last_is_blue | 0.023488 | -0.023488 |
| 9 | red triangle on the right | has_green | 0.022111 | -0.022111 |
| 9 | red triangle on the right | triangle_square_share_color | 0.018191 | -0.018191 |
| 9 | red triangle on the right | last_is_large | 0.016683 | -0.016683 |

#### Task 14 — palindrome aba

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 14 | palindrome aba | last_is_small | 0.011975 | 0.011975 |
| 14 | palindrome aba | exists_circle_before_last | 0.011876 | 0.011876 |
| 14 | palindrome aba | first_last_same_color | 0.009634 | 0.009634 |
| 14 | palindrome aba | has_cyan | 0.009156 | -0.009156 |
| 14 | palindrome aba | has_triangle | 0.009129 | -0.009125 |
| 14 | palindrome aba | side_by_side_same_size | 0.008160 | 0.008160 |
| 14 | palindrome aba | last_is_triangle | 0.006963 | 0.006963 |
| 14 | palindrome aba | last_is_blue | 0.006245 | -0.006245 |
| 14 | palindrome aba | last_is_red | 0.006029 | 0.006028 |
| 14 | palindrome aba | has_yellow | 0.005881 | 0.005881 |

#### Task 19 — traffic light

| task_id | task | concept | mean_abs_change | mean_score_drop |
|---:|---|---|---:|---:|
| 19 | traffic light | has_red | 0.055080 | 0.055080 |
| 19 | traffic light | has_yellow | 0.052650 | 0.052650 |
| 19 | traffic light | has_circle | 0.050952 | -0.050952 |
| 19 | traffic light | has_cyan | 0.037935 | -0.037935 |
| 19 | traffic light | traffic_light_color_order | 0.028246 | 0.028246 |
| 19 | traffic light | has_large | 0.026069 | -0.026069 |
| 19 | traffic light | last_is_large | 0.022825 | -0.022825 |
| 19 | traffic light | exists_blue_before_last | 0.022658 | -0.022658 |
| 19 | traffic light | exists_circle_before_last | 0.021797 | 0.021797 |
| 19 | traffic light | last_is_blue | 0.018422 | -0.018422 |

</details>

Rispetto alla testa lineare, nella MLP l'intervento sui concetti produce variazioni generalmente più ampie: il MLP sembra quindi sfruttare maggiormente i concetti nella costruzione della predizione, con effetti più marcati quando alcuni di essi vengono alterati. Anche l'importanza dei concetti cambia: con la testa lineare l'influenza era più distribuita e spesso legata a concetti generali, mentre nella MLP emergono maggiormente alcuni concetti direttamente collegati al contenuto del task: ciò è evidente nel task 19 dove traffic_light_color_order, che nella testa lineare aveva un effetto quasi nullo, assume molta più importanza.

![ablation](results/neuro_symbolic_concept_cbm/plots/cumulative_mlp.png)

Rispetto alla testa lineare, nella MLP la curva mostra un calo iniziale più ripido: già la rimozione dei primi concetti più importanti porta a una diminuzione marcata della macro-F1. Dopo alcune oscillazioni, la performance continua a ridursi e, quando vengono rimossi molti concetti, il calo diventa ancora più evidente, indicando che la MLP sfrutta in modo significativo le combinazioni tra le informazioni presenti nel bottleneck.

#### Robustezza & Concetti Rumorosi

| noise_rate | task_accuracy |
|---:|---:|
| 0.00 | 0.876 |
| 0.05 | 0.834 |
| 0.10 | 0.796 |
| 0.20 | 0.728 |
| 0.30 | 0.638 |

Rispetto alla testa lineare, il MLP mostra una robustezza leggermente diversa: l'accuracy parte da 0.876 e diminuisce progressivamente fino a 0.638 con il 30% di rumore. Anche in questo caso gli errori sui concetti si propagano alla classificazione finale, con una degradazione crescente all'aumentare della corruzione.

#### Decomposizioni Errori

| task_id | task_name | oracle_concept_accuracy | predicted_concept_accuracy | accuracy_drop |
|---:|---|---:|---:|---:|
| 16 | car | 0.92 | 0.84 | 0.08 |
| 17 | tower | 1.00 | 0.96 | 0.04 |
| 15 | house | 0.76 | 0.72 | 0.04 |
| 13 | triangle and square, same color | 0.92 | 0.88 | 0.04 |
| 18 | wagon | 1.00 | 0.96 | 0.04 |
| 12 | red triangle on the right and at least one blue object | 0.88 | 0.88 | 0.00 |
| 2 | circle vs any | 1.00 | 1.00 | 0.00 |
| 1 | square vs any | 0.88 | 0.88 | 0.00 |
| 7 | magenta vs any | 1.00 | 1.00 | 0.00 |
| 4 | green vs any | 1.00 | 1.00 | 0.00 |
| 6 | cyan vs any | 0.72 | 0.72 | 0.00 |
| 5 | blue vs any | 1.00 | 1.00 | 0.00 |
| 19 | traffic light | 0.96 | 0.96 | 0.00 |
| 8 | yellow vs any | 1.00 | 1.00 | 0.00 |
| 10 | red triangle on the right and arbitrary objects | 0.96 | 0.96 | 0.00 |
| 0 | triangle vs any | 0.92 | 0.96 | -0.04 |
| 3 | red vs any | 0.96 | 1.00 | -0.04 |
| 9 | red triangle on the right | 0.96 | 1.00 | -0.04 |
| 11 | red triangle on the right and at least one circle | 0.88 | 0.92 | -0.04 |
| 14 | palindrome aba | 0.32 | 0.40 | -0.08 |

Confrontandola con la testa lineare, nella configurazione MLP il drop medio è nullo, anche se alcuni task mostrano una perdita di 0.04–0.08 punti mentre altri rimangono invariati o presentano una variazione negativa. L'impatto degli errori del bottleneck è quindi non uniforme e dipende dal task: per alcuni compiti la MLP riesce a mantenere la stessa performance anche con concetti predetti, mentre per altri gli errori nella rappresentazione concettuale si riflettono direttamente sulla classificazione.

#### Visualizzazioni

Qui rispetto alla testa lineare non possiamo andare a mostrare i pesi relativi alla testa di classificazione, questo perché per il MLP

![inter](results/neuro_symbolic_concept_cbm/plots/tasks_concepts_mlp_test_example_5.png)

---

## Conclusioni

Abbiamo dunque studiato diverse strategie interpretabili e neuro-simboliche applicate al dataset KANDY-Easy, seguendo un percorso che parte dall'analisi della struttura dei dati, passa attraverso la costruzione di rappresentazioni percettive interpretabili e arriva infine alla realizzazione di un Concept Bottleneck Model.

L'analisi iniziale del dataset ha evidenziato una difficoltà crescente tra i task: mentre i primi compiti possono essere risolti osservando proprietà semplici degli oggetti, come forma e colore, i task successivi richiedono di combinare più attributi e di considerare le relazioni tra gli elementi presenti nell'immagine. Inoltre, lo sbilanciamento tra le classi ha mostrato l'importanza di utilizzare metriche come precision, recall e soprattutto F1-score, evitando di affidarsi esclusivamente all'accuracy.

La seconda fase ha mostrato che è possibile costruire una pipeline completamente interpretabile attraverso una fase percettiva deterministica e un insieme di feature semantiche. Le feature estratte dalle immagini risultano sufficienti per ottenere prestazioni elevate nei task più semplici, mentre l'introduzione di informazioni relazionali produce effetti differenti a seconda del problema considerato. Questo risultato suggerisce che la semplice aggiunta di informazioni sulle relazioni non garantisce un miglioramento generale, ma che la rappresentazione deve essere costruita in funzione della struttura del task. Tra i modelli studiati, l'Explainable Boosting Machine è quello che mostra la performance media più elevata, evidenziando in particolare un vantaggio nei task che richiedono combinazioni più complesse di informazioni.

Il confronto tra gli encoder visivi ha invece evidenziato l'importanza del pretraining in presenza di un dataset di dimensioni ridotte. La ResNet-18 pretrained ha ottenuto il miglior risultato in validation e ha mantenuto una buona performance sul test set, mentre i modelli addestrati da zero hanno mostrato prestazioni sensibilmente inferiori. Questo risultato ha motivato la scelta della ResNet-18 come encoder per il successivo Concept Bottleneck Model.

L'esperimento con il CBM ha infine permesso di osservare direttamente il compromesso tra interpretabilità e performance. La versione con testa lineare raggiunge un risultato inferiore alla ResNet utilizzata direttamente per la classificazione e questo comportamento evidenzia il costo del bottleneck concettuale: costringere la rete a rappresentare le immagini attraverso un insieme limitato di 37 concetti migliora la trasparenza della rappresentazione, ma può comportare la perdita di informazioni visive utili alla classificazione.

L'introduzione di una testa MLP permette di recuperare una parte significativa di questa performance, raggiungendo un'accuracy media di 0.902 e una F1 media di 0.897. Il bottleneck rimane anch'esso molto accurato, con F1 medio 0.897, ma non supera quello della testa lineare (0.906). Rimangono tuttavia alcune difficoltà nei concetti maggiormente composizionali, mostrando che la qualità del vocabolario concettuale rappresenta ancora uno dei principali fattori limitanti del modello.

Nel complesso, i risultati mostrano quindi che interpretabilità e performance non devono necessariamente essere considerate come obiettivi incompatibili: il passaggio da una classificazione puramente visuale a una rappresentazione intermedia basata su concetti introduce effettivamente un costo dal punto di vista delle prestazioni, tuttavia una progettazione adeguata del bottleneck e della testa di classificazione permette di ridurre tale costo mantenendo al tempo stesso una rappresentazione molto più leggibile e analizzabile.

Il principale risultato del lavoro non è quindi l'individuazione di un singolo modello ottimale, ma l'evidenza che la scelta della rappresentazione intermedia è fondamentale. Nel caso di KANDY-Easy, le proprietà percettive sono sufficienti per molti task, mentre i problemi più difficili richiedono una rappresentazione esplicita delle relazioni e delle strutture composizionali. Un possibile sviluppo futuro consiste quindi nel migliorare il vocabolario dei concetti, riducendone le ridondanze e introducendo rappresentazioni relazionali più espressive, per poi verificare se gli stessi risultati possano essere mantenuti anche sulla variante KANDY-Hard.
