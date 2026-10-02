# KANDY-agent

## Obiettivo

Una volta completate le attività principali dell'elaborato su KANDY-Easy, è stato sviluppato un ulteriore esperimento utilizzando un piccolo modello multimodale locale, Qwen3.5-4B; il progetto è stato eseguito interamente sulla CPU del portatile.

L'idea nasce dal tentativo di affiancare al modello una memoria esterna persistente, così da permettergli di conservare informazioni che normalmente non verrebbero mantenute in modo esplicito tra episodi diversi. In particolare, l'agente può acquisire concetti visivi richiesti dal task, recuperare dalla memoria quelli già conosciuti e, in caso di errore, analizzare la possibile causa dell'errore e utilizzare questa informazione per raffinare il concetto memorizzato.

La pipeline può essere riassunta come segue:


1. domanda + immagine vengono inviate all'agente (ex. l'immagine contiene un cerchio? 2/0000.png)
        
2. si chiede all'llm di estrarre i concetti e le relazioni dalla domanda (ex. l'immagine contiene un cerchio? -> {cerchio})

3. si esegue un controllo della memoria (ex. {cerchio} è nella memoria?)
        
4. si acquisiscono i concetti mancanti (ex. {cerchio} non è in memoria, llm produce {cerchio} e lo aggiunge alla memoria)
        
5. classificazione dell'immagine (ex. llm analizza 2/0000.png)
        
6. predizione + evidenza + concetti utilizzati (ex. risultato: 0, l'immagine contiene un rombo giallo, nessun concetto in memoria utilizzato)

7. eventuale diagnosi dell'errore (ex. il risultato su 2/0000.png è errato, viene passata la ground truth e si fa elaborare all'llm un errore)

8. eventuale miglioramento della memoria (ex. si utilizza l'annotazione dell'errore ottenuta per migliorare la memoria del concetto)


L'obiettivo dell'esperimento non è quindi solamente verificare se la memoria migliori l'accuracy, ma soprattutto osservare se un modello multimodale molto piccolo possa sfruttare una rappresentazione concettuale esterna e modificarla sulla base della propria esperienza.

Per l'acquisizione di un concetto ho utilizzato un solo esempio visivo del training set; ciò è dovuto a due motivi principali, il primo riguardante i tempi di esecuzione, in quanto l'analisi dell'immagine è ciò appesantisce di più il processo e poi perché, in questo modo, si rende esplicito il rapporto tra esperienza osservata e conoscenza acquisita, permettendo di distinguere meglio la fase di acquisizione da quella di miglioramento.

Come detto sopra le informazioni relative ai concetti utilizzati e all'evidenza prodotte dal modello sono prodotte da Qwen stesso: non vengono quindi considerate una prova causale del processo decisionale interno del modello, ma come una forma di spiegazione esplicita e plausibile del comportamento osservato.

## Struttura

I moduli dell'agente sono organizzati nel seguente modo:

```text
agent/
├── agent.py
├── memory.py
├── qwen_client.py
├── kandy_loader.py
├── experiment.py
└── experiments/
    └── 03_memory_ablation.py
```

`agent.py` contiene la logica dell'agente: quindi estrae i concetti necessari, recupera quelli presenti in memoria e avvia l'acquisizione quando un concetto richiesto non è ancora disponibile.

`memory.py` gestisce la memoria esterna. I concetti vengono salvati con una versione, una provenienza e alcuni metadati, il "refinement" permette di aggiornare un concetto mantenendo traccia della modifica.

`qwen_client.py` contiene le interazioni con il modello locale tramite `llama-server`. Qwen viene utilizzato per l'estrazione dei concetti, per l'acquisizione, per la classificazione e per la diagnosi degli errori.

`kandy_loader.py` gestisce il caricamento degli esempi KANDY-Easy e, per i task supportati, deriva il ground truth a partire dalle descrizioni simboliche del dataset.

`experiment.py` contiene l'esperimento principale sul task 2 (`circle vs any`), mentre `experiments/` contiene una versione ridotta degli esperimenti successivi, progettata per contenere il numero di inferenze necessarie durante l'esecuzione, dati i limiti della CPU.

## Test svolti

Gli esperimenti sono stati organizzati per verificare proprietà diverse del comportamento dell'agente, tuttavia in quanto utilizzano un numero molto piccolo di esempi del dataset, parliamo di un range che varia dai 2 ai 5 esempi, i risultati non sono da considerarsi attendibili.

### Acquisizione, generalizzazione e primo test di refinement

Il primo esperimento è stato condotto sul task circle vs any. L'agente parte da una memoria vuota, osserva un singolo esempio del training set e utilizza Qwen per costruire una rappresentazione del concetto circle, composta da definizione, indizi visivi, varianti di aspetto e possibili confusioni.

Nel piccolo test successivo, la memoria acquisita ha prodotto un cambiamento concreto nel comportamento del modello. Un'immagine positiva particolarmente difficile, interpretata come un rombo in assenza di memoria, è stata invece riconosciuta correttamente dopo l'acquisizione di circle. Un'immagine negativa è rimasta correttamente classificata come negativa, senza introdurre un falso positivo.

È stato inoltre eseguito un primo test di error-driven refinement. Dopo l'acquisizione, il modello è stato esposto a una piccola sequenza di nuovi esempi del training set. Quando ha commesso un errore sull'immagine più difficile, Qwen ha valutato che il problema non richiedesse una modifica della rappresentazione di circle; gli altri esempi sono stati classificati correttamente. Di conseguenza, in questo esperimento il refinement non ha prodotto una nuova versione del concetto.

Questo risultato è comunque utile perché mostra che il meccanismo non modifica automaticamente la memoria dopo ogni errore: un errore percettivo, soprattutto in presenza di immagini molto piccole o rasterizzate, può essere considerato distinto da un errore della rappresentazione concettuale.

### Memory ablation

È stato poi eseguito un test di ablation per verificare quanto il contenuto della memoria influenzi il comportamento del modello. La stessa immagine e la stessa domanda vengono mantenute nelle diverse condizioni, mentre cambia soltanto la memoria fornita a Qwen.

Sono state confrontate quattro condizioni:

- nessuna memoria esterna;

- memoria rilevante contenente circle, acquisita da un'immagine reale del training set;

- memoria irrilevante ma appartenente allo stesso dominio visivo, contenente square;

- memoria irrilevante e appartenente a un dominio diverso, contenente cow.

Nel piccolo insieme di test utilizzato, la memoria circle ha corretto l'errore osservato nel baseline, mentre square non ha prodotto lo stesso effetto. È però emerso un risultato inatteso: anche la memoria cow ha corretto il caso difficile.

Il test suggerisce quindi che la presenza di una memoria esterna può modificare il comportamento del modello, ma non permette ancora di attribuire il miglioramento esclusivamente alla rilevanza semantica del concetto memorizzato. Il comportamento della memoria irrilevante rende necessario analizzare le spiegazioni e gli output prodotti dal modello e, in futuro, progettare controlli più rigorosi.

## Stato attuale e limiti

I risultati ottenuti finora, di nuovo, sono da considerarsi preliminari e poco incoraggianti. Gli esperimenti sono stati deliberatamente eseguiti su quantità molto ridotte di dati, sia per contenere i tempi di inferenza sia perché l'intero progetto è stato svolto utilizzando esclusivamente la CPU del portatile.

Questi esperimenti vanno pertanto considerati soprattutto come una prova di fattibilità della pipeline e come base per esperimenti successivi. In particolare, è ancora da verificare in modo più sistematico se gli errori concettuali possano produrre miglioramenti utili e se tali modifiche possano essere riutilizzate in esempi successivi, migliorando la capacità di generalizzazione.

Inoltre un "problema" noto è che nel prototipo presentato, è il modello stesso che si giudica, questo può ovviamente condurre a diagnosi errate o circolari. Tuttavia, in questa fase l'obiettivo non è assumere tali diagnosi come valutazioni oggettive del comportamento del modello, ma verificare se un meccanismo di questo tipo possa costituire una prima forma di miglioramento della memoria guidato dall'esperienza.

## Possibili miglioramenti ed esperimenti futuri

Ho poi individuato le seguenti possibilità di miglioramento:

- Scalare il numero di esempi utilizzate negli esperimenti, così da ottenere risultati più significativi e meno dipendenti da singoli casi
- Migliorare il meccanismo di miglioramento tramite errore, dividendo per categoria gli errori fatti dal modello.
- Rendere il miglioramento della memoria più robousto: modificare un concetto sulla base di più errori.
- Estendere la memoria alle relazioni e verificare se concetti e relazioni possano essere riutilizzati in task composizionali più complessi, ad esempio costruendo concetti composti a partire da primitive come triangle, square, stack e same_size.
- Se il modello mostrasse un qualche miglioramento sensibile, sarebbe forse interessante il seguente esperimento: valutare le prestazioni del modello su un task visivo simile ma distinto da KANDY-Easy, effettuare successivamente un fine-tuning su KANDY-Easy utilizzando, oltre agli esempi del dataset, anche le informazioni consolidate nella memoria esterna, trasformate in esempi di training relativi ai concetti acquisiti e agli errori osservati, e infine ripetere la valutazione sul task esterno. Il confronto tra le prestazioni ottenute prima e dopo il fine-tuning permetterebbe di osservare se la conoscenza acquisita su KANDY-Easy possa essere in qualche misura consolidata nei pesi del modello e trasferita anche a un contesto visivo differente.