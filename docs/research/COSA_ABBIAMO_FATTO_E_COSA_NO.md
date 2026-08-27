# Experiment 01 — cosa abbiamo fatto e cosa non abbiamo stabilito

## Scopo del documento

Questo documento è la guida scientifica centrale di Experiment 01. Ricostruisce
l'oggetto osservato, gli strumenti usati, la sequenza degli esperimenti e il
livello probatorio di ogni conclusione. Separa deliberatamente:

- risultati misurati;
- interpretazioni compatibili con i risultati;
- ipotesi ancora aperte;
- analisi proposte ma non eseguite.

Le specifiche congelate e i report di fase restano le fonti canoniche per i
numeri e le soglie. Questa guida non introduce nuovi outcome.

## 1. La domanda in una frase

La domanda non è se un encoder abbia un R² downstream più alto in assoluto. La
domanda è se obiettivi di pretraining diversi rendano l'informazione predittiva
più o meno accessibile a un reader dichiarato, con una quantità dichiarata di
dati etichettati e attraverso un readout dichiarato.

La distinzione operativa è:

```text
informazione potenzialmente contenuta nell'intera rappresentazione
    ≠ informazione preservata da uno specifico pooling/readout
    ≠ informazione recuperabile da un reader finito con n etichette
```

Experiment 01 misura soprattutto il terzo oggetto e alcune proprietà
geometriche associate al secondo. Non misura direttamente l'informazione
totale in senso information-theoretic o il Bayes risk della rappresentazione.

## 2. Che cosa entra ed esce dall'encoder

Ogni endpoint LOB è associato a una finestra di `K=20` istanti. A ogni istante
ci sono `S=4` ruoli token. L'encoder produce quindi una griglia

```text
20 istanti × 4 ruoli × 128 canali.
```

Le analisi principali non trattano tutti gli 80 token come 80 osservazioni
indipendenti. Applicano un readout alla griglia:

- `last_concat512`: prende l'ultimo istante e concatena i quattro vettori di
  ruolo, ottenendo `4 × 128 = 512` feature;
- `meanK_concatS`: media separatamente ciascun ruolo sui 20 istanti e poi
  concatena i quattro vettori, ottenendo ancora 512 feature.

Il confronto `last → meanK` cambia quindi l'operatore temporale mantenendo
uguali dimensione finale e reader.

La diagnostica token-role T2 opera invece sui quattro blocchi da 128 feature
del readout 512D. Ruota l'asse dei quattro ruoli con matrici Haar 4D e poi
solleva ogni asse sui 128 canali. Non ruota i 20 istanti e non tratta gli 80
token della finestra come un singolo spazio di ruolo.

## 3. I bracci confrontati

I nove encoder canonici sono tre obiettivi per tre seed:

| braccio | segnale usato nel training | ruolo in Experiment 01 |
|---|---|---|
| supervised | target LOB futuri dichiarati | baseline target-aligned |
| horizon-JEPA | predizione di latenti futuri | contrasto self-supervised primario |
| masked-JEPA | ricostruzione di latenti mascherati | controllo self-supervised secondario |

L'audit dei checkpoint ha verificato architettura, optimizer, target, budget,
validation history, identità delle righe e SHA-256. Dentro ogni encoder seed i
tre bracci canonici sono row-matched; coppie di seed diversi condividono solo
circa il 7,5% delle rispettive 500.000 righe.

Il confronto principale rimane supervised contro horizon-JEPA. Masked-JEPA è
un controllo informativo, ma non entra nelle conclusioni normalizzate quando
non supera le soglie di ceiling preregistrate.

Un confondimento fondamentale rimane: il supervised canonico ha visto durante
il pretraining target direzionali e di volatilità che vengono poi sondati. Per
questo Phase I non è una dimostrazione di label efficiency end-to-end. F16
riduce una parte di questo problema variando prospetticamente il volume di
supervisione target-aligned, ma non rende gli obiettivi supervised e JEPA
identici in ogni altro aspetto.

## 4. Dati, identità e split

Il dataset canonico contiene 8.039.246 endpoint ordinati provenienti da sette
titoli. Timestamp, simbolo e data di trading sono letti dai CSV originali e
non ricostruiti dalle matrici numeriche.

Il builder e il metadata sidecar hanno verificato fail-closed:

- numero totale e conteggi per stock;
- uguaglianza di `stock_ids` e `day_ids`;
- uguaglianza numerica di `book` e `mid_z`;
- ordine globale e identità `(stock_id, trading_date)`;
- hash per stock e globali.

Train, validation e test sono disgiunti per stock-day. Il train conserva tutti
gli stock-day del train storico; validation e test sono le due metà
cronologiche, per stock, del precedente held-out set. Di conseguenza il test è
fisso e separato nell'esecuzione corrente, ma non è una conferma esterna
pristina: deriva da dati held-out già esplorati storicamente.

## 5. Gli strumenti effettivamente usati

### 5.1 Reader lineari

Phase I usa OLS diagnostico e ridge. Nei confronti scientifici comuni la
regolarizzazione è trace-normalized:

```text
lambda = alpha × trace(covariance) / D.
```

Questo è necessario perché la scala globale di covarianza non è matched:
horizon/supervised è circa 1,40 su `last_concat512`. Confrontare un lambda
assoluto comune sarebbe confounded.

### 5.2 PCA e predictive mass

La PCA viene fittata soltanto sulle feature non etichettate del train. Per il
target `r` e la direzione PCA `j`, Phase II misura

```text
m[j,r] = (u_jᵀ Cov(Z,Y_r))² / (lambda_j Var(Y_r)).
```

La sua somma cumulativa localizza lungo lo spettro la massa predittiva lineare.
Non è R² test e non è una misura completa di informazione.

### 5.3 Null Haar

I sottospazi PCA top-k sono confrontati con sottospazi casuali della stessa
dimensione. T2 applica lo stesso principio nello spazio strutturato dei quattro
ruoli. Questi null rispondono alla domanda “questa direzione è eccezionale
rispetto a direzioni matched?”, non alla domanda causale “perché il training
l'ha prodotta?”.

### 5.4 Whitening progressivo

Il whitening è fittato sul train e applicato come trasformazione post-hoc. È
invertibile a full rank e cambia il conditioning visto dal reader. Non modifica
l'encoder e non dimostra che un regolarizzatore di training causerebbe lo
stesso miglioramento.

### 5.5 Reader MLP

Phase III-R usa il reader congelato
`Linear(D,256)-GELU-Dropout(0.10)-Linear(256,T)`, con selezione su validation.
Questo allarga una specifica classe di reader; non approssima tutti i decoder
possibili e non identifica il mutual information.

### 5.6 Incertezza

Gli intervalli Phase I–III resamplano principalmente seed computazionali e
sottocampioni: misurano robustezza computazionale, non generalizzazione a una
popolazione di mercati. F16 aggiunge 5.000 bootstrap gerarchici stock →
stock-day e sette leave-one-stock-out. Anche questi restano descrittivi perché
gli stock sono soltanto sette.

## 6. Che cosa ha fatto ogni fase

| fase | domanda | operazione | risultato sintetico |
|---|---|---|---|
| Phase I | il segnale è più costoso da recuperare con poche etichette? | learning curves, ridge trace-normalized, whitening | sì per il reader dichiarato; distinto gap di ceiling; `A1` è una label tecnica secondaria e threshold-sensitive |
| Phase II | dove si trova la massa predittiva nello spettro? | PCA ladder, predictive mass, null Haar, bande | horizon direzionale è fortemente anti-allineato alle prime PC |
| Phase III-R | il gap sopravvive a un reader più ricco e al whitening? | MLP nativo/full-whitened | il ceiling sale; il meccanismo low-budget non è identificato; `R3` resta una label tecnica |
| T2 | l'asse di ruolo all-ones è eccezionale? | null Haar 4D con blocchi 128/384D matched | no: il null strutturato non viene rifiutato |
| F16 | come cambia la rappresentazione sotto supervisione target-aligned crescente? | 12 nuovi encoder, quattro budget × tre seed | gap label-matched positivo; transizione già ampia al budget minimo e successiva saturazione, non legge smooth verificata |

Phase III v1, che prevedeva 21.456 modelli, è stata fermata prima del freeze di
selezione e prima dell'accesso al test perché sproporzionata. Non produce claim
test-inference. Phase III-R è un amendment preregistrato ridotto a 1.296 fit e
non comprende un capacity sweep.

## 7. Risultati che consideriamo stabiliti

### 7.1 Gap operativo lineare

Sul readout primario il ceiling lineare supervised meno horizon-JEPA è
`0,165405`, con intervallo di robustezza computazionale
`[0,160753, 0,168857]`. Questo è un gap di performance del reader lineare
dichiarato, non una differenza provata di informazione totale.

### 7.2 Penalità finite-sample più grande sul blocco direzionale

Il gap normalizzato Phase I è `0,5460` per direzione, `0,1838` per volatilità e
`0,1528` per timing. I rapporti direzione/controllo sono `2,97×/3,57×` sulla
scala normalizzata e `1,99×/2,35×` sulla scala R² grezza. Il segno della
specificità descrittiva regge, ma la magnitudine è scale-dependent. Non è un
test d'interazione corretto per la dipendenza tra famiglie e per l'incertezza
stock-day.

### 7.3 Anti-allineamento spettrale direzionale

Su `last_concat512`, la massa predittiva cumulativa nelle prime 8 PC è
`0,0001` per horizon-JEPA e `0,7518` per supervised; a 16 PC è `0,0064` contro
`0,8742`. Per horizon-JEPA tutti i 100 sottospazi Haar superano top-PCA a
`k=8` e `k=16` in tutti i seed. Per supervised top-PCA è al percentile 100 del
null alle profondità riportate. Le code 0/1 archiviate sono frazioni di
superamento su 100 draw, con risoluzione 0,01; non sono p-value continui né
decisioni inferenziali di popolazione.

Questa è forte evidenza di target–variance misalignment nel sistema studiato.
Non implica che “il segnale vive nella coda” come singolo blocco causale: a
`k=128` top-PCA torna a dominare il null e la massa è distribuita su molte
direzioni.

### 7.4 Fragilità al pooling temporale

Al full budget lineare, `last → meanK` cambia il R² direzionale horizon-JEPA da
`0,2199` a `0,0701`; supervised passa da `0,3853` a `0,3941`. È un confronto
dimension-matched e reader-matched. Stabilisce che la media temporale non è
information-neutral per queste rappresentazioni. Non identifica da sola una
dinamica temporale causale interna.

### 7.5 Whitening profondo, non correzione di poche PC

A `k=128` il whitening riduce il gap decisivo del `55,6%` senza eliminarlo. A
`k=508`, la massima profondità valida testata, la riduzione media raggiunge il
`92,6%`. I gap restano positivi e i lower bound non attraversano zero; fallisce
invece a entrambi i budget il criterio composto `lower > 0` e
`mean ≥ δ=0,10`. `k_nonrobust` è quindi un nome tecnico storico per una
transizione di effect size, non per la perdita della separazione statistica.
Il risultato è incompatibile con la narrazione “il problema è concentrato in
poche PC principali”.

Al threshold primario `δ=0,10` il classificatore tecnico restituisce `A1`; le
sensitivity preregistrate danno `D` a `δ=0,05` e `A1` a `δ=0,15`. Questa label
secondaria è quindi threshold-sensitive senza cambiare la curva osservata.

### 7.6 Un reader più ricco recupera performance operativa

Il ceiling MLP horizon-JEPA direzionale raggiunge `0,3448` in coordinate native
e `0,3609` dopo full whitening, rispettivamente `0,8602` e `0,9119` del
supervised. Questi risultati mostrano che il gap lineare non equivale ad
assenza di segnale operativo. Nel regime low-budget full-whitened, però, a
`b_1_4` entrambi i rami hanno il 100% di R² negativi (`-0,416` supervised,
`-2,416` horizon-JEPA in media). Il gap normalizzato che produce `R3` confronta
quindi due fit falliti e non identifica una difficoltà persistente oltre il
conditioning. `R3` resta registrato come output tecnico della regola congelata,
non come conclusione meccanicistica.

### 7.7 Il meccanismo token-role privilegiato non è supportato

La proiezione all-ones 128D conserva poca performance horizon-JEPA e quasi
tutta quella supervised nel vecchio confronto. Tuttavia, rispetto a 100
rotazioni Haar 4D matched, né il blocco comune risulta insolitamente debole né
il complemento 384D insolitamente forte in tutti i seed. La differenza di
proiezione rimane un fatto operativo; la spiegazione “il segnale è
intrinsecamente relazionale nei contrasti” non è supportata.

### 7.8 La supervisione target-aligned produce una transizione precoce

F16 addestra quattro volumi di supervisione per tre seed. Tutti i 12 gap
primari F16-supervised meno horizon-JEPA sono positivi, con intervalli grouped
che escludono zero e 84/84 confronti leave-one-stock-out positivi.

Il criterio preregistrato “supervised-like at low volume” passa a `b_1`
(28.446 righe). Significa che le coordinate normalizzate dichiarate superano
la soglia preregistrata; non significa equivalenza statistica con il
supervised canonico.

Il precedente conteggio di quattro famiglie su sei non costituisce quattro
evidenze indipendenti: `whitening_k128` replica quasi esattamente Axis B
(massima differenza raw R² `0,000340`, correlazione `0,999991`). Dopo la
deduplicazione passano tre famiglie distinte su cinque, meno delle quattro
richieste dalla regola preregistrata. Inoltre Spearman `rho=0,8` su quattro
budget consente una inversione adiacente: soltanto Axis A è strettamente
monotona in tutti i seed. La dipendenza smooth dal volume non è quindi
supportata.

Il fatto nuovo e più informativo è la saturazione precoce. Le 7.116 righe del
budget minimo sono lo `0,108%` del train completo, ma role retention, top-k
predictive mass e perdita al pooling hanno già percorso in media l'82–89% del
tragitto horizon→supervised. La forma osservata è una transizione rapida
seguita da plateau e piccole inversioni. La specificità direzionale non è
identificata perché la normalizzazione dei controlli usa gap di volatilità
quasi nulli.

## 8. Che cosa non abbiamo stabilito

| proposizione | stato | motivo |
|---|---|---|
| horizon-JEPA contiene meno informazione totale | non stabilita | nessun decoder universale o Bayes oracle |
| supervised è end-to-end più label-efficient | non stabilita | ha già visto target downstream nel pretraining |
| l'obiettivo JEPA causa l'anti-allineamento | non stabilita | confronto reale non isola tutti i fattori causali; F16 varia insieme label volume, target exposure e traiettoria di ottimizzazione |
| il whitening è un intervento causale sull'encoder | falso come descrizione | è una trasformazione post-hoc train-only |
| poche PC spiegano il problema | non supportata | servono 128 PC per dimezzare il gap; a 508 resta positivo ma scende sotto la soglia pratica composta |
| la banda 17:32 è povera e 33:64 è speciale | non verificata | confronto originale 16D contro 32D, quindi dimension-confounded |
| il segnale vive in un complemento token-role speciale | non supportata | T2 non rifiuta il null Haar strutturato |
| temporal pooling e role projection misurano lo stesso meccanismo | non stabilita | sono operatori distinti |
| un MLP elimina il gap | non stabilita | aumenta il ceiling, ma il protocollo low-budget è instabile e reader-specific |
| esiste co-adaptation encoder–decoder | non misurata direttamente | manca confronto nativo contro fresh completamente matched |
| l'effetto generalizza a nuovi mercati | non stabilita | sette titoli, un solo dominio |
| il test è una conferma esterna incontaminata | non vero | deriva da un held-out storico già esplorato |
| un regolarizzatore VICReg/SIGReg risolve il problema | non testata | nessun nuovo training di questi bracci |
| il simulatore matematico riproduce il fenomeno | non testata | il simulatore non è stato ancora implementato |

## 9. Gerarchia di evidenza attuale

### Livello 1 — fatti operativi robusti nel dataset

- equivalenza CSV→NPZ e split stock-day disgiunti;
- gap lineare supervised–horizon;
- anti-allineamento top-PCA/Haar;
- interazione con il pooling temporale;
- recupero del ceiling con MLP e whitening;
- null token-role non rifiutato;
- gap F16 positivo nei 12 confronti e nei leave-one-stock-out.

### Livello 2 — interpretazione sostenuta ma condizionale

Gli obiettivi sono associati a geometrie di accessibilità diverse per il
segnale direzionale nel sistema LOB studiato. “Geometria di accessibilità” è
qui un termine operativo relativo a readout, reader, regolarizzazione e budget.

### Livello 3 — ipotesi meccanicistiche aperte

- quale proprietà dell'obiettivo seleziona l'anti-allineamento;
- se la predicibilità intrinseca determina l'allocazione spettrale;
- se un intervento geometrico migliora l'accessibilità senza perdere content;
- se esiste co-adaptation con il decoder nativo;
- quali risultati sopravvivono in un sistema con Bayes predictor noto.

## 10. La formulazione che possiamo difendere

La formulazione più forte compatibile con tutte le evidenze è:

> Nel dataset, negli encoder e nei protocolli dichiarati, horizon-JEPA conserva
> segnale direzionale operativamente decodificabile ma lo organizza in modo
> meno allineato alla varianza principale, più fragile alla media temporale e
> più costoso per reader finite-sample rispetto al supervised target-aligned.
> Whitening e un reader MLP recuperano una parte sostanziale della performance,
> senza identificare informazione totale o il meccanismo causale di training.

F16 aggiunge che l'esposizione supervisionata target-aligned modifica
rapidamente alcune di queste proprietà già al budget minimo. Non stabilisce
una legge smooth del volume, non le muove tutte in modo monotono e non dimostra
specificità esclusiva per la direzione.

## 11. Vincoli empirici per il simulatore

Il simulatore non deve essere costruito per riprodurre ogni dettaglio osservato.
Deve prima consentire di verificare analiticamente content e Bayes risk. I
vincoli empirici sicuri da usare come target qualitativi sono:

1. content comparabile può coesistere con accessibilità finite-sample diversa;
2. la varianza principale può essere anti-allineata con la predittività;
3. un pooling dimension-matched può distruggere selettivamente il segnale;
4. una trasformazione invertibile di conditioning può ridurre il gap;
5. un reader più ricco può aumentare la performance senza provare uguaglianza
   di informazione;
6. anche una piccola esposizione alla supervisione target-aligned può
   selezionare rapidamente una geometria diversa; la forma continua della
   dipendenza dal volume resta non identificata.

Non va incorporato come fatto il privilegio dell'asse all-ones o del suo
complemento. VICReg, SIGReg, topologia e soglie di predicibilità restano bracci
o ipotesi da testare, non proprietà già osservate.

## 12. Decisioni da prendere con il relatore

1. Quale classe di equivalenza definisce “stesso contenuto”: trasformazioni
   invertibili, sufficienza predittiva o Bayes risk equivalente?
2. Quale funzionale di accessibilità ammette un'analisi finite-sample utile
   senza dipendere arbitrariamente da un solo reader?
3. Il primo simulatore deve essere interamente lineare per rendere analitica la
   selezione delle soluzioni?
4. Quali vincoli impediscono alle fattorizzazioni encoder–predictor equivalenti
   di rendere la geometria non identificabile?
5. Conviene testare prima un intervento stretto H0/H-VIC/H-SIG o isolare prima
   il solo effetto di persistenza e nuisance variance?
6. La componente topologica aggiunge una domanda indipendente oppure diluisce
   il contributo spettrale principale?

## 13. Percorso di verifica

Per una lettura rapida:

1. questo documento;
2. `README.md`;
3. `docs/research/RESEARCH_NOTE_GEOMETRY_ACCESSIBILITY.md`;
4. report Phase I e Phase II;
5. report Phase III-R, T2 e F16;
6. specifiche e manifest soltanto per l'audit procedurale.

La fotografia operativa, inclusi hash e asset locali, rimane in
`PROJECT_STATE.md`.
