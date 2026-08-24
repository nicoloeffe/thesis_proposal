# Stato del progetto — geometria e accessibilità dell’informazione predittiva

## Snapshot empirico post-P0/post-consolidamento e direzione di ricerca

**Data:** 28 luglio 2026

**Commit sorgenti del consolidamento:** `6a94bd5`

**Stato:** studio empirico LOB consolidato; programma teorico e meccanicistico
definito, non ancora eseguito.

> Questo documento sostituisce lo snapshot “post held-out” precedente. I numeri
> anteriori al P0 non vanno più usati come risultati definitivi. Il nuovo stato
> incorpora: correzione dello split, rigenerazione dei nove dump canonici,
> reader MLP multiseed, pooling alternativi, null casuale, Hadamard, angoli
> principali e diagnostici temporali/attenzionali. Conserva lo stato tecnico
> completo del progetto e formalizza la nuova direzione intellettuale: il LOB
> rimane il caso empirico in cui il fenomeno è stato scoperto, mentre la domanda
> scientifica diventa generale e riguarda rappresentazioni apprese di processi
> stocastici parzialmente osservati.

---

## 1. Executive summary

La tesi confronta tre obiettivi, a backbone sostanzialmente condiviso:

1. supervised end-to-end;
2. JEPA-horizon, che predice rappresentazioni future;
3. JEPA-masked/completion, che completa regioni mascherate.

Il risultato centrale non è che “supervised contiene informazione e JEPA no”.
È una separazione fra tre livelli:

```text
contenuto nella griglia
    → contenuto conservato da un pooling fisso
        → accessibilità del contenuto nelle direzioni di maggiore varianza
```

Il supervised produce un codice direzionale:

- temporalmente coerente;
- quasi interamente lineare;
- concentrato in poche direzioni ad alta varianza;
- robusto sia a `last_concat512` sia a `meanK_concatS`.

JEPA-horizon conserva invece molto segnale direzionale nell’ultimo timestep, ma:

- il segnale è fortemente non lineare;
- le prime componenti di varianza sono peggiori di sottospazi casuali;
- gran parte del contenuto vive nei contrasti fra token;
- la media temporale distrugge quasi tutto il segnale direzionale.

JEPA-masked è più diffuso dimensionalmente, ma contiene meno segnale
direzionale.

Il precedente claim “JEPA conserva circa l’82% del contenuto supervised” resta
vero **solo per `last_concat512`**. Con `meanK_concatS` il rapporto scende al
12,6%. Il contenuto non può quindi essere descritto indipendentemente dal
readout.

Il test held-out non dà più una risposta binaria “geometria genuina” contro
“co-adattamento”. I risultati indicano:

- forte co-adattamento sui target direzionali addestrati e sui target held-out
  di depth/imbalance;
- una componente più generale sul timing, dove il vantaggio supervised
  sopravvive anche a rango pieno e con MLP.

La tesi oggi sostiene solidamente un risultato **descrittivo e diagnostico**:

> Obiettivi diversi non determinano soltanto quanto segnale è rappresentato, ma
> dove viene collocato nella distribuzione di varianza, come viene distribuito
> fra token e quanto resta coerente attraverso il tempo.

Il LOB non è più considerato il fine scientifico del progetto, ma il sistema
reale che ha fatto emergere il problema. La prossima domanda è più generale:

> In che modo differenti obiettivi supervisionati e self-supervised organizzano
> l’informazione predittiva appresa da un processo stocastico parzialmente
> osservato, e quali aspetti di tale organizzazione determinano accessibilità,
> sample efficiency e co-adattamento?

La direzione concordata è costruire una replica meccanicistica in un sistema
generativo controllato, con stato latente, statistica sufficiente, nuisance e
rischio di Bayes noti. VICReg e SIGReg diventano interventi causali sulla
geometria, non il punto di partenza; LeJEPA completo resta un’eventuale
estensione successiva. Un livello topologico sarà introdotto soltanto mediante
processi con topologia latente nota, non come diagnostica decorativa su point
cloud reali.

---

## 2. Cosa è cambiato rispetto al vecchio snapshot

### 2.1 Correzione P0

Il vecchio protocollo confondeva posizioni dentro `valid_t` ed endpoint raw.
Questo poteva accoppiare readout e target riferiti a righe diverse. Dopo la
correzione:

- snapshot pre-fix: commit `2d619b8`, tag `pre-p0-fix-2026-07-28`;
- fix P0: commit `17c5ffd`;
- vecchi 104 GiB conservati come artefatti invalidi;
- nuova estrazione canonica: 9/9 checkpoint ep20, due readout base, 5,3 GiB;
- train/validation: 100k/50k endpoint;
- zero stock-day condivisi;
- hash, inventario, split e ordine campioni verificati fail-closed;
- suite corrente: `61 passed`.

I numeri del vecchio snapshot erano vicini in alcuni casi, ma non vanno citati:
la base metodologica era invalida.

### 2.2 Timing held-out

Il vecchio Gate 1 valutava il timing solo sui casi non censurati, ma salvava
anche le righe censurate come durate esatte/capped. Ora screening e artefatto
usano lo stesso estimand:

```text
log1p(durata osservata oppure capped)
```

su tutte le righe, con massimo 600. La censura è circa 0,1%; Gate 1 dà
`R²(time | 14 target addestrati) = 0,1817`.

È un target capped, non un modello survival. Questa distinzione va dichiarata.

### 2.3 Reader e geometria

Il vecchio MLP era una singola inizializzazione per 80 epoche. Ora usa:

- 5 seed reader per ogni seed encoder;
- split interno fisso derivato da `split_seed=0`;
- early stopping, patience 10;
- massimo 80 epoche;
- media e deviazione reader separate dalla deviazione encoder.

Sono inoltre stati aggiunti:

- `meanK_concatS`;
- `mean_all128`;
- blocchi Hadamard media/contrasti;
- null casuale numerico a 20 draw;
- ladder normalizzata `R²(m)/R²(D)` contro `m/D`;
- angoli principali;
- participation ratio ed effective rank;
- attenzione supervised;
- screening di tutti i 20 timestep;
- diagnostica di `final_norm.gamma`.

---

## 3. Setup canonico

- **Dataset:** LOBench/SZSE, 7 titoli, quote senza trade prints.
- **Encoder:** griglia token-preserving, `K=20`, `S=4`, `d_model=128`.
- **Bracci:** supervised, JEPA-horizon, JEPA-masked.
- **Seed:** 3 per braccio.
- **Stato analizzato:** epoca 20 per tutti i nove encoder.
- **Split:** stock-day grouped, `split_seed=0`, nessuna sovrapposizione.
- **Campione della batteria:** 100.000 train, 50.000 validation.
- **Target addestrati:** 12 direzionali indipendenti + 2 volatilità.
- **Held-out:** 8 imbalance, 8 depth, 1 timing.
- **Readout base:**
  - `last_concat512`: quattro token dell’ultimo timestep;
  - `meanK_concatS`: media temporale, poi concat dei quattro token.

La SVD dei target direzionali conferma rango 12: spread, best-bid-rel e
best-ask-rel contengono una ridondanza algebrica esatta; le copie ridondanti
restano nelle tabelle per-target ma sono escluse dagli aggregati.

---

## 4. Che cosa misura la batteria

### 4.1 Contenuto lineare

`R²(D)` è il risultato di una min-norm OLS sul readout completo. Dice quanto
target è linearmente recuperabile da quel readout.

Non misura tutta l’informazione possibile nella griglia: un pooling può averla
già distrutta.

### 4.2 Accessibilità

La ladder usa PCA centrata, non standardizzata, e min-norm OLS sulle prime `m`
componenti:

```text
A(m) = R²(top-m PCA) / R²(full-rank)
```

`A(m)` misura quanto del contenuto lineare è collocato nelle direzioni di
maggiore varianza.

### 4.3 Ceiling MLP

L’MLP a due layer hidden-256 è un riferimento non lineare operativo. Non è un
upper bound informativo assoluto e non vede la griglia completa: vede soltanto
il pooling scelto.

### 4.4 Null numerico

Il null usa sottospazi casuali ortonormali nello stesso readout e lo stesso
reader lineare.

La diagonale analitica `m/D` vale nel caso isotropo sintetico. Nei readout reali
fortemente anisotropi il null numerico può essere molto diverso.

---

## 5. Risultati attuali

### 5.1 Reader MLP direzionale

| pooling | braccio | R² MLP | std encoder | std reader | std totale |
|---|---|---:|---:|---:|---:|
| last | jepa_horizon | 0,3191 | 0,0064 | 0,0045 | 0,0078 |
| last | jepa_masked | 0,1501 | 0,0125 | 0,0037 | 0,0130 |
| last | supervised | 0,3881 | 0,0004 | 0,0037 | 0,0037 |
| meanK | jepa_horizon | 0,0494 | 0,0103 | 0,0039 | 0,0110 |
| meanK | jepa_masked | 0,0008 | 0,0007 | 0,0009 | 0,0011 |
| meanK | supervised | 0,3917 | 0,0006 | 0,0029 | 0,0030 |

Per `last_concat512`:

```text
0,3191 / 0,3881 = 82,23%
```

Per `meanK_concatS`:

```text
0,0494 / 0,3917 = 12,62%
```

Conclusione: l’82% descrive il contenuto nell’ultimo timestep, non il contenuto
globale dell’encoder e non una proprietà invariabile al pooling.

### 5.2 Ladder direzionale

Al punto comune `m/D=1/32`:

| readout | braccio | R²(m)/R²(D) | null numerico | R²(D) |
|---|---|---:|---:|---:|
| last | jepa_horizon | 0,0050 | 0,0563 | 0,2111 |
| last | jepa_masked | 0,0022 | 0,0488 | 0,1006 |
| last | supervised | 0,8971 | 0,6118 | 0,3756 |
| meanK | jepa_horizon | 0,0042 | 0,0530 | 0,0631 |
| meanK | jepa_masked | 0,0232* | 0,1956 | 0,0041 |
| meanK | supervised | 0,9777 | 0,7494 | 0,3865 |

`*` denominatore sotto 0,01: rapporto non affidabile.

Il supervised è sopra il null numerico. JEPA-horizon e JEPA-masked/last sono
sotto il null: le prime componenti sono sistematicamente anti-allineate al
segnale direzionale.

### 5.3 Dipendenza temporale

Nel supervised:

- timestep 1–19: `97,4–98,6%` del contenuto full-rank già a `m=16`;
- timestep 20: `90,1%`;
- R² full-rank quasi costante attraverso K.

In JEPA-horizon:

- il contenuto full-rank varia molto con la posizione;
- cresce fortemente verso l’ultimo timestep;
- l’accessibilità top-16 resta prossima a zero ovunque.

In JEPA-masked:

- solo le ultime tre posizioni hanno R² full-rank direzionale almeno 0,01;
- la media temporale rende il denominatore quasi nullo.

La media temporale è quindi un test di coerenza delle coordinate, non un pooling
neutrale. Il supervised mantiene un codice temporalmente allineato; i JEPA no.

### 5.4 Hadamard: componente comune e contrasti

Per il supervised:

| base | blocco | accessibilità a 1/32 | R² full-rank |
|---|---|---:|---:|
| last | media comune | 0,7185 | 0,3730 |
| last | contrasti | 0,3984 | 0,3331 |
| meanK | media comune | 0,7918 | 0,3892 |
| meanK | contrasti | 0,1672 | 0,3073 |

Il supervised rende il segnale disponibile in una componente condivisa dai
quattro token.

Per JEPA-horizon/last:

- media comune full-rank: circa `0,041`;
- contrasti full-rank: circa `0,205`;
- concat completo: circa `0,211`.

Il contenuto JEPA è quindi soprattutto relazionale fra token. `mean_all128` è
lo stesso sottospazio della media Hadamard meanK, a scala costante.

### 5.5 Rango e anisotropia

| pooling | braccio | participation ratio | effective rank |
|---|---|---:|---:|
| last | jepa_horizon | 7,91 | 11,75 |
| last | jepa_masked | 45,31 | 88,69 |
| last | supervised | 7,05 | 14,46 |
| meanK | jepa_horizon | 7,32 | 10,22 |
| meanK | jepa_masked | 43,29 | 82,64 |
| meanK | supervised | 8,58 | 15,05 |

Supervised e horizon sono entrambi fortemente anisotropi. La differenza è
l’allineamento del segnale con la varianza, non la semplice isotropia.

Masked è più diffuso dimensionalmente ma meno informativo. “Più isotropo” e
“più utile” non sono sinonimi.

### 5.6 Volatilità

La volatilità resta ben leggibile in tutti i bracci:

- MLP circa `0,42–0,51`;
- la media temporale non la distrugge;
- supervised è comunque più accessibile, essendo stato addestrato anche su
  questi target.

La volatilità è un controllo di specificità, non un vero held-out.

### 5.7 Attenzione e `final_norm`

La testa supervised ha:

- entropia normalizzata `0,975–0,992`;
- peso dell’ultimo timestep `5,3–6,9%`;
- riferimento uniforme sui 20 timestep: 5%.

La testa non privilegia nettamente l’ultimo timestep.

La media di `|gamma|` in `final_norm` è:

- horizon: 0,975;
- masked: 1,000;
- supervised: 0,987.

Non emerge un riscalamento affine grossolano che spieghi da solo la geometria.
Resta il limite del weight decay applicato ai parametri di normalizzazione.

### 5.8 Angoli principali

Tutti i punti superano la soglia di affidabilità del gap spettrale e a `m=D`
l’energia allineata è 1.

A basso rango, però, l’energia euclidea del sottospazio dei coefficienti è molto
piccola anche per il supervised. Non è una contraddizione con la ladder:

- gli angoli pesano la norma dei coefficienti;
- la ladder pesa l’effetto predittivo attraverso la covarianza;
- direzioni a bassa varianza richiedono coefficienti grandi e possono dominare
  la norma di `B`.

Gli angoli attuali caratterizzano la geometria dei coefficienti ma non sono una
seconda misura equivalente dell’R². Una versione futura dovrebbe usare un
sottospazio covariance-weighted, ad esempio `Σ^(1/2) B`.

---

## 6. Co-adattamento: verdetto aggiornato

In questa sezione il termine è usato nel senso ampio di allineamento prodotto
dai target di training. Il held-out misura quanto tale vantaggio sopravvive a
target non ottimizzati, ma non confronta ancora in modo controllato decoder
nativo e decoder fresh della stessa classe. È quindi evidenza di
task-specific alignment e possibile co-adattamento, non una sua identificazione
causale completa.

Il test A/B originario chiedeva:

- **A — geometria generale:** il vantaggio supervised resta su target mai
  ottimizzati;
- **B — co-adattamento:** il vantaggio collassa verso JEPA sui target held-out.

I nuovi reader MLP e i nuovi pooling producono un verdetto misto.

### 6.1 Depth

Ladder meanK:

- supervised m16/full: `0,021 / 0,072`;
- horizon m16/full: `0,005 / 0,082`.

Il supervised è più accessibile a basso rango ma non ha più contenuto
full-rank. Con MLP last:

- horizon: `0,174`;
- supervised: `0,162`.

Questo è il caso più pulito compatibile con task-specific alignment e
co-adattamento geometrico.

### 6.2 Imbalance

Ladder meanK:

- gap m16 supervised-horizon: circa `+0,052`;
- gap full-rank: circa `+0,023`.

MLP last:

- horizon: `0,201`;
- supervised: `0,193`.

Il vantaggio supervised è soprattutto accessibilità/organizzazione, non
contenuto assoluto.

### 6.3 Timing

Ladder meanK:

- supervised m16/full: `0,512 / 0,563`;
- horizon m16/full: `0,424 / 0,474`.

MLP meanK:

- supervised: `0,570`;
- horizon: `0,478`.

Il gap resta quasi invariato a rango pieno e con MLP. Il timing suggerisce una
componente più generale della rappresentazione supervised.

Gate 1 garantisce però soltanto bassa dipendenza lineare dai 14 target
addestrati. Timing e volatilità/attività possono essere collegati non
linearmente. Il risultato è forte evidenza, non prova causale.

### 6.4 Conclusione A/B

La formulazione aggiornata è:

> Il vantaggio supervised sui target direzionali e sui target held-out
> strutturalmente vicini è in larga parte compatibile con task-specific
> alignment e co-adattamento: l’informazione viene resa linearmente accessibile
> senza un aumento corrispondente del contenuto. Sul timing sopravvive invece un
> vantaggio di contenuto compatibile con una organizzazione più generale dello
> stato di mercato.

Non è più corretto scrivere che “il test held-out supporta principalmente A”.

---

## 7. Cosa è stabilito e cosa no

### Stabilito

- Il supervised concentra quasi tutto il segnale direzionale in poche
  componenti ad alta varianza.
- JEPA-horizon conserva molto segnale nell’ultimo timestep ma lo colloca in
  direzioni a bassa varianza e in relazioni fra token.
- Le prime componenti JEPA sono peggiori di sottospazi casuali per il target
  direzionale.
- Il contenuto JEPA è fortemente dipendente dal pooling temporale.
- JEPA-masked è più diffuso dimensionalmente ma meno informativo.
- Depth/imbalance supportano una componente forte di co-adattamento.
- Timing conserva un vantaggio supervised anche come contenuto.
- Isotropia, rango effettivo, contenuto e accessibilità sono proprietà
  distinte.

### Non stabilito

- Che l’EMA sia la causa dell’anti-allineamento.
- Che rendere JEPA isotropo aumenti l’accessibilità dei target LOB.
- Che una rappresentazione isotropa sia migliore per target domain-specific.
- Che il supervised abbia una geometria universalmente migliore.
- Che il segnale distrutto da `meanK` non esista nella griglia completa.
- Che gli angoli euclidei attuali misurino direttamente energia predittiva.
- Che il timing sia indipendente non linearmente dai target di training.

---

## 8. Claim di tesi consigliato

### Claim principale

> A parità di dominio e backbone, l’obiettivo di training determina non soltanto
> il contenuto predittivo della rappresentazione, ma la sua collocazione nello
> spettro di varianza, la distribuzione fra token e la coerenza temporale. La
> supervisione produce un codice task-aligned e linearmente accessibile; JEPA
> conserva parte rilevante del contenuto in una forma relazionale,
> position-dependent e anti-allineata alle componenti principali.

### Claim sul co-adattamento

> La maggiore accessibilità supervised è compatibile con co-adattamento ai
> target di training: sui target held-out di depth e imbalance il vantaggio di
> contenuto collassa. Il timing mostra però una componente più generale che non
> si esaurisce nel riordinamento a basso rango. La stima causale del
> co-adaptation gap richiede ancora un confronto native/fresh matched.

### Claim da non usare

- “JEPA e supervised contengono la stessa informazione.”
- “JEPA conserva sempre l’82%.”
- “Il supervised è più isotropo.”
- “Il held-out dimostra la geometria genuina.”
- “SIGReg/LeJEPA è già validato sui LOB.”

---

## 9. Posizione intellettuale raggiunta

### 9.1 Il LOB è il caso di scoperta, non più il fine

Il progetto è nato come studio di representation learning su Limit Order Book.
Il consolidamento ha però fatto emergere una domanda più generale del dominio:
due rappresentazioni possono conservare segnale predittivo comparabile e
organizzarlo in modo radicalmente diverso rispetto a varianza, token, tempo e
readout.

La posizione attuale è quindi:

> Il LOB rimane uno studio empirico difficile e autentico, utile perché il
> fenomeno è comparso senza essere costruito nel dataset. Il contributo
> scientifico cercato non riguarda primariamente la microstruttura, ma il modo
> in cui gli obiettivi di apprendimento organizzano l’informazione predittiva
> nelle rappresentazioni di sistemi dinamici.

Il lavoro già svolto non viene scartato né ridimensionato. Assume il ruolo di
`Study I — empirical discovery`, sul quale costruire una replica controllata.

### 9.2 Domanda centrale

> In che modo differenti obiettivi supervisionati e self-supervised preservano,
> collocano e rendono accessibili le variabili predittive di un processo
> stocastico parzialmente osservato?

Le domande operative sono:

1. quale contenuto predittivo è presente nella rappresentazione;
2. dove cade rispetto allo spettro di covarianza;
3. quanto è accessibile con pochi dati e readout semplici;
4. quanto è distribuito fra token e attraverso il tempo;
5. quanto il decoder nativo è co-adattato alla geometria dell’encoder;
6. se una regolarizzazione geometrica può modificare l’accessibilità senza
   distruggere contenuto;
7. in un sistema con topologia latente nota, quali invarianti vengono
   preservati dai diversi obiettivi.

### 9.3 Ambizione corretta

Lo stato attuale è già sufficiente per una tesi magistrale forte e per una nota
tecnica. Nel migliore degli esiti, il programma seguente ha potenziale da paper
perché combina:

```text
fenomeno reale
    → formalizzazione matematica
        → replica meccanicistica con ground truth
            → intervento causale
                → ritorno al caso reale
```

Il potenziale da paper non è ancora un risultato. Dipende dalla novità rispetto
alla letteratura, dalla robustezza della replica e dalla capacità di isolare un
solo meccanismo centrale.

---

## 10. Impianto matematico proposto

### 10.1 Sistema parzialmente osservato

Si considera uno stato latente composto da fattori predittivi e nuisance:

\[
X_t=(S_t,N_t),
\]

con osservazioni

\[
O_t=g(S_t,N_t,\varepsilon_t),
\]

target futuri

\[
Y_t=h(S_{t+1:t+H}),
\]

e rappresentazione appresa da una finestra osservata:

\[
Z_t=f_\theta(O_{t-L+1:t}).
\]

Nel simulatore saranno noti processo generativo, statistica sufficiente,
osservabilità, fattori nuisance e rischio di Bayes. Sul LOB queste quantità non
sono note e possono essere soltanto approssimate mediante probe e controlli.

### 10.2 Quattro oggetti da tenere distinti

#### Contenuto

Idealmente il contenuto predittivo è caratterizzato da

\[
R^*(Z)=\inf_g \mathbb{E}[\ell(Y,g(Z))],
\]

dove l’infimo è sugli stimatori misurabili. Nel simulatore questo rischio può
essere confrontato con il rischio di Bayes dato l’intero passato osservato. Le
versioni ristrette a una classe di funzioni sono misure operative. Sul LOB:

- il probe lineare misura contenuto linearmente recuperabile;
- l’MLP è un ceiling operativo più flessibile;
- nessuno dei due dimostra l’informazione totale contenuta nella griglia.

#### Accessibilità

L’accessibilità non è una proprietà binaria. Va descritta come funzione di:

- dimensione del readout \(m\);
- numero di esempi etichettati \(n\);
- classe di funzione;
- regolarizzazione;
- pooling;
- protocollo di ottimizzazione.

L’oggetto corretto è quindi una curva \(A(m,n)\), non soltanto
`R²(m)/R²(D)`.

#### Allineamento geometrico

L’allineamento confronta il sottospazio di varianza con quello predittivo. Oltre
alla PCA di \(\Sigma_Z\), si propone l’operatore in coordinate whitened

\[
M_{\mathrm{pred}}
=
\Sigma_Z^{-1/2}
\Sigma_{ZY}
\Sigma_Y^{-1}
\Sigma_{YZ}
\Sigma_Z^{-1/2},
\]

con pseudoinversa o ridge quando necessario. I suoi autospazi identificano le
direzioni predittive normalizzate per la covarianza. Il confronto con gli
autospazi di \(\Sigma_Z\) usa angoli principali, energia catturata e distanze
fra sottospazi.

#### Co-adattamento

Il co-adattamento è il vantaggio del decoder addestrato con l’encoder rispetto
a un decoder fresh della stessa classe, addestrato con gli stessi dati e un
protocollo comparabile. Va distinto da:

- contenuto della rappresentazione;
- generalizzazione a target held-out;
- sample efficiency;
- robustezza cross-seed.

Con questa definizione più stretta, il test held-out LOB attuale è un proxy di
task-specific alignment, non ancora una misura completa del co-adaptation gap.

Nel simulatore si potranno aggiungere decoder swap e allineamenti
Procrustes/CCA, ma non sono necessari per la prima definizione operativa.

### 10.3 Non-identificabilità geometrica

Questo è il punto teorico più promettente. Se un decoder produce \(WZ\), per
ogni trasformazione invertibile \(T\):

\[
Z'=TZ,
\qquad
W'=WT^{-1}
\]

produce la stessa predizione. Il rischio e il contenuto possono quindi restare
invariati mentre cambiano:

- spettro di covarianza;
- condizionamento;
- orientamento delle componenti principali;
- norma richiesta al decoder;
- sample efficiency sotto ridge;
- accessibilità a dimensione finita.

Un obiettivo predittivo può dunque identificare una classe di rappresentazioni
funzionalmente equivalenti senza identificarne la geometria euclidea. Implicit
bias dell’ottimizzatore, normalizzazione, weight decay e regolarizzatori
espliciti selezionano una particolare “gauge” dentro tale classe.

Questa osservazione fornisce un possibile ponte fra anti-allineamento,
co-adattamento e VICReg/SIGReg: la regolarizzazione non aggiunge necessariamente
nuovo contenuto, ma può scegliere una geometria diversa per contenuto già
presente.

### 10.4 Toy example minimo

Siano \(S\) il segnale predittivo, \(N\) un nuisance indipendente e

\[
Z=(aS,bN), \qquad Y=S.
\]

Per ogni \(a\neq0\), un decoder full-rank recupera \(Y\). Tuttavia la prima
componente principale segue il nuisance quando

\[
b^2\operatorname{Var}(N)
>
a^2\operatorname{Var}(S).
\]

Si ottengono quindi:

- stesso contenuto ideale;
- stessa informazione predittiva;
- diversa top-PCA accessibility;
- diverso condizionamento del decoder;
- possibile diversa sample efficiency con rumore, ridge e campioni finiti.

Il toy example dimostra la separazione concettuale; il progetto dovrà poi
mostrare quando questa geometria viene selezionata dagli obiettivi di training,
anziché inserirla artificialmente nel generatore.

### 10.5 Geometria e topologia non sono sinonimi

Una trasformazione lineare invertibile può modificare drasticamente spettro e
accessibilità pur essendo un omeomorfismo. Topologia, contenuto e geometria
euclidea sono quindi livelli differenti.

Gli strumenti topologici diventano pertinenti soltanto introducendo un processo
con struttura latente nota, per esempio \(S^1\), \(S^1\times S^1\) o un sistema
a regimi. Persistent homology o cohomology dovranno confrontare invarianti
ground truth, con null model e analisi di stabilità. Non verranno applicate
post-hoc al point cloud LOB come semplice visualizzazione.

---

## 11. Replica meccanicistica controllata

### 11.1 Livello A — processo lineare-gaussiano

Il primo generatore sarà minimale:

\[
S_{t+1}=F_SS_t+\eta_t,
\qquad
N_{t+1}=F_NN_t+\xi_t,
\]

\[
O_t=g(A_SS_t+A_NN_t+\varepsilon_t).
\]

Il target potrà essere \(S_{t+h}\), una sua proiezione o una funzione
multi-horizon. I parametri controllabili saranno:

- rapporto fra varianza predittiva e nuisance;
- persistenza temporale di \(S_t\) e \(N_t\);
- rapporto segnale-rumore;
- dimensione latente;
- osservabilità;
- orizzonte;
- mixing fra coordinate;
- eventuale non linearità \(g\).

Prima di introdurre reti profonde si dovranno:

1. calcolare o stimare il Bayes predictor;
2. verificare la statistica sufficiente;
3. derivare le covarianze di popolazione;
4. studiare encoder e predictor lineari;
5. identificare le simmetrie e le soluzioni equivalenti;
6. mostrare in quali regimi contenuto e accessibilità si separano.

### 11.2 Livello B — replica neurale

Sul medesimo processo verrà addestrata una piccola architettura condivisa con
tre obiettivi matched:

1. supervised;
2. horizon prediction;
3. masked prediction.

Capacità, dati, optimizer, update, seed e readout saranno controllati. Non serve
replicare subito la scala LOB: il primo obiettivo è identificare il meccanismo,
non ottenere prestazioni assolute.

Le quantità ground truth consentiranno di distinguere:

```text
informazione persa
    ≠ informazione conservata ma geometricamente nascosta
        ≠ informazione accessibile soltanto al decoder co-adattato
```

### 11.3 Livello C — estensione non lineare/topologica

Solo dopo aver compreso il caso lineare:

- diffusione su \(S^1\);
- dinamica su un toro;
- switching dynamical system;
- manifold latente con nuisance trasversali.

Qui la domanda sarà se l’obiettivo preserva componenti, cicli, coordinate
circolari e transizioni di regime. Questa è un’estensione autonoma: non deve
essere necessaria per salvare il claim geometrico principale.

### 11.4 Ritorno al caso reale

Il LOB diventa il test di validità ecologica:

> Il fenomeno identificato in un sistema con ground truth ricompare in un
> sistema reale nel quale lo stato latente non è osservabile?

Il simulatore dimostra la possibilità e isola un meccanismo; non dimostra da
solo che quel medesimo meccanismo sia la causa dei risultati LOB. Per sostenere
quest’ultimo passaggio servirà un intervento trasferito al training reale.

---

## 12. Intervento causale sulla geometria

### 12.1 Ordine degli interventi

L’ordine raccomandato è:

1. baseline predittivo matched;
2. regularizer VICReg-style;
3. SIGReg dentro il JEPA esistente;
4. soltanto dopo, eventuale LeJEPA completo;
5. masked come test di generalizzazione del meccanismo.

VICReg modifica esplicitamente varianza e covarianza. SIGReg aggiunge un
vincolo distribuzionale più forte mediante proiezioni casuali. Il LeJEPA
completo cambia anche teacher, predictor e struttura della loss e non è quindi
un’ablation pulita dell’isotropia.

### 12.2 Primo confronto causale

| arm | loss predittiva | EMA/predictor | regolarizzatore |
|---|---|---|---|
| H0 | horizon corrente | invariati | nessuno |
| H-VIC | horizon corrente | invariati | variance + covariance |
| H-SIG | horizon corrente | invariati | SIGReg |

Il confronto viene eseguito prima nel simulatore e poi, se informativo, sul
LOB. Tutti gli arm devono condividere split, seed, numero di update, batch,
backbone, predictor, schedule EMA e optimizer.

Nel retraining LOB si deve inoltre usare lo stesso no-weight-decay per bias e
parametri di normalizzazione in tutti gli arm. Questa modifica richiede un H0
nuovo e impedisce il confronto causale diretto con il vecchio checkpoint.

### 12.3 Locus della regolarizzazione

Per una griglia token-preserving, la scelta primaria rimane:

```text
L_reg = media su (k,s) di Reg(Z[:, k, s, :])
```

La regolarizzazione viene quindi applicata attraverso il batch a ciascuna
posizione separatamente. Non si regolarizza direttamente `meanK_concatS`,
perché questo hard-coderebbe nel training proprio il readout usato per la
misura.

Per SIGReg il costo del calcolo su tutte le posizioni dovrà essere misurato. Un
eventuale sampling di posizioni deve essere target-blind, deterministico dato
il seed e coperto da log.

### 12.4 Selezione degli iperparametri

Il peso del regolarizzatore non deve essere scelto sul downstream direzionale.
La selezione avviene su un Pareto target-blind:

- prediction loss di validation;
- manipulation check geometrico;
- assenza di collapse;
- stabilità;
- costo computazionale.

Il downstream viene aperto soltanto dopo aver fissato il protocollo.

### 12.5 Criteri di successo

Un intervento è meccanicisticamente informativo se:

1. modifica la geometria nella direzione prevista;
2. preserva il contenuto rispetto al Bayes risk nel simulatore;
3. migliora accessibilità o sample efficiency;
4. riduce il gap dal null casuale quando tale confronto è identificabile;
5. non distrugge struttura relazionale o temporale;
6. riduce, oppure spiega, il co-adaptation gap;
7. sopravvive a seed e regimi generativi differenti.

Se la covariance diventa quasi isotropa, le top PCA non sono identificabili. In
quel regime le metriche principali diventano:

- distribuzione su sottospazi casuali;
- finite-sample ridge/OLS;
- condizionamento;
- rischio medio e worst-case dei reader;
- stabilità cross-seed.

Non è corretto richiedere semplicemente che “le top PCA diventino supervised”.

---

## 13. Ipotesi falsificabili

### H1 — objective-dependent allocation

A parità di processo osservato e capacità, gli obiettivi producono differenze
sistematiche nella collocazione del contenuto rispetto a varianza, token e
tempo.

### H2 — content/accessibility separation

Esistono regimi in cui due encoder raggiungono contenuto o rischio comparabile,
ma differiscono materialmente in \(A(m,n)\), condizionamento e stabilità del
readout.

### H3 — geometric intervention

Un regolarizzatore può modificare accessibilità e sample efficiency senza
aumentare il contenuto ideale. Se migliora soltanto riducendo il contenuto o
riscrivendo il pooling, non conferma l’ipotesi.

### H4 — co-adaptation

Una parte del vantaggio del decoder nativo deriva dalla scelta congiunta della
gauge rappresentazionale. La parte residua deve essere separata da differenze
reali di contenuto.

### H5 — topology, opzionale

Su processi con topologia latente nota, obiettivi differenti possono preservare
il contenuto predittivo locale senza preservare allo stesso modo la struttura
globale.

### Esiti negativi informativi

- Se gli obiettivi matched convergono alla stessa geometria, il fenomeno LOB è
  probabilmente architetturale o domain-specific.
- Se il gap di accessibilità scompare controllando il contenuto, non serve una
  teoria geometrica aggiuntiva.
- Se VICReg/SIGReg modificano lo spettro ma non sample efficiency, l’anisotropia
  non è una causa sufficiente.
- Se il simulatore riproduce il fenomeno soltanto inserendolo manualmente nei
  parametri, la replica non è meccanicistica.
- Se gli invarianti topologici non sono stabili a sampling e rumore,
  l’estensione topologica va rimossa.

---

## 14. Architettura consigliata della tesi e del possibile paper

### Study I — scoperta empirica

Il consolidamento LOB già completato:

- supervised, horizon e masked;
- contenuto, pooling e accessibilità;
- geometria token-temporale;
- held-out e co-adattamento;
- protocollo P0 riproducibile.

### Study II — formalizzazione

- non-identificabilità della geometria sotto trasformazioni invertibili;
- definizioni di content, accessibility, alignment e co-adaptation;
- toy example e risultati lineari;
- condizioni sotto cui PCA accessibility diverge dal contenuto.

### Study III — replica controllata

- processo stocastico con ground truth;
- confronto matched tra obiettivi;
- sweep dei parametri generativi;
- misura rispetto a Bayes risk e stato latente.

### Study IV — intervento

- H0/H-VIC/H-SIG;
- effetto causale su geometria, sample efficiency e co-adattamento;
- eventuale trasferimento sul LOB;
- topologia soltanto come estensione motivata.

Titolo di lavoro:

> **Geometry and Accessibility of Predictive Information in Learned
> Representations of Stochastic Dynamical Systems**

Una variante più meccanicistica:

> **How Predictive Learning Organizes Latent Information: Geometry,
> Accessibility and Co-adaptation**

---

## 15. Valore attuale e ruolo del relatore

La tesi è già difendibile senza il nuovo programma:

- bug metodologico identificato e corretto;
- protocollo fail-closed;
- distinzione contenuto/pooling/accessibilità;
- multiseed encoder e reader;
- controlli held-out;
- risultato geometrico netto e non banale.

Il nuovo lavoro non serve a salvare la tesi, ma a trasformare:

```text
fenomeno descrittivo ben caratterizzato
```

in:

```text
meccanismo formalizzato, replicato e manipolato causalmente
```

Il relatore cercato non deve essere prioritariamente un esperto di
microstruttura. Il profilo ideale copre almeno due fra:

- processi stocastici e sistemi dinamici;
- statistica multivariata e metodi spettrali;
- geometria dei dati o topological data analysis;
- teoria del representation learning;
- inverse problems e latent-variable models.

Il suo contributo sostanziale dovrebbe riguardare:

1. identificabilità degli oggetti geometrici;
2. invarianti rispetto a cambi di coordinate;
3. separazione fra risultati dimostrabili ed evidenza empirica;
4. disegno del processo controllato;
5. delimitazione dello scope.

---

## 16. Limiti da mantenere espliciti

- Lo studio reale riguarda un solo dominio, una scala temporale, sette titoli e
  `S=4`.
- Le finestre LOB si sovrappongono; assunzioni i.i.d. non si trasferiscono
  letteralmente.
- I target held-out sono indipendenti soltanto nel senso verificato dal Gate 1.
- Il timing è capped, non survival.
- L’MLP è un ceiling operativo sul readout, non una misura dell’informazione
  totale.
- L’accessibilità è intenzionalmente dipendente da metrica, coordinate, classe
  di reader e budget campionario; non è un invariante della rappresentazione.
- Un simulatore può dimostrare un meccanismo possibile, non la causa reale dei
  risultati LOB.
- Il generatore non deve incorporare per costruzione il risultato che si vuole
  osservare.
- Isotropia e task alignment non coincidono; l’isotropia può essere
  subottimale per target specifici.
- Una covariance isotropa rende instabile l’ordinamento PCA.
- Il vecchio weight decay richiede un baseline matched per qualsiasi nuovo
  retraining.
- L’estensione topologica è opzionale e deve essere eliminata se non produce
  invarianti stabili e interpretabili.
- Un caso reale più un simulatore non autorizzano claim di universalità.

---

## 17. Sorgenti e artefatti

### Stato e risultati locali

- `CONSOLIDATION_20260728.md`
- `validation/readouts_v2_20260728/analysis_manifest.json`
- `validation/readouts_v2_20260728/analysis_consolidation_20260728/`
- `validation/readouts_v2_20260728/analysis_consolidation_20260728/mlp_agg.csv`
- `validation/readouts_v2_20260728/analysis_consolidation_20260728/principal_angles.csv`
- `validation/readouts_v2_20260728/analysis_consolidation_20260728/diagnostics/`

### Fonti primarie già identificate

- Bardes, Ponce, LeCun — [VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), ICLR 2022.
- Balestriero, LeCun — [LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics](https://arxiv.org/abs/2511.08544), arXiv 2025.
- Assran et al. — [Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.
- Sobal et al. — [Joint Embedding Predictive Architectures Focus on Slow
  Features](https://arxiv.org/abs/2211.10831), 2022.
- Implementazione ufficiale —
  [galilai-group/lejepa](https://github.com/galilai-group/lejepa).

### Letteratura da mappare prima di fissare la novità

- identifiability e gauge freedom delle rappresentazioni;
- linear self-supervised learning e reduced-rank prediction;
- representation similarity, CCA e geometria dei sottospazi;
- predictive state representations e sufficient statistics;
- representation learning per sistemi dinamici;
- persistent homology/cohomology di dinamiche latenti;
- sample efficiency e conditioning dei linear probe.

---

## 18. Formula breve per una discussione esterna

> In un sistema reale abbiamo osservato che differenti obiettivi possono
> conservare informazione predittiva ma collocarla in modi molto diversi
> rispetto a varianza, token e tempo. Il supervised produce un codice
> task-aligned; JEPA-horizon conserva segnale soprattutto in direzioni a bassa
> varianza e nei contrasti fra token. Vogliamo formalizzare la distinzione fra
> contenuto, accessibilità e co-adattamento, osservando che il rischio predittivo
> non identifica in generale la geometria interna. Costruiremo quindi un
> processo stocastico con stato latente e rischio di Bayes noti, replicheremo il
> fenomeno e interverremo causalmente con regolarizzazione geometrica. Il LOB
> rimarrà il caso reale di scoperta e validità ecologica.

---

## 19. Cosa eseguire prima di contattare il professore

L’obiettivo non è completare il nuovo progetto prima del confronto. È arrivare
con abbastanza sostanza da rendere la discussione scientifica, lasciando ancora
aperte le decisioni sulle quali il professore può incidere.

### A. Congelare lo stato empirico

- considerare chiuso il consolidamento LOB;
- non lanciare nuovi training LOB;
- verificare che questo documento, manifest e commit siano coerenti;
- selezionare tre figure decisive:
  1. MLP content `last` contro `meanK`;
  2. ladder top-PCA contro null casuale;
  3. Hadamard/common-contrast o held-out co-adaptation.

**Deliverable:** snapshot tecnico riproducibile e tre figure leggibili senza
aprire il codice.

### B. Estrarre una research note di 3–5 pagine

La nota deve contenere:

1. fenomeno osservato;
2. domanda generale;
3. quattro definizioni operative;
4. non-identificabilità sotto \(Z'=TZ\);
5. toy example \(Z=(aS,bN)\);
6. modello stocastico proposto;
7. ipotesi H1–H4;
8. limiti e possibili risultati negativi.

**Deliverable:** documento autonomo, comprensibile a un matematico che non
conosce il LOB.

### C. Completare il pacchetto offline sui dump esistenti

Senza retraining:

1. angoli e sottospazi covariance-weighted;
2. curve finite-sample ridge/OLS per
   `n = 256, 1k, 4k, 16k, 100k`;
3. whitening diagnostico train-fit;
4. confronto fra top-PCA, sottospazi casuali e coordinate whitened;
5. incertezza su seed encoder e subsample.

**Deliverable:** una figura di sample efficiency e una tabella che colleghi
contenuto, allineamento e accessibilità. Questo chiude l’ultima ambiguità
geometrica rimasta nei dati reali.

### D. Implementare soltanto il prototipo lineare-gaussiano

Prima del contatto è sufficiente:

- generatore configurabile e deterministico;
- oracle/Bayes predictor;
- covarianze teoriche o numeriche validate;
- toy encoder lineari;
- sweep minimo varianza nuisance/segnale;
- una figura che mostri stesso contenuto e diversa accessibilità;
- test che evitino leakage e confermino la statistica sufficiente.

Non servono ancora Transformer, VICReg, SIGReg o persistent homology.

**Deliverable:** script riproducibile e una figura meccanicistica minima.

### E. Preparare una mappa della letteratura

Non una review esaustiva, ma una tabella con:

- lavoro;
- oggetto matematico;
- claim;
- assunzioni;
- relazione con il nostro fenomeno;
- differenza residua.

Le aree prioritarie sono identifiability/gauge, linear SSL, predictive state,
representation geometry e sample efficiency. La topologia viene mappata in una
seconda pagina separata.

**Deliverable:** una pagina che permetta di non presentare come nuova
un’osservazione già nota con un altro vocabolario.

### F. Preparare il pacchetto per il professore

- abstract di una pagina;
- research note;
- documento tecnico completo;
- tre figure empiriche e una sintetica;
- repository/commit riproducibile;
- cinque domande precise sulle quali chiedere contributo;
- elenco esplicito di ciò che è fuori scope.

Le cinque domande consigliate sono:

1. La non-identificabilità proposta è formulata nel modo corretto?
2. Quale nozione di accessibilità è matematicamente più difendibile?
3. Quale famiglia di processi è minimale ma non banale?
4. Quali risultati possono essere analitici e quali soltanto sperimentali?
5. L’estensione topologica aggiunge un contributo reale oppure disperde il
   progetto?

### Definition of done prima del contatto

Si contatta il professore quando sono disponibili:

- stato LOB congelato;
- research note;
- toy proposition con derivazione;
- prototipo lineare-gaussiano validato;
- una figura sintetica;
- prima mappa della letteratura;
- domande scientifiche precise.

Non si aspetta il completamento della replica neurale. Non si eseguono prima
del confronto:

- full LeJEPA;
- sweep H-VIC/H-SIG sul LOB;
- simulatori realistici di mercato;
- persistent homology sui dati reali;
- estensioni architetturali non necessarie.

Questo punto di arresto rende il progetto già credibile, ma lascia al
professore spazio reale per contribuire al formalismo, al simulatore e alla
delimitazione della tesi.
