# Experiment 01 — Mappa delle affermazioni di Phase II

## 1. Domanda scientifica

Phase I mostra un gap di accessibilità lineare finite-sample tra supervised e
horizon-JEPA. Phase II chiede **dove**, lungo lo spettro di covarianza della
rappresentazione, si trovi il segnale utile ai target.

La domanda non è se l'encoder contenga informazione in senso assoluto. È se
l'informazione linearmente associata ai target sia allineata alle direzioni di
alta varianza, oppure distribuita più in profondità nello spettro.

## 2. Oggetto osservato

Per ogni ramo, encoder seed e readout si calcola la PCA sulle sole feature del
train non etichettate. Le direzioni sono ordinate dalla varianza maggiore alla
minore. Il readout primario è `last_concat512`; il rango numerico valido è 508.

Per il target `r` e la direzione PCA `j`, la statistica è:

```text
m[j,r] = (u_j^T Cov(Z,Y_r))^2 / (lambda_j * Var(Y_r))
```

La curva mostrata nel report è la frazione cumulativa:

```text
M_r(k) = sum_{j<=k} m[j,r] / sum_{j<=D_valid} m[j,r]
```

`M_r(k)` è quindi la quota della massa predittiva lineare train che cade nelle
prime `k` direzioni di massima varianza. Non è un R² out-of-sample e vale 1 a
`D_valid` per costruzione.

I riepiloghi usano i target indipendenti: 12 direzionali, 2 di volatilità e 1
di timing. Gli intervalli risamplano i tre encoder seed e i target entro seed;
misurano robustezza computazionale, non incertezza di popolazione.

## 3. Claim 1 — anti-allineamento spettrale direzionale

### Test

Confrontare la massa cumulativa nelle prime PC fra horizon-JEPA e supervised,
tenendo separati i tre blocchi target.

### Risultato

Su `last_concat512`:

| ramo | M directional(8) | M directional(16) | M directional(128) | M directional(256) |
| --- | ---: | ---: | ---: | ---: |
| horizon-JEPA | 0.0001 | 0.0064 | 0.6458 | 0.8330 |
| supervised | 0.7518 | 0.8742 | 0.9748 | 0.9903 |

Nelle prime 8 PC horizon-JEPA colloca circa lo 0,009% della massa direzionale,
contro il 75,18% del supervised. La differenza non equivale a dire che il
segnale direzionale manca: dice che, in horizon-JEPA, non è allineato agli assi
di varianza dominante.

### Specificità

Per horizon-JEPA a `k=8`, la massa è 0.5601 per volatilità e 0.8141 per timing.
L'estremità del fenomeno è quindi specifica del blocco direzionale nel dataset
e nei target studiati.

### Stato del claim

**Pass descrittivo forte.** L'effetto è molto grande e replicato sui tre seed,
ma non dispone di resampling per stock o stock-day e non stabilisce causalità.

## 4. Claim 2 — le prime PC di horizon-JEPA sono peggiori di un sottospazio casuale

### Test

Per ogni `k`, il test R² del sottospazio delle prime `k` PC è confrontato con
100 sottospazi Haar casuali della stessa dimensione. Il reader è lo stesso
min-norm OLS diagnostico e usa il train completo.

Questo null controlla la **dimensione del sottospazio**. Non controlla la quota
di varianza, che è precisamente ciò rispetto a cui le prime PC sono speciali.

### Risultato

Per horizon-JEPA direzionale:

| k | top-PCA R² | Haar R² medio | esito sui tre seed |
| ---: | ---: | ---: | --- |
| 8 | 0.0000 | 0.0024 | 100/100 Haar sopra top-PCA in ogni seed |
| 16 | 0.0014 | 0.0093 | 100/100 Haar sopra top-PCA in ogni seed |
| 32 | 0.0237 | 0.0300 | transizione eterogenea |
| 64 | 0.0773 | 0.0706 | transizione eterogenea |
| 128 | 0.1538 | 0.1271 | top-PCA sopra tutti i Haar in ogni seed |
| 256 | 0.1917 | 0.1806 | top-PCA sopra tutti i Haar in ogni seed |

Per supervised, top-PCA è sopra tutti i 100 sottospazi casuali a tutte le
profondità 8–256 e in tutti i seed.

Le quantità 0 e 1 conservate negli artefatti sono **frazioni di superamento**
su 100 draw, non p-value continui. La risoluzione è 0.01; non sono usate per una
decisione inferenziale preregistrata.

### Stato del claim

**Pass descrittivo.** Alle profondità 8 e 16 gli assi di massima varianza di
horizon-JEPA sono sistematicamente meno utili di una direzione casuale
dimension-matched. La transizione 32–64 non è uniforme fra seed.

## 5. Claim 3 — coerenza con il whitening profondo di Phase I

Phase I mostra che il gap direzionale viene ridotto del 55,6% a `k=128` e del
92,6% a `k=508`. A `k=508` i lower bound del gap restano positivi: viene meno il
criterio composto `lower > 0 and mean >= delta=0.10`, non la separazione da
zero.

Phase II mostra che horizon-JEPA ha recuperato soltanto il 64,6% della massa
direzionale entro `k=128` e l'83,3% entro `k=256`, mentre supervised è già al
97,5% e 99,0%.

Le due curve sono coerenti con la stessa storia: il segnale direzionale
horizon-JEPA è distribuito in profondità e un reader regolarizzato isotropico
lo usa con difficoltà a basso budget.

### Limite logico

Questo è un **ponte descrittivo, non una mediazione causale**. La ladder PCA
tronca il sottospazio; il whitening di Phase I mantiene il rango e riscala gli
assi. La loro concordanza non dimostra che la massa predittiva sia la causa
unica del gap.

## 6. Claim 4 — la non-monotonia 8, 16, 32, 64 non è stata localizzata

L'ipotesi storica confrontava la banda `17:32` con `33:64`. Il confronto non è
dimension-matched: la prima ha 16 direzioni, la seconda 32. Un R² band-only
maggiore nella seconda può quindi derivare semplicemente dalla dimensione.

I null Haar matched separati per ciascuna banda e la massa per direzione sono
diagnostiche utili, ma non trasformano la differenza fra le due bande in un
contrasto paired valido.

### Stato del claim

**Non verificato.** Phase II non identifica la specifica banda responsabile
della non-monotonia osservata in Phase I. Questo non modifica A1.

## 7. Cosa Phase II consente di dire

La formulazione più forte difendibile è:

> Nel readout `last_concat512`, l'associazione lineare con i target direzionali
> di horizon-JEPA è fortemente esclusa dalle direzioni di massima varianza più
> superficiali e distribuita più in profondità dello spettro rispetto al
> supervised. A `k=8` e `k=16`, top-PCA è anche peggiore dei sottospazi Haar
> dimension-matched in tutti i seed. Il pattern è specifico rispetto ai
> controlli di volatilità e timing ed è coerente, ma non causalmente
> identificato, con la necessità di whitening profondo osservata in Phase I.

Non consente di affermare che:

- horizon-JEPA contenga meno informazione totale in senso generale;
- la predictive mass train sia identica al test R²;
- una singola banda spettrale causi la non-monotonia;
- il whitening dimostri causalmente il meccanismo;
- gli intervalli descrivano una popolazione di titoli o giornate.

## 8. Artefatti sorgente

- `validation/experiment01/execution_20260730/phase2/predictive_mass.parquet`
- `validation/experiment01/execution_20260730/phase2/random_subspace_null.parquet`
- `validation/experiment01/execution_20260730/phase2/spectral_bands.parquet`
- `validation/experiment01/execution_20260730/phase2/phase1_phase2_bridge.parquet`
- `validation/experiment01/execution_20260730/phase2/summary.json`
- `validation/experiment01/execution_20260730/phase2/metadata.json`

Nessun fit, soglia o risultato di Phase I è modificato da questa mappa.
