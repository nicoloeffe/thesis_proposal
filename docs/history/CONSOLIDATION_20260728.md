# Consolidamento post-P0 — 2026-07-28

## Stato e perimetro

Il pacchetto è stato eseguito sui nove dump canonici ep20 in
`validation/readouts_v2_20260728`, senza retraining e senza modificare file in
`training/`. Il readout principale è `meanK_concatS`, alias esplicito del
`tmean_concat512` già estratto; tutte le varianti a 128/384 dimensioni sono
trasformazioni offline, fisse e target-blind.

Il timing held-out usa ora un solo estimand in screening e artefatto:
`log1p(observed_or_capped_duration)` su tutte le righe, con cap 600. Il loop non
può più salvare il sentinel 601. La censura osservata è 0,1% e Gate 1 dà
`R²(time | 14 target addestrati) = 0,1817`, quindi il target resta ammissibile.

## Reader MLP multiseed

Ogni cella usa 3 seed encoder × 5 seed reader, split interno fisso derivato da
`split_seed=0`, early stopping con patience 10 e massimo 80 epoche. Le deviazioni
encoder e reader sono riportate separatamente; `totale` è la loro composizione
quadratica.

| pooling | braccio | R² direzionale | std encoder | std reader | std totale |
|---|---|---:|---:|---:|---:|
| `last_concat512` | jepa_horizon | 0,3191 | 0,0064 | 0,0045 | 0,0078 |
| `last_concat512` | jepa_masked | 0,1501 | 0,0125 | 0,0037 | 0,0130 |
| `last_concat512` | supervised | 0,3881 | 0,0004 | 0,0037 | 0,0037 |
| `meanK_concatS` | jepa_horizon | 0,0494 | 0,0103 | 0,0039 | 0,0110 |
| `meanK_concatS` | jepa_masked | 0,0008 | 0,0007 | 0,0009 | 0,0011 |
| `meanK_concatS` | supervised | 0,3917 | 0,0006 | 0,0029 | 0,0030 |

Il precedente claim “JEPA conserva circa l’82% del contenuto supervised” è
confermato **solo** per `last_concat512`: `0,3191 / 0,3881 = 82,23%`. Non è una
proprietà del contenuto indipendente dal readout: con `meanK_concatS` il rapporto
scende a `12,62%`. La media temporale conserva il supervised e quasi annulla il
segnale direzionale non lineare dei due JEPA.

Sui target held-out, il quadro dipende meno dal supervised:

| pooling | blocco | jepa_horizon | supervised |
|---|---|---:|---:|
| `last_concat512` | imbalance | 0,2015 | 0,1925 |
| `last_concat512` | depth | 0,1740 | 0,1621 |
| `last_concat512` | timing | 0,5041 | 0,5664 |
| `meanK_concatS` | imbalance | 0,0939 | 0,1218 |
| `meanK_concatS` | depth | 0,0862 | 0,0936 |
| `meanK_concatS` | timing | 0,4781 | 0,5699 |

Su depth/imbalance il vantaggio supervised collassa o si inverte con il readout
last, coerentemente con una componente importante di co-adattamento. Il timing
mantiene invece un vantaggio supervised di circa 0,06–0,09.

## Ladder normalizzata, null e Hadamard

La frazione è calcolata come rapporto tra le medie R² del blocco, non come media
dei rapporti per target. Un denominatore full-rank medio sotto 0,01 viene
marcato non affidabile.

Al punto comune `m/D = 1/32`:

| readout | braccio | R²(m)/R²(D) | null numerico |
|---|---|---:|---:|
| `last_concat512` | jepa_horizon | 0,0050 | 0,0563 |
| `last_concat512` | jepa_masked | 0,0022 | 0,0488 |
| `last_concat512` | supervised | 0,8971 | 0,6118 |
| `meanK_concatS` | jepa_horizon | 0,0042 | 0,0530 |
| `meanK_concatS` | jepa_masked | 0,0232* | 0,1956 |
| `meanK_concatS` | supervised | 0,9777 | 0,7494 |

`*` denominatore full-rank `R²=0,0041`, quindi non interpretabile.

I JEPA sono sotto il null numerico: le prime componenti di varianza sono
anti-allineate al segnale direzionale. Il supervised è sopra il null e recupera
quasi tutto il contenuto nelle prime 16 componenti.

Il null numerico non coincide con la diagonale analitica `m/D=3,125%` sui dati
reali. Non è un errore di centratura: i readout sono fortemente anisotropi e
spesso hanno participation ratio 7–9. Un sottospazio casuale di dimensione 16
può quindi intercettare gran parte dello spazio effettivamente occupato. Il test
isotropo sintetico restituisce correttamente `m/D`; nei grafici reali vanno
mostrati sia la diagonale analitica sia il null numerico.

La decomposizione Hadamard mostra, per il supervised:

| base | sottoblocco | R²(m)/R²(D) a 1/32 | R²(D) |
|---|---|---:|---:|
| last | media 128 | 0,7185 | 0,3730 |
| last | contrasti 384 | 0,3984 | 0,3331 |
| meanK | media 128 | 0,7918 | 0,3892 |
| meanK | contrasti 384 | 0,1672 | 0,3073 |

L’accessibilità supervised a basso rango è più forte nella componente comune
che nei contrasti, soprattutto dopo la media temporale. I contrasti conservano
comunque contenuto full-rank. `mean_all128` è, a scala costante, lo stesso
sottospazio di `meanK_hadamard_mean128` e non costituisce evidenza indipendente.

## Angoli, rango e diagnostici

Gli angoli principali sono calcolati sui blocchi direzionale, volatilità e
timing. Tutti i 1.782 punti della griglia superano il gap relativo `1e-3`; a
`m=D` l’energia allineata è 1 entro tolleranza. A `m=16` l’energia euclidea
allineata al sottospazio dei coefficienti è molto piccola (`~10⁻⁶–10⁻⁴`):
questo osservabile non va equiparato direttamente alla quota di R², perché non
è pesato dalla covarianza né dalla forza predittiva delle direzioni.

Participation ratio / effective rank medi:

| pooling | braccio | participation ratio | effective rank |
|---|---|---:|---:|
| last | jepa_horizon | 7,91 | 11,75 |
| last | jepa_masked | 45,31 | 88,69 |
| last | supervised | 7,05 | 14,46 |
| meanK | jepa_horizon | 7,32 | 10,22 |
| meanK | jepa_masked | 43,29 | 82,64 |
| meanK | supervised | 8,58 | 15,05 |

L’attenzione supervised è diffusa: entropia normalizzata media
`0,975–0,992`; il peso totale dell’ultimo timestep è `5,27–6,93%`, vicino al
5% uniforme. Lo screening temporale conferma che il supervised recupera
`97,4–98,6%` del proprio R² full-rank a `m=16` nelle posizioni 1–19, con una
flessione al 90,1% in posizione 20. JEPA horizon resta vicino a zero a tutte le
posizioni; per JEPA masked solo 3/20 posizioni hanno un denominatore full-rank
almeno 0,01.

I pesi `|gamma|` di `final_norm` hanno media 0,975 (horizon), 1,000 (masked) e
0,987 (supervised). Non emerge un riscalamento affine grossolano capace da solo
di spiegare la geometria, pur restando aperto il limite metodologico del weight
decay sui parametri di normalizzazione.

## Verifica e artefatti

- Test suite: `61 passed`.
- Preflight finale: 9/9 dump, manifest stage-1 `complete`, held-out coerente.
- Hash di tutti gli output e dei sorgenti verificati contro i manifest.
- Nessun file di training modificato.

Output principali:

- `validation/readouts_v2_20260728/analysis_consolidation_20260728/analysis_manifest.json`
- `ladder_long.csv`, `ladder_fraction_agg.csv`
- `refs_long.csv`, `mlp_reader_runs.csv`, `mlp_agg.csv`
- `random_subspace_null.csv`, `principal_angles.csv`
- `spectral_diagnostics.csv`
- `diagnostics/final_norm_gamma.csv`
- `diagnostics/supervised_attention_*.csv`
- `diagnostics/temporal_screen.csv`

Limiti residui: il timing è un target capped, non un modello survival; la
correzione del weight decay di `final_norm` richiede retraining ed è fuori
scope; il concat completo da 10.240 dimensioni e la regolarizzazione di
covarianza restano esperimenti separati.
