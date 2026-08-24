# Audit pre-implementazione — Experiment 01

> Documento storico del 2026-07-30. Descrive lo stato e i percorsi disponibili
> al momento della costruzione del bundle. La directory top-level `legacy/` e i
> CSV raw che conteneva sono stati rimossi il 2026-08-24. I sette CSV canonici
> sono stati poi ripristinati sotto `data/lobench/raw/`, con hash invariati; per
> lo stato operativo corrente fare riferimento a
> [`PROJECT_STATE.md`](../../PROJECT_STATE.md).

Data audit: 2026-07-30  
Repository: `/home/nicolo/Deep_Learning/thesis`  
Commit sorgente degli artefatti: `17c5ffdba1cb4c2e762856771cd6a6a507520bda`  
Commit corrente: `6a94bd5` (`a1-v2-sequence-aware`, working tree non pulita)  
Specifica originale (allora denominata con suffisso `(1)`):
[`SPEC_EXPERIMENT_01_SAMPLE_EFFICIENCY_20260730.md`](SPEC_EXPERIMENT_01_SAMPLE_EFFICIENCY_20260730.md),
versione 2.0

## Esito

Gli artefatti in `validation/readouts_v2_20260728` sono i nove dump canonici
corretti post-P0 e superano integralmente il validatore fail-closed v2
(`validate_stage1_inputs`: 9/9 dump, hash file, fingerprint, schema, shape,
dtype, checkpoint, split e ordine endpoint).

Da soli non costituiscono però un input sufficiente per eseguire Experiment 01
v2.0:

1. hanno soltanto `train` e `val`, non tre split distinti
   `train/validation/test`;
2. `train_t` è un campione casuale senza rimpiazzo di 100.000 endpoint da
   7.323.510 endpoint validi, non l'insieme completo e contiguo delle sequenze
   endpoint dei giorni di training;
3. il dataset processato non serializza timestamp, simboli originali o una
   tabella `stock_id -> simbolo`, ma questi metadati sono ricostruibili
   deterministicamente dai sette CSV sorgente presenti nel repository;
4. non contengono un test set separato né consentono di costruire dai soli dump
   i blocchi frazionari contigui richiesti dalla specifica.

L'implementazione di Experiment 01 deve pertanto accettare soltanto un nuovo
bundle post-P0 a tre split, con endpoint completi e ordinati per stock-day,
timestamp/row key stabili e hash dichiarati. Il preflight deve rifiutare i dump
v2 correnti per l'esecuzione del grid, pur usandoli per il gate gratuito di
riproduzione del risultato storico.

La correzione del task preregistra ora il terzo split: tutti i giorni del train
storico restano train; per titolo, la prima metà cronologica dei giorni
held-out diventa validation e la seconda metà test, con l'eventuale giorno
dispari nel test. Questa regola risolve il precedente blocco di protocollo
senza spostare nel test alcun giorno usato per addestrare gli encoder.

Il sidecar e il gate risultanti sono in
`validation/experiment01_inputs_20260730/sidecar`; il manifest dello split è in
`validation/experiment01_inputs_20260730/split3/split_manifest.json`.

## 0. Provenienza e ricostruzione del dataset

Il builder è:

`scripts/dataset/build_encoder_dataset_lobench.py`

e i sette sorgenti si trovano in:

`legacy/data/lobench/raw/sz*_processed.csv` (percorso storico, non più presente)

Il builder:

1. ordina lessicograficamente i file e assegna `stock_id=0..6`;
2. scarta le righe con orario `>=14:57`;
3. scarta i book collassati
   (`BidPrice1==BidPrice10` o `AskPrice1==AskPrice10`, tolleranza `1e-6`);
4. usa stride 1 per il dataset completo;
5. costruisce il book best-first e calcola `mid_z`;
6. assegna `day_id = dayofyear - min(dayofyear)` per titolo;
7. concatena i titoli nell'ordine dei file.

Il sidecar conserva `raw_csv_row_index` come indice zero-based della riga dati
nel CSV originale, esclusa l'intestazione e prima dei filtri.
`endpoint_order` è invece ricalcolato dopo i filtri ed è consecutivo
`0..n_day-1` entro `(stock_id, trading_date)`.

La ricostruzione è stata verificata contro
`data/lobench_processed.npz`: tutti gli 8.039.246 record sono contabilizzati;
per ciascun titolo coincidono il conteggio post-filtro, il primo e l'ultimo
book, il primo `day_id` e l'ultimo `day_id`.

| stock_id | simbolo | righe post-filtro | primo timestamp | ultimo timestamp |
|---:|---|---:|---|---|
| 0 | `sz000001` | 1.156.650 | 2019-01-02 09:30:03 | 2019-12-31 14:56:57 |
| 1 | `sz000002` | 1.156.649 | 2019-01-02 09:30:03 | 2019-12-31 14:56:57 |
| 2 | `sz000651` | 1.122.630 | 2019-01-02 09:30:00 | 2019-12-31 14:56:57 |
| 3 | `sz000858` | 1.156.679 | 2019-01-02 09:30:03 | 2019-12-31 14:56:57 |
| 4 | `sz002415` | 1.147.303 | 2019-01-02 09:30:00 | 2019-12-31 14:56:57 |
| 5 | `sz300147` | 1.142.775 | 2019-01-02 09:30:03 | 2019-12-31 13:01:18 |
| 6 | `sz300750` | 1.156.560 | 2019-01-02 09:30:03 | 2019-12-31 14:56:57 |

Hash SHA-256 dei CSV, nello stesso ordine:

| simbolo | SHA-256 |
|---|---|
| `sz000001` | `cfc88e926c06b87f7e82506ec0973d07afde838d1b949353c21a6c7ab049842b` |
| `sz000002` | `eaf43ffda67970fb467e38fdc0984784a94c2e141f1e90c9525d18fef77e3465` |
| `sz000651` | `527e082a61f30f42e4ce5ec117cb2d99f42b3eeb6798de4f8237a9d2b14fea59` |
| `sz000858` | `d9ad8f2f341e3868c59bcc1e382e761038ea3ddb86c1f28c89c59f8ef136b14f` |
| `sz002415` | `2c801af4e923e3abf1bc2fec35ddbc9289027e9ceb2f95d17d61975cba60073a` |
| `sz300147` | `60bfb8fee288b028f773b389066696ed18878d3e1c26ffeffdd9636738f97062` |
| `sz300750` | `7ed3d0b250871c19fb5829a4270777c028a6847845e264dfbef9541bf25ac938` |

## 1. Artefatti canonici post-P0

Root canonica:

`/home/nicolo/Deep_Learning/thesis/validation/readouts_v2_20260728`

Dataset sorgente:

| path | SHA-256 | byte |
|---|---|---:|
| `/home/nicolo/Deep_Learning/thesis/data/lobench_processed.npz` | `7617dbbfcee56377f980a606267397861f6613017f0a2aca1e218407726ef862` | 169857435 |

Artefatti comuni:

| path relativo | SHA-256 | contenuto |
|---|---|---|
| `analysis_manifest.json` | verificato dal validatore v2 | inventario completo |
| `split.npz` | `0c5149c1260c153c8bdbe3ac8a453750816b4ef62eaa6b54ac03ffb396245cc3` | 7.323.510 endpoint validi, campioni train/val |
| `targets_shared.npz` | `f2ab87577875e8c535d9e7ebdd4b60df991f20c2564cb9c8d57aeaa9ac9e9ac9` | 22 target train/val |
| `targets_heldout.npz` | `0175786ec804415ffc6342bc5c6a1d90701d29bac30bd0d535b9635c5aeb13ef` | 17 target held-out train/val |

Dump di feature:

| branch/seed | path relativo | SHA-256 dump | SHA-256 checkpoint |
|---|---|---|---|
| `jepa_horizon/0` | `readouts/jepa_horizon_seed0_ep020.npz` | `5e20db54bc9bc9232839607f3242c9b69cfa97bd0c738241244d45d8023312f4` | `756aa9dfd88b65eb5cfabca8e2d93c6fefa52994e39ae321f3ad23435f5ea619` |
| `jepa_horizon/1` | `readouts/jepa_horizon_seed1_ep020.npz` | `3af48701101d516564033ca14c572ba3000dc1b7e8719c977e01dba55bdb606b` | `fa3fc130f3421895130ea2174a88f698fb7db1692f8de6d76368153d2a3a096a` |
| `jepa_horizon/2` | `readouts/jepa_horizon_seed2_ep020.npz` | `929083d4289d729f9bfdae7aae27fac3f81429cdb046e52d873bc3b0ef3387c9` | `c4a5d3c011ff79b1cc887f0f7045f55aba043ecacd0b054280a470507f0b9ecc` |
| `jepa_masked/0` | `readouts/jepa_masked_seed0_ep020.npz` | `41113da279d5887a7784a25dd0393065027598f19e44707032dd9f7f2b83db73` | `4a20451e3ab47a3ba6dab14cbbaaa076027fb075a384f7ab749259492e57d189` |
| `jepa_masked/1` | `readouts/jepa_masked_seed1_ep020.npz` | `c9ea840417162c346f40840746401ba3125756dc8004552ac8c01b98ff01ff97` | `75a74630448b0244e99df781e4c4db1140e79b9bc3d8dd9ccf54d88365b9289f` |
| `jepa_masked/2` | `readouts/jepa_masked_seed2_ep020.npz` | `fc2717ef4828dd855f9eccfefce5f66843320f043c2c5c1bf237f352b42b4e98` | `67ee7db0ee92b4402225e3d1161fe7c71d55c545cdbc72c1a5b115baddecb94d` |
| `supervised/0` | `readouts/supervised_seed0_ep020.npz` | `94b884ef8126ffd0f9fb0e9556142788b370873c2edb07bd9b17c294a4ecf7f0` | `92657fe4c1b6c1ee1e3e0b0f3b31d1ca92980cb1c63cbe08c7c010c9cfd468db` |
| `supervised/1` | `readouts/supervised_seed1_ep020.npz` | `c6f176b42c2892775b06e5713a504ebc2817c9505441ba7f661e50e37c2633c0` | `1ec5fa7df081e6c4c392a19c13a21bd24895507d13998efb68680aaa8424f8ff` |
| `supervised/2` | `readouts/supervised_seed2_ep020.npz` | `e934a4d0c41dcc632122df37bab8799b4abc000e6c7fb002dc5b2242e770a18a` | `29ff331f3f531cdbba9c14a223b577014f03a97efe42fada9eec03737b68d7ed` |

Tutti i checkpoint sono epoch 20. I seed canonici sono esattamente
`{0, 1, 2}` per ciascuno dei tre branch.

## 2. Readout canonici

L'encoder produce una griglia `(B, K=20, S=4, d_model=128)`.

- `last_concat512`:
  `grid[:, -1, :, :].reshape(B, S * d_model)`;
- `meanK_concatS`, chiamato `tmean_concat512` nei dump:
  `grid.mean(dim=1).reshape(B, S * d_model)`.

Entrambi hanno dimensione 512. Non sono learned pooling e la loro definizione
non deve essere modificata.

Ogni dump contiene quattro matrici `float32`:

- train: `(100000, 512)` per entrambi i readout;
- val: `(50000, 512)` per entrambi i readout.

## 3. Split, stock-day e identità di riga

Schema split: `thesis.stage1.split`, versione 2.  
Algoritmo: `grouped_split_by_stock_day.v1`.  
`split_seed=0`, `subsample_seed=0`, `val_frac=0.1`, `K=20`,
`max_horizon=20`, `vol_clip=5.0`.

Hash/fingerprint:

- fingerprint split:
  `aba0105a4e1049e24a96aa751671a4618da48543dd5fd011f73fbbda0e21649c`;
- train endpoint:
  `f82da32386f8bb7e5d0e77159b974bc59ab0a5e42ffd146f1ba381c252269e3d`;
- val endpoint:
  `f150978c0de04d3a7bcad8883f363d212f3cfeaec91640ce5b86e9d2f9f97983`.

Le righe nei dump sono identificate oggi da:

`(stock_id, day_id, raw_endpoint_index)`.

L'NPZ non contiene un timestamp. `raw_endpoint_index` è strettamente ordinato
nel file e individua l'endpoint; il timestamp richiesto dal nuovo protocollo
può però essere ricostruito esattamente dai CSV verificati applicando gli
stessi filtri del builder. I context window validi verificano che tutto
`[t-K+1, t+max_horizon]` resti nello stesso `(stock_id, day_id)`.

Lo split v2 contiene:

| stock_id | giorni train osservati nel campione | giorni val | righe train campionate | righe val campionate |
|---:|---:|---:|---:|---:|
| 0 | 226 | 18 | 14312 | 5332 |
| 1 | 218 | 26 | 13614 | 7419 |
| 2 | 211 | 26 | 15172 | 8538 |
| 3 | 215 | 29 | 13131 | 7698 |
| 4 | 219 | 23 | 13943 | 6225 |
| 5 | 223 | 19 | 15928 | 6121 |
| 6 | 215 | 29 | 13900 | 8667 |

Totale: 1.527 stock-day train osservati, 170 val, zero overlap. Un unico
stock-day valido di stock 5 non compare nei due campioni finiti. I simboli non
sono serializzati nel bundle, ma sono recuperabili senza ambiguità dalla
mappatura verificata nella sezione 0.

### 3.1 Split preregistrato corretto

La ricostruzione integrale contiene 7.323.510 endpoint validi in 1.698
stock-day. L'algoritmo storico con `split_seed=0` assegna 1.528 stock-day al
train e 170 all'held-out. Il train storico è conservato integralmente.

Per ogni titolo i giorni held-out sono ordinati per `trading_date`; la prima
metà (`floor(n/2)`) costituisce la nuova validation e la seconda metà il test.
In presenza di un numero dispari, il giorno aggiuntivo è assegnato al test.

| stock_id | held-out storico | validation | test |
|---:|---:|---:|---:|
| 0 | 18 | 9 | 9 |
| 1 | 26 | 13 | 13 |
| 2 | 26 | 13 | 13 |
| 3 | 29 | 14 | 15 |
| 4 | 23 | 11 | 12 |
| 5 | 19 | 9 | 10 |
| 6 | 29 | 14 | 15 |
| **totale** | **170** | **83** | **87** |

Conteggi finali:

| split | stock-day | endpoint |
|---|---:|---:|
| train | 1.528 | 6.596.688 |
| validation | 83 | 352.931 |
| test | 87 | 373.891 |

Fingerprint del protocollo:
`0c15740e494b9121b308c687d09d76af279a399f99896a45f23778a3b7e18648`.

Validation e test derivano quindi dal precedente held-out set, già usato nelle
analisi esplorative storiche. Il nuovo test è esplicitamente escluso dalla
selezione di alpha, whitening-k e di qualsiasi altro iperparametro di
Experiment 01.

## 4. Target e ridondanze

`targets_shared.npz` contiene 22 target raw `float32`:

- 20 direzionali: cinque famiglie
  (`d_spread_z`, `d_microprice_rel`, `d_best_bid_rel`,
  `d_best_ask_rel`, `d_top_imbalance`) agli orizzonti `{1,5,10,20}`;
- due volatilità: `realized_vol@{5,20}`.

Per ogni orizzonte vale la ridondanza esatta indotta da
`mid=(best_bid+best_ask)/2`: `d_spread_z`, `d_best_bid_rel` e
`d_best_ask_rel` descrivono un'unica quantità di rango uno. L'aggregato
canonico conserva `d_spread_z` e rimuove le otto copie
`d_best_bid_rel@*`, `d_best_ask_rel@*`, lasciando 12 target direzionali
indipendenti. Tutti i 20 devono comunque restare nelle tabelle per-target.

`targets_heldout.npz` contiene 8 target imbalance, 8 depth e il timing
`time_to_next_mid_move`. Il timing usa esattamente
`log1p(observed_or_capped)` su tutte le righe, cap 600; non è un estimand
survival.

## 5. Metrica, aggregazione e reader lineare canonico

Il `R²` per target è:

`1 - sum((y - y_hat)^2) / max(sum((y - mean(y_eval))^2), 1e-12)`.

Il baseline usa quindi la media dello split di valutazione. L'aggregato
assoluto canonico è:

1. media dei `R²` per target indipendenti dentro ciascun encoder seed;
2. media e deviazione fra i tre encoder seed.

Il reader canonico:

- centra `X` sulla media train;
- centra `Y` sulla media train e reintroduce tale media come intercetta;
- non standardizza le coordinate;
- usa `numpy.linalg.lstsq(..., rcond=None)` / soluzione min-norm;
- PCA train-only, centrata e non standardizzata.

Utility esistenti:

- ridge e statistiche sufficienti: `ladder_accessibility.py`,
  `consolidation_geometry.py`;
- PCA, ladder, basi Haar e sottospazi casuali:
  `consolidation_geometry.py`;
- nessuna implementazione canonica di progressive top-k whitening;
- ridge storico usa lambda assoluta, non la nuova griglia alpha
  trace-normalized.

Missing values: i target/features canonici sono attesi finiti; il codice
storico non imputa. Il nuovo protocollo deve rifiutare NaN/Inf e target
costanti devono avere uno stato esplicito, non NaN silenziosi.

## 6. Ambiente

Ambiente che ha prodotto i dump:

- Python 3.12.3;
- NumPy 2.2.4;
- PyTorch 2.9.1+rocm6.3;
- ROCm 6.3.42134-a9a80e791;
- AMD Radeon RX 7800 XT.

Ambiente corrente `../rocm_env`:

- Python 3.12.3;
- NumPy 2.2.4;
- pandas 2.2.3;
- pyarrow 24.0.0;
- SciPy 1.15.2;
- scikit-learn 1.6.1;
- pytest 9.1.1;
- matplotlib 3.10.1;
- PyTorch 2.9.1+rocm6.3;
- psutil 7.0.0.

La Phase I lineare è CPU/BLAS; l'eventuale MLP Phase III usa ROCm/CUDA.

## 7. Gate storico disponibile e gate v2 bloccato

Il consolidamento storico sui dump verificati dà, per
`last_concat512/full-rank/min-norm OLS/direzionale`:

- `jepa_horizon`: 0.2111;
- `supervised`: 0.3756.

Questo gate può e deve essere riprodotto gratuitamente sui dump post-P0.
Non autorizza però l'uso della vecchia `val` contemporaneamente per tuning e
test. Il grid Phase I v2 rimane bloccato finché non viene prodotto e
preregistrato un bundle conforme a tre split.
