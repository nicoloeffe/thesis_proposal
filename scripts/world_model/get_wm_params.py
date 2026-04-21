import torch
import numpy as np
from pathlib import Path
import sys

# Aggiungi i path necessari per importare i modelli
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.world_model import LOBWorldModel, WorldModelConfig

def inspect_wm_volatility(ckpt_path="checkpoints/wm_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Caricamento del checkpoint
    if not Path(ckpt_path).exists():
        print(f"Errore: Checkpoint non trovato in {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 2. Ripristino configurazione (d_latent=16 per il modello finale)
    cfg = WorldModelConfig()
    for k, v in ckpt["cfg"].items():
        setattr(cfg, k, v)
    
    print(f"Modello caricato: {ckpt_path}")
    print(f"Configurazione: K={cfg.n_gmm}, d_latent={cfg.d_latent}, d_model={cfg.d_model}\n")

    # 3. Istanza del modello e caricamento pesi
    model = LOBWorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 4. Estrazione parametri della testa GMM (sig_head)
    # log_sig è l'output di un layer lineare. Analizziamo il bias per avere 
    # un'idea della volatilità "base" appresa dal modello.
    with torch.no_grad():
        log_sig_bias = model.gmm_head.sig_head.bias.reshape(cfg.n_gmm, cfg.d_latent)
        sig_values = torch.exp(log_sig_bias) # Convertiamo da log-std a std

    print("--- Analisi Volatilità GMM (Standard Deviations) ---")
    names = ["low_vol", "mid_vol", "high_vol"] # Basato sui regimi del simulatore
    
    for k in range(cfg.n_gmm):
        avg_sig = sig_values[k].mean().item()
        min_sig = sig_values[k].min().item()
        max_sig = sig_values[k].max().item()
        
        print(f"Componente {k}:")
        print(f"  σ Media: {avg_sig:.6f}")
        print(f"  Range σ: [{min_sig:.6f}, {max_sig:.6f}]")
    
    print("\nNota: I valori rappresentano la scala della volatilità latente.")
    print(f"Il valore medio globale nel tuo test era ≈ 0.55.")

if __name__ == "__main__":
    inspect_wm_volatility()