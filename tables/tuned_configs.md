# Configurações ajustadas (gerado por `scripts/18_tuning_table.py`)

> Ajuste: 8 configurações por método (a nº 0 é a padrão) × 2 sementes de ajuste (1000, 1001) × 200.000 passos; escolha pelo val médio dos 2 últimos pontos da curva. Maior (menos negativo) = melhor.

| escala | método | cfg escolhida | val ajuste (escolhida) | val ajuste (padrão) | ganho | configuração |
|---|---|---|---|---|---|---|
| cb4wd | dqn | 1 | -8.84 | -11.58 | +2.73 | lr=0.000443, target_sync=4000, train_every=2, batch=128, buffer=50000, eps_decay_frac=0.7 |
| cb4wd | linear | 1 | -29.54 | -37.37 | +7.83 | lr=0.000988, target_sync=4000, train_every=2, batch=32, buffer=50000, eps_decay_frac=0.5 |
| cb4wd | tab_station | 5 | -39.24 | -40.81 | +1.57 | alpha=0.5, eps_decay_frac=0.7, eps_end=0.05 |
| cb4wd | tab_cluster | 0 | -39.53 | -39.53 | +0.00 | alpha=0.1, eps_decay_frac=0.7, eps_end=0.05 |
| cb4wd | pg | 1 | -34.33 | -35.81 | +1.48 | lr_actor=0.000364, lr_critic=0.00146, entropy=0.1, episodes_per_update=4 |
| cb8wd | dqn | 1 | -42.11 | -60.73 | +18.62 | lr=0.000443, target_sync=4000, train_every=2, batch=128, buffer=50000, eps_decay_frac=0.7 |
| cb8wd | linear | 5 | -82.67 | -88.25 | +5.58 | lr=0.000222, target_sync=500, train_every=4, batch=128, buffer=10000, eps_decay_frac=0.3 |
| cb8wd | tab_station | 1 | -96.95 | -97.49 | +0.54 | alpha=0.2, eps_decay_frac=0.3, eps_end=0.1 |
| cb8wd | tab_cluster | 1 | -94.81 | -95.47 | +0.66 | alpha=0.2, eps_decay_frac=0.3, eps_end=0.1 |
| cb8wd | pg | 1 | -83.64 | -94.98 | +11.34 | lr_actor=0.000364, lr_critic=0.00146, entropy=0.1, episodes_per_update=4 |
| cb12wd | dqn | 1 | -84.94 | -106.91 | +21.97 | lr=0.000443, target_sync=4000, train_every=2, batch=128, buffer=50000, eps_decay_frac=0.7 |
| cb12wd | linear | 2 | -128.51 | -133.46 | +4.96 | lr=0.000557, target_sync=500, train_every=1, batch=32, buffer=10000, eps_decay_frac=0.7 |
| cb12wd | tab_station | 4 | -136.39 | -137.89 | +1.50 | alpha=0.05, eps_decay_frac=0.9, eps_end=0.05 |
| cb12wd | tab_cluster | 3 | -134.20 | -136.89 | +2.69 | alpha=0.2, eps_decay_frac=0.3, eps_end=0.01 |
| cb12wd | pg | 1 | -130.49 | -138.21 | +7.72 | lr_actor=0.000364, lr_critic=0.00146, entropy=0.1, episodes_per_update=4 |
