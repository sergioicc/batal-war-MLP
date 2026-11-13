# Batalla Naval — Agente RL Único (`MLP` de 4 neuronas)

Proyecto de Aprendizaje por Refuerzo (**PyTorch + NumPy**) que entrena una única red neuronal Perceptrón Multicapa (**MLP**) para jugar Batalla Naval en modo atacante solitario ($10 \times 10$, flota reglamentaria de 5 barcos / 17 casillas), diseñado para ejecutarse localmente o en **Google Colab**.

---

## Arquitectura del agente y entrenamiento

* **Entrada matricial `(100, 4)` con MLP de 4 neuronas (`4 → 64 → 32 → 1`):**
  En lugar de aplanar el tablero en cientos de entradas independientes, cada casilla candidata se representa con 4 características y el MLP aplica los mismos pesos compartidos sobre las 100 casillas:
  1. **Contexto de ataques previos (`HIT` activos):** `1.0` si extiende línea de 2+ `HIT`, `0.5` si tiene un `HIT` ortogonal adyacente, `0.0` en otro caso.
  2. **Disponibilidad local (`UNSHOT`):** proporción de vecinos ortogonales aún sin atacar (`0.0` si ya fue atacada o está aislada).
  3. **Mapa heurístico inicial:** densidad de probabilidad estática sobre tablero vacío, normalizada en `[0, 1]`.
  4. **Barco vivo más pequeño:** longitud normalizada del menor barco aún a flote (`min_alive_len / 5.0`).
* **Doble escala REINFORCE secuencial (tasa única `learning_rate = 0.1`):**
  1. **Por disparo (rondas secuenciales `1..20`):** pérdida directa `-(recompensa * log_prob)` con refuerzos pequeños (`HIT/SUNK = +0.030`, `MISS = -0.006`).
  2. **Por bloque de 20 rondas:** compara el promedio ($\mu_{\text{actual}}$) contra la referencia histórica ($\mu_{\text{EMA}}$) con atenuación multiplicativa por varianza:
     $$R_{\text{bloque}} = \frac{w_{\mu}(\mu_{\text{EMA}} - \mu_{\text{actual}})}{1 + w_{\text{var}} \cdot \sigma_{\text{actual}}^2}$$
     De este modo, cuando el promedio mejora, la señal macro sigue siendo positiva y una varianza alta únicamente atenúa su magnitud en valor absoluto.
* **Bots de referencia (`baselines.py`):**
  * **Bot Aleatorio** (límite inferior).
  * **Bot Heurístico Sencillo (Hunt & Target):** barrido con paridad de tablero de ajedrez cada 2 espacios en *Hunt* y caza ortogonal priorizando ejes alineados en *Target*.

---

## Resultado inicial verificado (200 rondas secuenciales, `lr = 0.1`)

Evaluación sobre **100 mapas fijos compartidos** (`eval_seed_base = 10000`):

| Agente | Media de disparos | Desv. estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| MLP (sin entrenar) | 94.96 | 4.31 | 82 | 100 |
| Bot Aleatorio | 95.30 | 4.34 | 81 | 100 |
| Bot Heurístico (Hunt & Target) | 50.56 | 9.88 | 27 | 67 |
| **MLP entrenado (`lr = 0.1`, 200 rondas)** | **47.10** | **8.27** | **30** | **64** |

---

## Estructura del proyecto

```text
batal-war-MLP/
├── battleship_rl/
│   ├── config.py                # Hiperparámetros centralizados (lr = 0.1)
│   ├── main.ipynb               # Notebook para Google Colab
│   ├── game/
│   │   ├── constants.py         # Reglas oficiales 10x10 y flota de 17 casillas
│   │   ├── engine_ref.py        # Motor OOP, validación y features (B, 100, 4)
│   │   └── engine_vec.py        # Motor vectorizado para evaluación por lotes
│   ├── agents/
│   │   ├── mlp_agent.py         # MLPPolicyNet (4 -> 64 -> 32 -> 1) y MLPAgent
│   │   └── baselines.py         # Bot Aleatorio y Bot Heurístico (Hunt & Target)
│   ├── learning/
│   │   └── trainer.py           # Entrenador REINFORCE secuencial de doble escala
│   ├── utils/
│   │   ├── heuristic_map.py     # Mapa heurístico estático normalizado [0, 1]
│   │   ├── metrics.py           # Estadísticas, atenuación por varianza y evaluación
│   │   └── checkpoint.py        # Guardado/carga de pesos y CSVs (Colab / Drive)
│   ├── experiments/
│   │   └── run_first_bot_eval.py # Script reproducible de entrenamiento y comparativa
│   └── tests/                   # 7 pruebas de integridad, observaciones y gradientes
├── docs/
│   └── arquitectura.md          # Especificación técnica completa
├── requirements.txt
└── .gitignore
```

---

## Uso rápido

```bash
pip install -r requirements.txt

# 1. Verificar integridad del motor, entradas, baselines y gradientes
pytest battleship_rl/tests -v

# 2. Entrenar 200 rondas (10 bloques de 20) y comparar contra baselines
python -m battleship_rl.experiments.run_first_bot_eval --blocks 10 --eval-games 100 --eval-every 5 --lr 0.1
```

Para ejecutar en **Google Colab**, abre `battleship_rl/main.ipynb`. Para más detalles de diseño, consulta [docs/arquitectura.md](docs/arquitectura.md).
