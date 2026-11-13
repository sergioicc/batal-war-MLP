# Batalla Naval: Aprendizaje por Refuerzo de Agente Único

Especificación de arquitectura, reglas y funcionamiento para entrenar una única red neuronal (MLP) contra mapas aleatorios. Entorno de ejecución: **Google Colab** (PyTorch + NumPy).

---

## 1. Objetivo

Entrenar una **única red neuronal (Multilayer Perceptron)** que juegue Batalla Naval en solitario (solo tablero atacante) frente a configuraciones aleatorias de flotas, minimizando la cantidad de disparos (turnos) necesarios para hundir todos los barcos enemigos.

El entrenamiento se estructura en un **esquema REINFORCE de doble nivel**:

1. **Retroalimentación inmediata (por disparo):** en cada turno, tras ejecutar un ataque (`HIT` —incluyendo el golpe que hunde un barco, `SUNK`, sin bono adicional— o `MISS`), la red recibe una señal de refuerzo positiva o negativa muy pequeña.
2. **Retroalimentación por bloque (cada 20 rondas/partidas):** al completar un ciclo de **20 partidas** en paralelo, se evalúa el promedio de disparos necesarios para ganar ($\mu_{\text{actual}}$) y su **desviación estándar** ($\sigma_{\text{actual}}$), otorgando una retroalimentación macro respecto a una media móvil exponencial ($\mu_{\text{EMA}}$), penalizando la alta dispersión y ponderando el mérito relativo de cada ronda mediante re-evaluación segura de lotes (con *ratio clipping* estilo PPO).

---

## 2. Decisiones de diseño consolidadas

| Tema | Decisión |
|---|---|
| Agente | Una única red neuronal (**MLP** en PyTorch), sin población ni torneo |
| Modalidad de juego | **Tablero atacante únicamente** (modo solitario contra entorno) |
| Colocación de flota enemiga | **Aleatoria** en cada ronda/partida (barajando orden de barcos) |
| Entradas de la red (`(100, 4)`) | Matriz de 100 casillas × **4 neuronas de entrada** con pesos compartidos: contexto de `HIT` previos, contexto de `UNSHOT` disponibles, mapa heurístico inicial `[0, 1]` y barco vivo más pequeño |
| Esquema de entrenamiento | **Doble actualización REINFORCE**: refuerzo inmediato por disparo + refuerzo macro cada **20 rondas** con re-evaluación de buffer, $\mu_{\text{EMA}}$, penalización por $\sigma$ y bono de entropía |
| Valores de hiperparámetros | Centralizados en `battleship_rl/config.py` para calibración experimental |
| Referencias de comparación | **Bot Aleatorio** y **Bot Heurístico Sencillo (Hunt & Target: paridad de 2 + caza ortogonal)** sobre semillas fijas de evaluación |
| Cómputo | Google Colab (GPU/CPU), motor vectorizado en NumPy por lotes de 20 partidas |

---

## 3. Reglas del juego y métricas de partida

- **Tablero:** cuadrícula de $10 \times 10$ (100 casillas). Coordenadas `(r, c)`, con $0 \le r, c < 10$.
- **Flota enemiga** (17 casillas en total, IDs `0..4`):

| ID | Barco | Longitud |
|---|---|---|
| `0` | Portaaviones | 5 |
| `1` | Acorazado | 4 |
| `2` | Crucero | 3 |
| `3` | Submarino | 3 |
| `4` | Destructor | 2 |

- **Colocación válida:** cada barco en línea recta continua, horizontal o vertical; dentro del tablero; sin solapamiento (se permite adyacencia).
- **Dinámica de turnos:** no hay tablero defensor ni contraataque rival. El agente dispara una vez por turno hasta hundir las 17 casillas de la flota.
- **Métrica principal de desempeño:** conteo total de ataques (turnos) realizados para finalizar la partida (mínimo teórico: 17; máximo posible: 100).
- **Resultado de un disparo:**
  - `MISS`: agua.
  - `HIT`: impacto en un barco aún a flote.
  - `SUNK`: impacto que termina de hundir un barco (marca todas sus casillas como `SUNK` y actualiza el barco vivo más pequeño; sin bono adicional distinto de `HIT`).
- **Visibilidad:** el agente nunca ve la posición real de los barcos enemigos; solo percibe las observaciones derivadas de sus propios disparos.

---

## 4. Estructura del proyecto

```text
battleship_rl/
├── game/
│   ├── constants.py         # reglas oficiales, tamaños y códigos de estado
│   ├── engine_ref.py        # motor OOP, validación de flotas y matriz (B, 100, 4)
│   └── engine_vec.py        # motor vectorizado por lotes con masking asíncrono de `done`
├── agents/
│   ├── mlp_agent.py         # MLP único con 4 neuronas de entrada por casilla (4 -> ocultas -> 1)
│   └── baselines.py         # Bot Aleatorio y Bot Heurístico Sencillo (Hunt & Target)
├── learning/
│   └── trainer.py           # bucle REINFORCE de doble actualización (paso + bloque 20)
├── utils/
│   ├── heuristic_map.py     # mapa heurístico inicial estático normalizado [0, 1]
│   ├── checkpoint.py        # guardado/carga de pesos, optimizadores, mu_ema y CSVs
│   └── metrics.py           # cálculo de bloques, feedback EMA/sigma y evaluación reproducible
├── experiments/
│   └── run_first_bot_eval.py # experimento listo para entrenar y comparar primer bot
├── tests/                   # 7 pruebas de integridad, observaciones, baselines y gradientes
├── config.py                # hiperparámetros centralizados
└── main.ipynb               # notebook principal para Google Colab
```

---

## 5. Componentes y requisitos

### 5.1 Motor de entorno en modo atacante (`engine_vec.py`)

Gestiona un lote paralelo de `batch_size = 20` partidas en modo solitario.

- **Estado por partida:** `ship_id` (`-1` agua, `0..4` barco), `shots` (`UNSHOT`, `MISS`, `HIT`, `SUNK`), `ship_hits` (`5`), `alive_ships` (`5`), `n_shots` y `done`.
- **Manejo asíncrono de finalización:** como cada partida del lote termina en un turno distinto, las llamadas posteriores a `step()` congelan el tablero y el contador `n_shots` de las partidas con `done == True` (devolviendo `OUTCOME_NOOP`), y el entrenador filtra siempre las partidas activas (`~done`).

### 5.2 Entradas y arquitectura de la red neuronal (`mlp_agent.py`)

El entorno entrega una matriz de forma `(B, 100, 4)` y el MLP tiene literalmente **4 neuronas de entrada** y **1 neurona de salida**, evaluando las 100 casillas con **pesos compartidos**:

| Neurona de entrada (por casilla $i$) | Rango | Descripción |
|---|---|---|
| **1. Hacia dónde ya atacó (`HIT` local)** | `{0.0, 0.5, 1.0}` | `1.0` si extiende línea de 2+ `HIT` activos, `0.5` si tiene un `HIT` ortogonal adyacente, `0.0` en otro caso (y `0.0` si ya fue atacada). |
| **2. Dónde falta por atacar (`UNSHOT` local)** | `[0, 1]` | Proporción de vecinos ortogonales aún `UNSHOT` (`0.0` si la casilla ya fue atacada o está aislada). |
| **3. Mapa heurístico inicial** | `[0, 1]` | Valor estático normalizado del mapa heurístico en la casilla $i$. |
| **4. Barco vivo más pequeño** | `[0, 1]` | `min_alive_length / 5.0` (`0.0` al finalizar la partida). |
| **Total de neuronas de entrada** | **4** | Aplicado en paralelo sobre las 100 casillas → `(B, 100)` logits enmascarados. |

### 5.3 Esquema de doble entrenamiento REINFORCE secuencial (`trainer.py`)

El entrenador utiliza **una única tasa de aprendizaje (`learning_rate`)** y **rondas estrictamente secuenciales** (`ronda 1 → ronda 2 → ... → ronda 20`), diferenciando ambos niveles únicamente por el valor del refuerzo:

1. **Actualización inmediata (por cada disparo, en cada ronda secuencial):**
   - `HIT` (incluye `SUNK`, sin bono adicional) recibe `reward_hit` pequeño; `MISS` recibe `reward_miss` pequeño.
   - Pérdida directa: `perdida_disparo = -(recompensa * log_prob)` y actualización inmediata con `optimizer.step()`.
2. **Actualización macro por bloque (al finalizar las 20 rondas secuenciales):**
   - Calcula $\mu_{\text{actual}}$ y $\sigma_{\text{actual}}$ de los 20 conteos $\{T_1, \dots, T_{20}\}$.
   - Calcula la señal del bloque respecto a la media móvil histórica $\mu_{\text{EMA}}$ con atenuación multiplicativa por varianza ($\sigma_{\text{actual}}^2$):
     $$R_{\text{bloque}} = \frac{w_{\mu}(\mu_{\text{EMA}} - \mu_{\text{actual}})}{1 + w_{\text{var}} \cdot \sigma_{\text{actual}}^2}$$
     y actualiza $\mu_{\text{EMA}} \leftarrow (1 - \alpha_{\text{EMA}})\mu_{\text{EMA}} + \alpha_{\text{EMA}}\mu_{\text{actual}}$ (en el bloque 1 inicia $\mu_{\text{EMA}} = \mu_{\text{actual}}$). De este modo, si el promedio bajó, $R_{\text{bloque}}$ **siempre es positivo** (la varianza alta solo reduce su magnitud en valor absoluto sin volverlo negativo).
   - Re-evalúa las jugadas de las 20 rondas y actualiza con el mismo optimizador: `perdida_bloque = -(R_bloque * log_prob)`.

### 5.4 Bots de referencia (`baselines.py`)

1. **Bot Aleatorio:** elige uniformemente al azar entre casillas `UNSHOT`.
2. **Bot Heurístico Sencillo (Hunt & Target):**
   - **Hunt:** ataca casillas con paridad de tablero de ajedrez `(r + c) % 2 == 0` cada dos espacios (y completa `(r + c) % 2 == 1` si se agotan).
   - **Target:** al existir `HIT` activos, prioriza extender ejes alineados de `HIT` adyacentes y, en su defecto, ataca los vecinos ortogonales disponibles hasta hundir el barco.

### 5.5 Evaluación reproducible y persistencia (`metrics.py`, `checkpoint.py`)

- Todas las comparativas periódicas contra los bots de referencia usan un conjunto fijo y reproducible de mapas (`eval_seed_base`) para medir progreso real sin sesgo de muestreo.
- Guarda automáticamente pesos del MLP, ambos optimizadores, `mu_ema`, mejor promedio evaluado e historiales en CSV (`training_metrics.csv`, `eval_metrics.csv`) en Drive o localmente.

---

## 6. Pruebas obligatorias (`tests/`)

1. **Integridad del tablero:** 200 mapas aleatorios con 17 casillas exactas y 5 barcos reglamentarios sin solapamientos.
2. **Consistencia de las 501 entradas:** partición exacta de las 100 casillas sin intersección, mapa heurístico estático normalizado en `[0, 1]` y actualización progresiva de `min_alive_length / 5.0` al hundir barcos (distinguiendo los dos barcos de longitud 3).
3. **Sin disparos repetidos:** la máscara impide repetir casillas y toda partida termina en $\le 100$ disparos.
4. **Cálculo de bloque de 20 rondas:** verificación de media, desviación estándar y reducción estricta de la señal al aumentar $\sigma$.
5. **Bots de referencia:** verificación de no repetición en Bot Aleatorio y patrón de paridad 2 (Hunt) + vecinos ortogonales (Target) en Bot Heurístico Sencillo.
6. **Lotes vectorizados con `done` asíncrono:** las partidas que terminan antes congelan `n_shots` y `shots` y devuelven `OUTCOME_NOOP`.
7. **Smoke test de gradientes PyTorch:** ejecución completa de 2 bloques de 20 rondas validando actualización de pesos sin errores de grafo ni `NaN`/`Inf`.
