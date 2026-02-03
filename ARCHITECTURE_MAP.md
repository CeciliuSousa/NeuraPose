# Mapa da Arquitetura NeuraPose

## 1. Fluxo de Dados (Pipeline Completo)

O sistema opera sob uma premissa de **Amostragem Temporal (10 FPS)** encapsulada em um fluxo de vídeo de alta taxa (30 FPS) para fluidez visual.

```mermaid
graph TD
    A[Vídeo Bruto (30fps)] -->|Logical Skip 30->10fps| B(processador.py)
    B -->|Gravação| C[Vídeo Preview (30fps Interpolado)]
    B -->|Extração| D[JSONs de Pose (10fps Sparse)]
    D -->|converte_pt.py| E[data.pt (Tensores PyTorch)]
    E -->|treinador.py| F[Modelo Temporal (TFT/LSTM)]
    F -->|Salva| G[model_best.pt]
    G -->|Carrega| H(app/testar_modelo.py)
    A -->|Inferência 30->10fps| H
    H -->|Classificação| I[Relatório Final (JSON/PDF)]
```

### Detalhe do "Logical Skip"
* **Entrada**: Frames 0, 1, 2, 3, 4, 5... (30fps)
* **Processamento**: IA roda apenas em 0, 3, 6... (10fps efetivos)
* **JSON**: Contém apenas frames chaves [0, 3, 6...]
* **Conversão**: `converte_pt.py` empilha esses frames sequencialmente. O modelo "vê" uma sequência contínua t=0, t=1... que corresponde temporalmente a 0s, 0.1s...

## 2. Módulos Identificados

| Estágio | Arquivo Responsável | Descrição Técnica |
| :--- | :--- | :--- |
| **Extração** | `neurapose_backend/pre_processamento/pipeline/processador.py` | **Refatorado**: Lê vídeo a 30fps, aplica detecção/pose a cada 3 frames, grava vídeo fluido repetindo último frame processado, e salva JSON esparso. Usa `Sanitizer` anti-teleporte. |
| **Preparação** | `neurapose_backend/pre_processamento/converte_pt.py` | Lê JSONs esparsos + `labels.json`. Agrupa keypoints por ID. Divide em janelas de tempo (`TIME_STEPS=30` frames = 3 segundos reais). Normaliza (Z-Score) e salva em `data.pt`. |
| **Treino** | `neurapose_backend/LSTM/pipeline/treinador.py` | Carrega `data.pt`. Treina modelo (TFT/LSTM) usando `FocalLoss` e `AugmentedDataset`. Gera checkpoints em `modelos-temporais`. |
| **Teste** | `neurapose_backend/app/testar_modelo.py` | **Refatorado**: Script de validação que replica o pipeline de Extração + Inferência do Modelo Temporal. Gera métricas (Acurácia/F1) e tabelas de acerto por vídeo. |
| **Inference App** | `neurapose_backend/app/pipeline/processador.py` | Modulo espelho do processador central, adaptado para rodar a inferência LSTM logo após a extração RTMPose. |

## 3. Detalhes do Dataset

### Estrutura Intermediária (JSON)
Lista de objetos para cada detecção (apenas frames processados):
```json
[
  { "frame": 0, "id_persistente": 1, "keypoints": [[x,y,c]...], "bbox": [...] },
  { "frame": 3, "id_persistente": 1, "keypoints": [[x,y,c]...], "bbox": [...] }
]
```

### Formato Final (Tensors - `data.pt`)
Arquivo binário PyTorch contendo dicionário:
* **`data`**: Tensor Float32 de shape `(N_Amostras, 30, 34)`
    * `N_Amostras`: Quantidade total de clipes extraídos.
    * `30`: `TIME_STEPS` (Janela temporal de 3 segundos a 10fps).
    * `34`: `NUM_JOINTS (17) * CHANNELS (2)`. (Score é descartado na conversão).
* **`labels`**: Tensor Long de shape `(N_Amostras)` (0 ou 1).
* **`metadata`**: Tensor de indexação `(scene, clip, pid, sample_idx)`.

## 4. Oportunidades e Riscos

### ✅ Otimizações Já Implementadas
1.  **Logical Skip System-Wide**: Implementado no `nucleo` e no `app`. Garante que não processamos 2/3 dos frames inutilmente.
2.  **Visualização Fluida**: O vídeo de saída (preview) mantém 30fps para agradar o usuário, mesmo com IA a 10fps.
3.  **Sanitização Centralizada**: O módulo `sanatizer.py` é chamado em ambos os pipelines de extração.

### ⚠️ Pontos de Atenção (Riscos)
1.  **Sincronia JSON -> Modelo**: Confirmar se o `converte_pt.py` apenas empilha a lista. Se o vídeo tiver gaps grandes (ex: oclusão de 2 segundos), o `converte_pt.py` vai colar o frame t=0 com t=60 como se fossem vizinhos?
    *   *Análise*: `converte_pt.py` usa `extract_sequence` que apenas dá append: `frames.append(coords)`.
    *   *Risco*: Sim, o modelo não saberá que houve um salto temporal se o ID sumir e voltar.
    *   *Mitigação*: O Tracker (DeepOCSORT) tenta manter o ID vivo (predict) durante oclusões curtas. Oclusões longas quebram o ID, gerando novos `id_persistente`, o que é o comportamento correto (nova sequência).

2.  **Inferência em Tempo Real**: Se o usuário quiser "Live Streaming" (webcam), o `processador.py` (que é baseado em arquivo `cv2.VideoCapture`) precisará de adaptação para *bufferizar* 30 frames (3 segundos) antes de enviar para o LSTM.
