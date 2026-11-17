# 🎬 AI Video Editor — Edição de Vídeo Orientada por Referência Visual e Rítmica

Sistema completo em Python que recebe uma **pasta de vídeos/áudios brutos**, um **link de vídeo de referência** (YouTube, Reels, TikTok, Shorts ou arquivo local) e uma **pasta de saída**, extraindo automaticamente o **"DNA de Edição"** da referência e aplicando-o aos seus vídeos brutos.

---

## 🏗️ Arquitetura e Pipeline Modular

```text
ai-video-editor/
├── app.py                      # Interface Web (Streamlit) com progresso e logs em tempo real
├── core/
│   ├── reference_analyzer.py   # Módulos 1 e 2: yt-dlp + PySceneDetect + Librosa + Gemini Blueprint
│   ├── footage_indexer.py      # Módulo 3: OpenCV (nitidez, luz, movimento, foco) + Gemini Vision
│   └── video_editor.py         # Módulo 4: MoviePy + FFmpeg (Smart Crop, Color Grading, Beat Sync, Loudnorm)
├── utils/
│   └── config.py               # Validações de ambiente (FFmpeg, pastas), presets e limpeza temporária
├── requirements.txt            # Dependências do projeto
├── .env.example                # Modelo de configuração da GEMINI_API_KEY
└── README.md                   # Documentação passo a passo
```

### Como cada módulo funciona:
1. **Módulo 1 — Downloader de Referência (`core/reference_analyzer.py`):**
   - Baixa o vídeo da URL informada usando `yt-dlp` em um diretório temporário auto-limpável (`TempWorkspace`).
2. **Módulo 2 — Análise de Estilo da Referência (`core/reference_analyzer.py`):**
   - **`PySceneDetect` (`ContentDetector`):** Detecta todos os cortes secos/transições e calcula a duração exata de cada take.
   - **`Librosa`:** Extrai a trilha de áudio da referência, calcula o **BPM** e mapeia os timestamps exatos das batidas musicais (`beat_track`).
   - **`Google GenAI` (`gemini-2.5-flash` ou `gemini-2.5-pro`):** Recebe os frames-chave + métricas objetivas e retorna um **Blueprint de Edição estruturado em JSON** (via `Pydantic` schema), contendo: ritmo, transições, intensidade de zoom dinâmico, parâmetros de LUT/Color Grading (saturação, contraste, brilho, temperatura), estilo de legendas e a lista sequencial de `cut_slots`.
3. **Módulo 3 — Indexação dos Arquivos Brutos (`core/footage_indexer.py`):**
   - **`OpenCV`:** Varre os vídeos da pasta de entrada medindo nitidez (variância do Laplaciano), iluminação, estabilidade/movimento e o centro horizontal de ação (`focus_x`) para recorte inteligente.
   - **`Gemini Vision`:** Avalia os melhores candidatos atribuindo nota de interesse visual (`0-10`), nível de energia (`low`, `medium`, `high`), papel narrativo (`hook`, `buildup`, `climax`, `b-roll`, `outro`) e sugestões de legendas dinâmicas.
4. **Módulo 4 — Montagem e Renderização (`core/video_editor.py`):**
   - Pareia cada `cut_slot` do Blueprint com o melhor segmento bruto catalogado, quantizando a duração dos cortes nas batidas (BPM) da música.
   - Aplica **Smart Crop** para a proporção escolhida (`9:16`, `16:9` ou `1:1`), **Zoom Dinâmico**, **Color Grading**, **Legendas estilizadas via Pillow** (sem exigir instalação do ImageMagick) e normalização de áudio **EBU R128 (`loudnorm`)** via `FFmpeg`.

---

## 🛠️ Pré-requisitos do Sistema

1. **Python 3.10+**
2. **FFmpeg instalado no sistema operacional:**
   - **Ubuntu / Debian:**
     ```bash
     sudo apt update && sudo apt install -y ffmpeg
     ```
   - **macOS (Homebrew):**
     ```bash
     brew install ffmpeg
     ```
   - **Windows (Winget):**
     ```powershell
     winget install Gyan.FFmpeg
     ```
3. **Chave de API do Google Gemini (`GEMINI_API_KEY`):**
   - Crie gratuitamente no [Google AI Studio](https://aistudio.google.com/apikey).

---

## 🚀 Instalação Passo a Passo

1. **Acesse a pasta do projeto:**
   ```bash
   cd /home/ramon/.gemini/antigravity/scratch/ai-video-editor
   ```

2. **Crie e ative um ambiente virtual Python (`venv`):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -r requirements.txt
   ```

4. **Configure sua `GEMINI_API_KEY`:**
   ```bash
   cp .env.example .env
   ```
   Edite o arquivo `.env` e insira sua chave:
   ```env
   GEMINI_API_KEY="AIzaSySuaChaveAqui..."
   GEMINI_MODEL="gemini-2.5-flash"
   ```
   *(Você também pode colar a chave diretamente na barra lateral da interface Streamlit).*

---

## ▶️ Como Executar

Com o ambiente virtual ativado, inicie a interface web do Streamlit:

```bash
.venv/bin/streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

### Passo a passo na Interface:
1. **Pasta de Entrada:** Informe o caminho absoluto da pasta com seus vídeos brutos (`.mp4`, `.mov`, `.mkv`, `.avi`) e, opcionalmente, uma música de fundo (`.mp3`, `.wav`).
2. **URL do Vídeo de Referência:** Cole o link do vídeo (YouTube Shorts, Instagram Reels, TikTok ou vídeo horizontal do YouTube) ou o caminho de um vídeo local.
3. **Pasta de Saída:** Informe a pasta onde o vídeo editado `.mp4` e o arquivo `editing_blueprint.json` serão salvos.
4. **Proporção e Qualidade:** Escolha entre `9:16 (Reels/Shorts)`, `16:9 (YouTube)` ou `1:1 (Quadrado)` e o perfil de qualidade.
5. Clique em **🚀 Iniciar Análise e Edição Automatizada** e acompanhe a barra de progresso e os logs em tempo real.
