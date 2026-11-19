#!/usr/bin/env bash
# ==============================================================================
# Script de Inicialização Automatizada - AI Video Editor
# Funciona no Linux e no macOS
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "🎬 Iniciando o AI Video Editor..."
echo "======================================================"

# 1. Verifica se o ambiente virtual existe, se não cria
if [ ! -d ".venv" ]; then
    echo "📦 Criando ambiente virtual Python (.venv)..."
    python3 -m venv .venv
fi

# 2. Ativa o ambiente virtual
source .venv/bin/activate

# 3. Garante que as dependências principais estão instaladas
echo "🔍 Verificando dependências essenciais..."
python -c "import streamlit, moviepy, faster_whisper, cv2" 2>/dev/null || {
    echo "⬇️ Instalando / atualizando dependências do requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

# 4. Inicia o Streamlit
echo "🚀 Abrindo interface web do AI Video Editor..."
echo "Acesse no seu navegador: http://localhost:8501"
echo "======================================================"
streamlit run app.py
