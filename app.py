import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
import streamlit as st

import utils.config
import core.reference_analyzer
import core.footage_indexer
import core.video_editor

# Garante que atualizações nos módulos sejam recarregadas pelo servidor Streamlit sem precisar reiniciar o processo
importlib.reload(utils.config)
importlib.reload(core.reference_analyzer)
importlib.reload(core.footage_indexer)
importlib.reload(core.video_editor)

from utils.config import (
    ASPECT_RATIOS,
    EXPORT_QUALITIES,
    TempWorkspace,
    check_ffmpeg_installed,
    ensure_output_folder,
    get_gemini_api_key,
    get_gemini_model,
    validate_input_folder,
)
from core.reference_analyzer import ReferenceAnalyzer
from core.footage_indexer import FootageIndexer
from core.video_editor import VideoEditor

st.set_page_config(
    page_title="AI Video Style Transfer & Editor",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Automação de Edição de Vídeo Orientada por Referência Visual e Rítmica")
st.markdown(
    "Analise o ritmo, cortes, BPM, enquadramento e color grading de qualquer vídeo de referência "
    "(YouTube, Reels, TikTok ou arquivo local) e replique automaticamente nos seus vídeos brutos com **Gemini Vision + PySceneDetect + Librosa + MoviePy**."
)

# Inicialização de estados de sessão
if "logs" not in st.session_state:
    st.session_state.logs = []
if "blueprint" not in st.session_state:
    st.session_state.blueprint = None
if "indexed_segments" not in st.session_state:
    st.session_state.indexed_segments = None
if "output_video_path" not in st.session_state:
    st.session_state.output_video_path = None

# Barra lateral para Configurações de API e Diagnóstico do Sistema
with st.sidebar:
    st.header("⚙️ Configurações & Sistema")

    ffmpeg_ok, ffmpeg_info = check_ffmpeg_installed()
    if ffmpeg_ok:
        st.success(f"FFmpeg detectado: `{ffmpeg_info}`")
    else:
        st.error(ffmpeg_info)

    api_key_input = st.text_input(
        "🔑 GEMINI_API_KEY",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Insira sua chave do Google AI Studio ou configure no arquivo .env",
    )

    model_choice = st.selectbox(
        "🤖 Modelo Gemini",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="gemini-2.5-flash é mais rápido; gemini-2.5-pro oferece raciocínio multimodal mais profundo.",
    )

    use_ref_audio = st.checkbox(
        "🎵 Usar áudio da referência se não houver música na pasta de entrada",
        value=True,
        help="Caso a pasta de entrada não tenha arquivos .mp3/.wav separados, usa a trilha rítmica da referência.",
    )

    reuse_footage_cache = st.checkbox(
        "⚡ Reaproveitar leitura dos vídeos brutos (Cache)",
        value=True,
        help="Se os vídeos da pasta de entrada já foram lidos antes (ex: você editou um Short e agora quer gerar um vídeo longo), reaproveita a indexação instantaneamente.",
    )

    st.divider()
    st.caption("Pipeline Modular:")
    st.markdown(
        "1. **Downloader (`yt-dlp`)**\n"
        "2. **Estilo & Ritmo (`scenedetect` + `librosa` + `Gemini`)**\n"
        "3. **Indexador (`OpenCV` + `Gemini Vision`)**\n"
        "4. **Renderizador (`MoviePy` + `FFmpeg`)**"
    )

# Formulário Principal de Entrada
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📂 Fontes e Destino")
    input_folder = st.text_input(
        "Pasta de Entrada (Vídeos/Áudios Brutos)",
        placeholder="/caminho/para/meus_videos_brutos",
        help="Pasta contendo arquivos .mp4, .mov, .mkv, .avi e opcionalmente uma trilha .mp3/.wav.",
    )

    reference_url = st.text_input(
        "URL do Vídeo de Referência (YouTube / Reels / TikTok ou caminho local)",
        placeholder="https://www.youtube.com/shorts/...",
        help="Link público do vídeo cujo ritmo de cortes e estética visual serão copiados.",
    )

    output_folder = st.text_input(
        "Pasta de Saída (Destino do Vídeo Editado)",
        placeholder="/caminho/para/videos_editados",
        help="Pasta onde o arquivo MP4 renderizado e o JSON do Blueprint serão salvos.",
    )

with col_right:
    st.subheader("🎞️ Parâmetros de Exportação")
    aspect_ratio_label = st.selectbox(
        "Proporção de Tela (Aspect Ratio)",
        options=list(ASPECT_RATIOS.keys()),
        index=0,
    )

    quality_label = st.selectbox(
        "Qualidade de Exportação",
        options=list(EXPORT_QUALITIES.keys()),
        index=1,
    )

    st.info(
        f"**Resolução Alvo:** `{ASPECT_RATIOS[aspect_ratio_label][0]}x{ASPECT_RATIOS[aspect_ratio_label][1]}`\n\n"
        f"**Preset FFmpeg:** `{EXPORT_QUALITIES[quality_label]['preset']}` (CRF `{EXPORT_QUALITIES[quality_label]['crf']}`)"
    )

start_btn = st.button("🚀 Iniciar Análise e Edição Automatizada", type="primary", use_container_width=True)

# Containers para Barra de Progresso e Logs em Tempo Real
progress_bar = st.progress(0, text="Aguardando início do processamento...")
log_expander = st.expander("📋 Logs Detalhados das Etapas em Tempo Real", expanded=True)
log_placeholder = log_expander.empty()


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    st.session_state.logs.append(entry)
    log_placeholder.code("\n".join(st.session_state.logs), language="text")


if start_btn:
    st.session_state.logs = []
    st.session_state.blueprint = None
    st.session_state.indexed_segments = None
    st.session_state.output_video_path = None

    try:
        # Passo 0: Validações de Ambiente e Pastas
        progress_bar.progress(5, text="Validando dependências, chave de API e diretórios...")
        append_log("Verificando instalação do FFmpeg e configurações...")

        ffmpeg_ok, ffmpeg_msg = check_ffmpeg_installed()
        if not ffmpeg_ok:
            raise RuntimeError(ffmpeg_msg)

        api_key = get_gemini_api_key(api_key_input)
        model_name = get_gemini_model(model_choice)

        video_files, audio_files = validate_input_folder(input_folder)
        out_dir = ensure_output_folder(output_folder)
        append_log(f"Pasta de entrada validada: {len(video_files)} vídeo(s) e {len(audio_files)} áudio(s) encontrados.")

        if not reference_url or not reference_url.strip():
            raise ValueError("Informe a URL (ou caminho local) do vídeo de referência.")

        with TempWorkspace() as temp_dir:
            append_log(f"Workspace temporário criado em: {temp_dir}")

            # Módulo 1: Download do Vídeo de Referência
            progress_bar.progress(18, text="Módulo 1/4: Baixando vídeo de referência via yt-dlp...")
            analyzer = ReferenceAnalyzer(api_key=api_key, model_name=model_name, log_fn=append_log)
            ref_video_path = analyzer.download_reference_video(reference_url.strip(), temp_dir)

            # Módulo 2: Análise de Estilo da Referência (PySceneDetect + Librosa + Gemini)
            progress_bar.progress(40, text="Módulo 2/4: Extraindo cortes, BPM e gerando Blueprint de Edição com Gemini...")
            blueprint = analyzer.generate_editing_blueprint(ref_video_path, temp_dir)
            st.session_state.blueprint = blueprint

            # Salva uma cópia do Blueprint na pasta de saída para auditoria/reuso
            blueprint_path = out_dir / "editing_blueprint.json"
            blueprint_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
            append_log(f"Blueprint de Edição salvo em: {blueprint_path}")

            # Módulo 3: Indexação dos Arquivos Brutos (OpenCV + Gemini Vision)
            progress_bar.progress(65, text="Módulo 3/4: Carregando/analisando vídeos brutos (OpenCV + Gemini Vision)...")
            indexer = FootageIndexer(api_key=api_key, model_name=model_name, log_fn=append_log)
            avg_dur = float(blueprint.get("avg_shot_duration", 2.2))
            indexed_segments = indexer.index_footage_folder(
                video_files,
                target_avg_duration=avg_dur,
                temp_dir=temp_dir,
                use_cache=reuse_footage_cache,
            )
            st.session_state.indexed_segments = indexed_segments

            # Módulo 4: Montagem e Renderização (MoviePy + FFmpeg)
            progress_bar.progress(85, text="Módulo 4/4: Montando timeline, aplicando cor/cortes e renderizando MP4 final...")
            editor = VideoEditor(log_fn=append_log)
            final_video_path = editor.render_edited_video(
                blueprint=blueprint,
                indexed_segments=indexed_segments,
                audio_files=audio_files,
                reference_video_path=ref_video_path,
                output_dir=out_dir,
                target_resolution=ASPECT_RATIOS[aspect_ratio_label],
                quality_cfg=EXPORT_QUALITIES[quality_label],
                temp_dir=temp_dir,
                use_reference_audio_if_no_music=use_ref_audio,
            )

            st.session_state.output_video_path = str(final_video_path)
            append_log("Limpando arquivos temporários...")

        progress_bar.progress(100, text="✅ Edição concluída com sucesso!")
        st.success(f"🎉 Vídeo editado e exportado com sucesso em: `{st.session_state.output_video_path}`")

    except Exception as exc:
        append_log(f"❌ ERRO: {exc}")
        progress_bar.progress(0, text="Interrompido devido a erro.")
        st.error(f"Erro durante o processamento: {exc}")

# Exibição de Resultados (Vídeo Renderizado, Blueprint e Catálogo de Takes)
if st.session_state.output_video_path or st.session_state.blueprint:
    st.divider()
    tab_video, tab_blueprint, tab_footage = st.tabs(
        ["🎬 Vídeo Final Renderizado", "📐 Blueprint de Edição (JSON)", "🗂️ Catálogo de Takes Brutos"]
    )

    with tab_video:
        if st.session_state.output_video_path and Path(st.session_state.output_video_path).exists():
            vid_col, info_col = st.columns([2, 1])
            with vid_col:
                st.video(st.session_state.output_video_path)
            with info_col:
                st.markdown("### Detalhes do Arquivo")
                out_p = Path(st.session_state.output_video_path)
                st.write(f"**Arquivo:** `{out_p.name}`")
                st.write(f"**Caminho completo:** `{out_p}`")
                st.write(f"**Tamanho:** `{out_p.stat().st_size / (1024 * 1024):.2f} MB`")
                with open(out_p, "rb") as f:
                    st.download_button(
                        label="⬇️ Baixar Vídeo MP4",
                        data=f,
                        file_name=out_p.name,
                        mime="video/mp4",
                        use_container_width=True,
                    )

    with tab_blueprint:
        if st.session_state.blueprint:
            st.markdown(f"**Resumo do Estilo:** {st.session_state.blueprint.get('style_summary', '')}")
            st.json(st.session_state.blueprint)

    with tab_footage:
        if st.session_state.indexed_segments:
            st.dataframe(st.session_state.indexed_segments, use_container_width=True)
