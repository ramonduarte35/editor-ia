import importlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List
import streamlit as st

import utils.config
import core.reference_analyzer
import core.footage_indexer
import core.video_editor
import core.speech_cleaner

# Garante que atualizações nos módulos sejam recarregadas pelo servidor Streamlit sem precisar reiniciar o processo
importlib.reload(utils.config)
importlib.reload(core.reference_analyzer)
importlib.reload(core.footage_indexer)
importlib.reload(core.video_editor)
importlib.reload(core.speech_cleaner)

from utils.config import (
    ASPECT_RATIOS,
    EXPORT_QUALITIES,
    TempWorkspace,
    check_ffmpeg_installed,
    ensure_output_folder,
    get_gemini_api_key,
    get_gemini_model,
    open_os_folder_dialog,
    validate_input_folder,
)
from core.reference_analyzer import ReferenceAnalyzer
from core.footage_indexer import FootageIndexer
from core.video_editor import VideoEditor
from core.speech_cleaner import AudioSpeechCleaner

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
if "input_folder_path" not in st.session_state:
    st.session_state.input_folder_path = ""
if "output_folder_path" not in st.session_state:
    st.session_state.output_folder_path = ""
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
        options=["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="gemini-3.8-flash é a versão mais moderna e rápida com suporte multimodal avançado.",
    )

    use_ref_audio = st.checkbox(
        "🎵 Usar áudio da referência se não houver música na pasta de entrada",
        value=True,
        help="Caso a pasta de entrada não tenha arquivos .mp3/.wav separados, usa a trilha rítmica da referência.",
    )

    st.subheader("🎙️ Limpeza Inteligente de Fala")
    remove_silence = st.checkbox(
        "✂️ Cortar silêncios e pausas longas (Jump Cut)",
        value=True,
        help="Detecta pausas na fala maiores que 0.5s e remove automaticamente.",
    )

    clean_speech_mistakes = st.checkbox(
        "🧠 Remover erros, repetições, gagueiras e espirros (Gemini)",
        value=True,
        help="O Gemini analisa a fala dos vídeos para identificar frases refeitas, travamentos ou ruídos e cortá-los fora.",
    )

    order_criteria = st.selectbox(
        "🗂️ Ordem dos Vídeos Brutos",
        options=["Ordem Alfabética (video_01, video_02...)", "Ordem Cronológica (Data de Gravação)"],
        index=0,
        help="Garante que a linha do tempo respeite o encadeamento correto da sua gravação.",
    )

    reuse_footage_cache = st.checkbox(
        "⚡ Reaproveitar leitura dos vídeos brutos (Cache)",
        value=True,
        help="Se os vídeos da pasta de entrada já foram lidos antes (ex: você editou um Short e agora quer gerar um vídeo longo), reaproveita a indexação instantaneamente.",
    )

    st.divider()
    st.caption("Pipeline Sequencial Executado:")
    st.markdown(
        "1. **Ordenar Brutos** (Alfabética ou Cronológica)\n"
        "2. **Corte de Silêncio** (FFmpeg silencedetect)\n"
        "3. **Filtro de Erros/Gagueira/Espirros** (Gemini Speech)\n"
        "4. **Curadoria das Melhores Cenas** (OpenCV + Gemini Vision)\n"
        "5. **Montagem da Timeline Sequencial**\n"
        "6. **Tratamento de Cor & Smart Crop**\n"
        "7. **Equalização & Loudnorm EBU R128 no Áudio Final**"
    )

# Formulário Principal de Entrada
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📂 Fontes e Destino")

    # Seleção da Pasta de Entrada com Explorador do SO
    st.markdown("**Pasta de Entrada (Vídeos/Áudios Brutos)**")
    inp_col1, inp_col2 = st.columns([4, 1])
    with inp_col1:
        input_folder = st.text_input(
            "Caminho da Pasta de Entrada",
            value=st.session_state.input_folder_path,
            placeholder="/caminho/para/meus_videos_brutos",
            help="Pasta contendo arquivos .mp4, .mov, .mkv, .avi e opcionalmente uma trilha .mp3/.wav.",
            label_visibility="collapsed",
        )
        st.session_state.input_folder_path = input_folder
    with inp_col2:
        if st.button("📁 Procurar...", key="btn_browse_input", use_container_width=True, help="Abre o explorador de arquivos do sistema operacional para selecionar a pasta"):
            chosen_dir = open_os_folder_dialog(title="Selecione a Pasta de Vídeos Brutos")
            if chosen_dir:
                st.session_state.input_folder_path = chosen_dir
                st.rerun()

    reference_url = st.text_input(
        "URL do Vídeo de Referência (Opcional - YouTube / Reels / TikTok ou arquivo)",
        placeholder="https://www.youtube.com/shorts/... (ou deixe em branco para edição sequencial de fala)",
        help="Opcional! Se informado, o ritmo musical e cortes serão copiados dessa referência. Se vazio, o editor monta o vídeo contínuo focado na sua fala e narrativa.",
    )

    # Seleção da Pasta de Saída com Explorador do SO
    st.markdown("**Pasta de Saída (Destino do Vídeo Editado)**")
    out_col1, out_col2 = st.columns([4, 1])
    with out_col1:
        output_folder = st.text_input(
            "Caminho da Pasta de Saída",
            value=st.session_state.output_folder_path,
            placeholder="/caminho/para/videos_editados",
            help="Pasta onde o arquivo MP4 renderizado e o JSON do Blueprint serão salvos.",
            label_visibility="collapsed",
        )
        st.session_state.output_folder_path = output_folder
    with out_col2:
        if st.button("📁 Procurar...", key="btn_browse_output", use_container_width=True, help="Abre o explorador de arquivos do sistema operacional para selecionar a pasta de destino"):
            chosen_dir = open_os_folder_dialog(title="Selecione a Pasta de Saída")
            if chosen_dir:
                st.session_state.output_folder_path = chosen_dir
                st.rerun()

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

        sort_by = "creation_time" if "Cronológica" in order_criteria else "alphabetical"
        video_files, audio_files = validate_input_folder(input_folder, sort_by=sort_by)
        out_dir = ensure_output_folder(output_folder)
        append_log(f"Pasta de entrada validada: {len(video_files)} vídeo(s) ordenados por {sort_by}.")

        has_ref = bool(reference_url and reference_url.strip())
        if not has_ref:
            append_log("Nenhum vídeo de referência informado. Ativando Modo Narrativa Contínua de Fala (Jump Cuts).")

        with TempWorkspace() as temp_dir:
            append_log(f"Workspace temporário criado em: {temp_dir}")

            # Módulo 1 e 2: Vídeo de Referência (se fornecido) ou Criação de Blueprint Sequencial Automático
            ref_video_path = None
            if has_ref:
                progress_bar.progress(18, text="Módulo 1/5: Baixando vídeo de referência via yt-dlp...")
                analyzer = ReferenceAnalyzer(api_key=api_key, model_name=model_name, log_fn=append_log)
                ref_video_path = analyzer.download_reference_video(reference_url.strip(), temp_dir)

                progress_bar.progress(30, text="Módulo 2/5: Extraindo cortes, BPM e gerando Blueprint com Gemini...")
                blueprint = analyzer.generate_editing_blueprint(ref_video_path, temp_dir)
            else:
                progress_bar.progress(25, text="Gerando Blueprint narrativo padrão para fala contínua...")
                blueprint = {
                    "style_summary": "Edição Contínua de Fala e Narrativa (Jump Cuts dinâmicos e cortes de silêncio/erros).",
                    "pacing_category": "natural_speech",
                    "bpm": 120.0,
                    "avg_shot_duration": 3.0,
                    "recommended_transition": "hard_cut",
                    "crossfade_duration": 0.0,
                    "framing_style": "dynamic_zoom",
                    "dynamic_zoom_intensity": 0.02,
                    "color_grading": {
                        "preset_name": "vibrant",
                        "saturation": 1.08,
                        "contrast": 1.05,
                        "brightness": 0.01,
                        "warmth": 0.02,
                    },
                    "subtitle_style": {
                        "enabled": False,
                        "position": "bottom",
                        "font_scale": "large",
                        "primary_color_hex": "#FFFFFF",
                        "stroke_color_hex": "#000000",
                        "overlay_phrases": [],
                    },
                    "cut_slots": [],
                }

            st.session_state.blueprint = blueprint

            # Salva uma cópia do Blueprint na pasta de saída para auditoria/reuso
            blueprint_path = out_dir / "editing_blueprint.json"
            blueprint_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
            append_log(f"Blueprint de Edição salvo em: {blueprint_path}")

            # Módulo 3: Detecção de Silêncio e Limpeza de Erros de Fala (Gagueira/Espirros/Repetições)
            valid_intervals_map = {}
            speech_cleaner = AudioSpeechCleaner(api_key=api_key, model_name=model_name, log_fn=append_log)

            if remove_silence or clean_speech_mistakes:
                progress_bar.progress(45, text="Módulo 3: Cortando silêncios e auditando erros de fala nos brutos...")
                for vid in video_files:
                    append_log(f"Processando áudio e fala de: {vid.name}")
                    # Extrai áudio temporário em MP3 para análise
                    temp_audio = temp_dir / f"{vid.stem}_audio.mp3"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(vid), "-vn", "-ac", "1", "-ar", "24000", "-b:a", "96k", str(temp_audio)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                    # Obtém duração total com ffprobe
                    dur_cmd = [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(vid)
                    ]
                    dur_res = subprocess.run(dur_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    try:
                        v_duration = float(dur_res.stdout.strip())
                    except Exception:
                        v_duration = 60.0

                    # 1. Cortar silêncio
                    if remove_silence:
                        silences = speech_cleaner.detect_silence_intervals(vid, noise_threshold_db=-32.0, min_silence_duration=0.5)
                        intervals = speech_cleaner.extract_clean_voice_intervals(v_duration, silences)
                        append_log(f"  -> {len(silences)} silêncios removidos. {len(intervals)} trechos com som/fala identificados.")
                    else:
                        intervals = [(0.0, v_duration)]

                    # 2. Auditar erros, repetições, gagueiras e espirros com Gemini
                    if clean_speech_mistakes and temp_audio.exists():
                        speech_res = speech_cleaner.analyze_speech_mistakes_with_gemini(temp_audio)
                        if speech_res.cuts_to_remove:
                            append_log(f"  -> Gemini identificou {len(speech_res.cuts_to_remove)} imperfeições para remover:")
                            rem_list = []
                            for err in speech_res.cuts_to_remove:
                                append_log(f"     ✂️ [{err.error_type}] {err.start_sec}s-{err.end_sec}s: {err.reason}")
                                rem_list.append((err.start_sec, err.end_sec))
                            intervals = speech_cleaner.subtract_intervals(intervals, rem_list)
                            append_log(f"  -> Intervalos válidos após limpeza de fala: {len(intervals)} bloco(s)")

                    valid_intervals_map[vid.name] = intervals

            # Módulo 4: Indexação e Curadoria das Melhores Cenas (OpenCV + Gemini Vision)
            progress_bar.progress(65, text="Módulo 4: Analisando as melhores cenas nos trechos limpos...")
            indexer = FootageIndexer(api_key=api_key, model_name=model_name, log_fn=append_log)
            avg_dur = float(blueprint.get("avg_shot_duration", 2.2))
            indexed_segments = indexer.index_footage_folder(
                video_files,
                target_avg_duration=avg_dur,
                temp_dir=temp_dir,
                use_cache=reuse_footage_cache,
                valid_intervals_map=valid_intervals_map if valid_intervals_map else None,
            )
            st.session_state.indexed_segments = indexed_segments

            # Módulo 5: Montagem, Efeitos de Cor e Equalização de Áudio (MoviePy + FFmpeg)
            progress_bar.progress(85, text="Módulo 5: Montando timeline, aplicando cor/cortes e equalizando áudio final...")
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
