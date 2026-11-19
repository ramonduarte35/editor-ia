import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import cv2
import librosa
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import yt_dlp
from google import genai
from google.genai import types
from utils.config import ensure_opencv_compatible_video


class CutSlot(BaseModel):
    slot_index: int = Field(description="Índice sequencial do take na timeline (0, 1, 2...)")
    duration_sec: float = Field(description="Duração ideal deste take em segundos")
    shot_role: str = Field(description="Papel narrativo do plano: hook, buildup, climax, b-roll ou outro")
    energy_level: str = Field(description="Nível de energia visual exigido: low, medium ou high")
    transition_in: str = Field(description="Tipo de transição de entrada: hard_cut, crossfade ou flash")


class ColorGradingSpec(BaseModel):
    preset_name: str = Field(description="Estilo de cor: vibrant, cinematic_warm, cool_teal, high_contrast, bw_punch")
    saturation: float = Field(description="Multiplicador de saturação (ex: 0.8 a 1.4)", default=1.1)
    contrast: float = Field(description="Multiplicador de contraste (ex: 0.9 a 1.3)", default=1.08)
    brightness: float = Field(description="Ajuste de brilho (ex: -0.10 a 0.10)", default=0.02)
    warmth: float = Field(description="Balanço quente/frio (-0.2 frio a +0.2 quente)", default=0.0)


class SubtitleStyleSpec(BaseModel):
    enabled: bool = Field(description="Se o vídeo de referência possui legendas/textos dinâmicos na tela")
    position: str = Field(description="Posição vertical das legendas: center, bottom ou top", default="bottom")
    font_scale: str = Field(description="Tamanho relativo da fonte: small, medium, large", default="large")
    primary_color_hex: str = Field(description="Cor principal do texto em hexadecimal (ex: #FFFFFF ou #FFD700)", default="#FFFFFF")
    stroke_color_hex: str = Field(description="Cor da borda do texto em hexadecimal", default="#000000")
    overlay_phrases: List[str] = Field(
        description="Frases curtas de impacto sugeridas para sobrepor aos cortes principais (gancho, retenção, fechamento)",
        default_factory=list,
    )


class EditingBlueprint(BaseModel):
    style_summary: str = Field(description="Resumo executivo do estilo de edição, ritmo e estética da referência")
    pacing_category: str = Field(description="Classificação do ritmo: fast_paced, rhythmic_medium, cinematic_slow")
    bpm: float = Field(description="BPM musical ou rítmico detectado na referência")
    avg_shot_duration: float = Field(description="Duração média dos planos em segundos")
    recommended_transition: str = Field(description="Transição predominante: hard_cut ou crossfade")
    crossfade_duration: float = Field(description="Duração do crossfade em segundos (0.0 se corte seco)", default=0.0)
    framing_style: str = Field(description="Estilo de enquadramento: dynamic_zoom, center_subject, stable_wide")
    dynamic_zoom_intensity: float = Field(description="Intensidade do zoom dinâmico (0.0 a 0.12)", default=0.04)
    color_grading: ColorGradingSpec
    subtitle_style: SubtitleStyleSpec
    cut_slots: List[CutSlot] = Field(description="Lista ordenada de takes que compõem a estrutura rítmica do vídeo")


class ReferenceAnalyzer:
    """
    Módulo 1 & 2:
    Faz o download do vídeo de referência via yt-dlp e extrai métricas objetivas
    (PySceneDetect + Librosa) combinadas com análise multimodal via Gemini API
    para gerar o Blueprint de Edição em JSON.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash", log_fn: Optional[Callable[[str], None]] = None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.log = log_fn or (lambda msg: None)

    def download_reference_video(self, url_or_path: str, temp_dir: Path) -> Path:
        """
        Módulo 1: Baixa o vídeo de referência usando yt-dlp (ou copia se já for um arquivo local).
        """
        candidate_local = Path(url_or_path).expanduser()
        if candidate_local.exists() and candidate_local.is_file():
            self.log(f"Usando arquivo local de referência: {candidate_local.name}")
            return ensure_opencv_compatible_video(candidate_local, temp_dir, log_fn=self.log)

        self.log(f"Iniciando download do vídeo de referência via yt-dlp: {url_or_path}")
        output_template = str(temp_dir / "reference_video.%(ext)s")

        # Prioriza codec H.264 (avc1) e evita AV1 (av01) para compatibilidade total com OpenCV e PySceneDetect
        ydl_opts = {
            "format": (
                "bestvideo[height<=1080][vcodec^=avc]+bestaudio[ext=m4a]/"
                "bestvideo[height<=1080][vcodec!^=av01][ext=mp4]+bestaudio[ext=m4a]/"
                "best[height<=1080][vcodec^=avc]/"
                "best[height<=1080][ext=mp4]/best"
            ),
            "format_sort": ["vcodec:h264", "res:1080", "ext:mp4:m4a"],
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_or_path])
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao baixar o vídeo de referência ({url_or_path}). Verifique se o link é público e válido. Detalhe: {exc}"
            ) from exc

        downloaded_files = list(temp_dir.glob("reference_video.*"))
        if not downloaded_files:
            raise FileNotFoundError("O download via yt-dlp terminou, mas nenhum arquivo de vídeo foi gerado.")

        video_path = downloaded_files[0]
        self.log(f"Download concluído: {video_path.name} ({video_path.stat().st_size / (1024 * 1024):.1f} MB)")
        return ensure_opencv_compatible_video(video_path, temp_dir, log_fn=self.log)

    def extract_objective_metrics(self, video_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Módulo 2 (Parte A): Usa PySceneDetect e Librosa para medir cortes, duração dos planos e BPM/beats.
        """
        self.log("Detectando cortes de cena com PySceneDetect...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo de referência com OpenCV: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = frame_count / fps if fps > 0 else 15.0
        cap.release()

        # Detecção de cenas com PySceneDetect
        shot_durations: List[float] = []
        cut_timestamps: List[float] = []
        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=24.0, min_scene_len=int(max(8, fps * 0.35))))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            for start_tc, end_tc in scene_list:
                start_sec = start_tc.get_seconds()
                end_sec = end_tc.get_seconds()
                dur = max(0.35, round(end_sec - start_sec, 2))
                shot_durations.append(dur)
                cut_timestamps.append(round(start_sec, 2))
        except Exception as exc:
            self.log(f"Aviso no PySceneDetect (aplicando fallback rítmico): {exc}")

        if not shot_durations:
            # Caso o vídeo seja um plano sequência ou corte suave não detectado
            step = max(1.5, min(3.0, total_duration / 6.0))
            cur = 0.0
            while cur < total_duration:
                nxt = min(total_duration, cur + step)
                if nxt - cur >= 0.4:
                    shot_durations.append(round(nxt - cur, 2))
                    cut_timestamps.append(round(cur, 2))
                cur = nxt

        avg_shot_duration = float(np.mean(shot_durations)) if shot_durations else 2.0

        # Análise de áudio e BPM com Librosa
        self.log("Analisando trilha sonora e BPM com Librosa...")
        bpm = 120.0
        beat_times: List[float] = []
        audio_wav = temp_dir / "ref_audio.wav"
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "22050",
                    "-t",
                    "90",
                    str(audio_wav),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            if audio_wav.exists():
                y, sr = librosa.load(str(audio_wav), sr=22050, mono=True)
                if len(y) > sr:
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                    bpm_val = float(np.ndim(tempo) and np.asarray(tempo).item() or tempo)
                    if 50.0 <= bpm_val <= 220.0:
                        bpm = round(bpm_val, 1)
                    beat_times = [round(float(t), 2) for t in librosa.frames_to_time(beats, sr=sr)]
        except Exception as exc:
            self.log(f"Aviso na extração de BPM com Librosa (usando 120 BPM padrão): {exc}")

        metrics = {
            "total_duration_sec": round(total_duration, 2),
            "resolution": f"{width}x{height}",
            "aspect_ratio_detected": "9:16" if height > width else ("16:9" if width > height else "1:1"),
            "num_cuts": len(shot_durations),
            "avg_shot_duration_sec": round(avg_shot_duration, 2),
            "min_shot_duration_sec": round(float(min(shot_durations)), 2),
            "max_shot_duration_sec": round(float(max(shot_durations)), 2),
            "shot_durations_sec": shot_durations[:25],
            "cut_timestamps_sec": cut_timestamps[:25],
            "bpm": bpm,
            "beat_count": len(beat_times),
            "sample_beat_timestamps_sec": beat_times[:30],
        }
        self.log(
            f"Métricas extraídas: {metrics['num_cuts']} cenas | Duração média: {metrics['avg_shot_duration_sec']}s | BPM: {metrics['bpm']}"
        )
        return metrics

    def extract_representative_frames(self, video_path: Path, max_frames: int = 6) -> List[Image.Image]:
        """Extrai frames-chave distribuídos ao longo do vídeo de referência para inspeção visual do Gemini."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        indices = np.linspace(int(total_frames * 0.05), int(total_frames * 0.92), num=min(max_frames, max(1, total_frames)), dtype=int)
        frames: List[Image.Image] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pil_img.thumbnail((640, 640))
                frames.append(pil_img)

        cap.release()
        return frames

    def generate_editing_blueprint(self, video_path: Path, temp_dir: Path) -> Dict[str, Any]:
        """
        Módulo 2 (Parte B): Combina métricas objetivas + frames-chave e solicita ao Gemini
        a geração estruturada do Blueprint de Edição em JSON.
        """
        metrics = self.extract_objective_metrics(video_path, temp_dir)
        keyframes = self.extract_representative_frames(video_path, max_frames=6)

        self.log(f"Consultando Gemini ({self.model_name}) para construir o Blueprint de Edição...")

        prompt = f"""
Você é um Diretor de Edição e Colorista Sênior. Analise os frames-chave extraídos de um vídeo de referência e as métricas objetivas medidas via PySceneDetect e Librosa abaixo:

MÉTRICAS OBJETIVAS DA REFERÊNCIA:
{json.dumps(metrics, indent=2, ensure_ascii=False)}

SUA TAREFA:
Gere um "Blueprint de Edição" completo e fiel ao ritmo e estilo visual dessa referência para ser aplicado automaticamente a novos vídeos brutos.
Regras para os `cut_slots`:
1. Crie uma sequência de takes (`cut_slots`) que reflita o ritmo real medido em `shot_durations_sec` (entre 4 e 16 takes no total).
2. Cada `duration_sec` deve respeitar o ritmo da referência (mínimo 0.6s, máximo 6.0s).
3. Classifique o papel de cada take (`hook`, `buildup`, `climax`, `b-roll`, `outro`) e o nível de energia (`low`, `medium`, `high`).
4. Defina a paleta de cor (`color_grading`) com valores numéricos realistas de saturação (0.8 a 1.35), contraste (0.95 a 1.25), brilho (-0.08 a 0.08) e temperatura/warmth (-0.15 a 0.15).
5. Se os frames mostrarem texto/legendas na tela, ative `subtitle_style.enabled = true` e sugira de 3 a 6 frases curtas de impacto em português em `overlay_phrases` compatíveis com o estilo dinâmico do vídeo.
"""

        contents: List[Any] = [prompt]
        contents.extend(keyframes)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EditingBlueprint,
                    temperature=0.2,
                ),
            )
            blueprint_data = json.loads(response.text)
            blueprint_data["objective_metrics"] = metrics
            self.log("Blueprint de Edição gerado com sucesso pelo Gemini!")
            return blueprint_data
        except Exception as exc:
            self.log(f"Aviso: Falha na resposta estruturada da API Gemini ({exc}). Gerando Blueprint determinístico a partir das métricas...")
            return self._build_fallback_blueprint(metrics)

    def _build_fallback_blueprint(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback caso a chamada de rede/API sofra limite de cota ou indisponibilidade."""
        durations = metrics.get("shot_durations_sec") or [2.0, 1.8, 1.5, 2.2, 1.6, 2.5]
        avg_dur = metrics.get("avg_shot_duration_sec", 2.0)
        pacing = "fast_paced" if avg_dur < 1.8 else ("rhythmic_medium" if avg_dur < 3.2 else "cinematic_slow")

        slots = []
        for idx, dur in enumerate(durations[:12]):
            role = "hook" if idx == 0 else ("outro" if idx == len(durations[:12]) - 1 else ("climax" if idx % 3 == 0 else "buildup"))
            energy = "high" if dur < 1.8 else ("medium" if dur < 3.0 else "low")
            slots.append(
                {
                    "slot_index": idx,
                    "duration_sec": float(np.clip(dur, 0.7, 5.5)),
                    "shot_role": role,
                    "energy_level": energy,
                    "transition_in": "hard_cut" if pacing == "fast_paced" else "crossfade",
                }
            )

        return {
            "style_summary": f"Edição rítmica ({pacing}) com média de {avg_dur:.2f}s por corte e sincronia em {metrics.get('bpm', 120)} BPM.",
            "pacing_category": pacing,
            "bpm": metrics.get("bpm", 120.0),
            "avg_shot_duration": avg_dur,
            "recommended_transition": "hard_cut" if pacing == "fast_paced" else "crossfade",
            "crossfade_duration": 0.0 if pacing == "fast_paced" else 0.25,
            "framing_style": "dynamic_zoom",
            "dynamic_zoom_intensity": 0.05,
            "color_grading": {
                "preset_name": "vibrant",
                "saturation": 1.15,
                "contrast": 1.08,
                "brightness": 0.02,
                "warmth": 0.03,
            },
            "subtitle_style": {
                "enabled": False,
                "position": "bottom",
                "font_scale": "large",
                "primary_color_hex": "#FFFFFF",
                "stroke_color_hex": "#000000",
                "overlay_phrases": [],
            },
            "cut_slots": slots,
            "objective_metrics": metrics,
        }
