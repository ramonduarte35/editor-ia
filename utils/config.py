import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Callable
import cv2
from dotenv import load_dotenv

# Desativa aceleração de hardware VAAPI/VDPAU quebrada para AV1 no OpenCV embutido
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

load_dotenv()

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg"}

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "9:16 (Reels / Shorts / TikTok)": (1080, 1920),
    "16:9 (YouTube / Horizontal)": (1920, 1080),
    "1:1 (Feed Quadrado)": (1080, 1080),
}

EXPORT_QUALITIES: Dict[str, Dict[str, str]] = {
    "Alta (1080p - Bitrate Elevado)": {"preset": "medium", "crf": "18", "bitrate": "8000k", "scale": 1.0},
    "Média (1080p - Balanceado)": {"preset": "fast", "crf": "23", "bitrate": "4500k", "scale": 1.0},
    "Rápida / Preview (720p)": {"preset": "ultrafast", "crf": "26", "bitrate": "2500k", "scale": 0.6667},
}


def get_gemini_api_key(override_key: str = "") -> str:
    """Retorna a chave da API do Gemini da UI ou do arquivo .env."""
    key = (override_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key or key == "sua-chave-gemini-api-aqui":
        raise ValueError(
            "Chave GEMINI_API_KEY não configurada. Informe na barra lateral ou defina no arquivo .env."
        )
    return key


def get_gemini_model(override_model: str = "") -> str:
    """Retorna o modelo do Gemini configurado."""
    return (override_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()


def check_ffmpeg_installed() -> Tuple[bool, str]:
    """Verifica se o binário do FFmpeg está disponível no PATH do sistema."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return (
            False,
            "FFmpeg não encontrado no PATH do sistema. Instale com `sudo apt install ffmpeg` (Linux), `brew install ffmpeg` (macOS) ou `winget install Gyan.FFmpeg` (Windows).",
        )
    return True, ffmpeg_path


def validate_input_folder(
    folder_path: str,
    sort_by: str = "alphabetical",
) -> Tuple[List[Path], List[Path]]:
    """
    Valida a pasta de entrada e retorna listas de arquivos de vídeo e de áudio encontrados.
    Ordena os vídeos de acordo com `sort_by`:
    - 'alphabetical': Ordem alfabética pelo nome do arquivo (ex: video_01.mp4, video_02.mp4)
    - 'creation_time': Ordem cronológica da gravação/modificação do arquivo
    Levanta exceções claras se a pasta não existir ou estiver vazia.
    """
    if not folder_path or not folder_path.strip():
        raise ValueError("O caminho da pasta de entrada não pode estar vazio.")

    path = Path(folder_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"A pasta de entrada não existe: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"O caminho informado não é uma pasta válida: {path}")

    videos: List[Path] = []
    audios: List[Path] = []

    for item in path.iterdir():
        if item.is_file():
            ext = item.suffix.lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                videos.append(item)
            elif ext in SUPPORTED_AUDIO_EXTENSIONS:
                audios.append(item)

    if not videos:
        raise ValueError(
            f"Nenhum arquivo de vídeo suportado ({', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}) foi encontrado em: {path}"
        )

    # Ordenação conforme preferência
    if sort_by == "creation_time":
        videos.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
    else:
        videos.sort(key=lambda p: p.name.lower())

    audios.sort(key=lambda p: p.name.lower())
    return videos, audios


def ensure_output_folder(folder_path: str) -> Path:
    """Garante que a pasta de saída existe e tem permissão de escrita."""
    if not folder_path or not folder_path.strip():
        raise ValueError("O caminho da pasta de saída não pode estar vazio.")

    path = Path(folder_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Sem permissão de escrita na pasta de saída: {path}")

    return path


class TempWorkspace:
    """Context manager para criação e limpeza garantida de arquivos temporários."""

    def __init__(self, prefix: str = "ai_video_editor_"):
        self.prefix = prefix
        self.temp_dir: Path | None = None

    def __enter__(self) -> Path:
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def ensure_opencv_compatible_video(
    video_path: Path,
    temp_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Testa se o OpenCV consegue decodificar frames reais do vídeo.
    Caso o vídeo use codec AV1 (av01) ou outro formato que falhe no decoder embutido do OpenCV,
    transcodifica automaticamente para H.264 (libx264) via FFmpeg do sistema.
    """
    log = log_fn or (lambda msg: None)
    cap = cv2.VideoCapture(str(video_path))
    can_read_frame = False

    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            can_read_frame = True
    cap.release()

    if can_read_frame:
        return video_path

    log(f"Codec incompatível com OpenCV detectado em '{video_path.name}' (ex: AV1). Convertendo para H.264 com FFmpeg...")
    converted_path = temp_dir / f"h264_{video_path.stem}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(converted_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not converted_path.exists():
        raise RuntimeError(
            f"Falha ao converter vídeo '{video_path.name}' para H.264 com FFmpeg: {res.stderr[-300:] if res.stderr else 'erro desconhecido'}"
        )

    log(f"Conversão H.264 concluída: {converted_path.name}")
    return converted_path

