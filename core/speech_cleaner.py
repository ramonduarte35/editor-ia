import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class SilenceInterval(BaseModel):
    start: float
    end: float
    duration: float


class SpeechCorrection(BaseModel):
    error_type: str = Field(description="Tipo do erro: gagueira, repeticao, falsa_partida, tosse_espirro, ruido_estranho")
    start_sec: float = Field(description="Timestamp inicial do trecho defeituoso em segundos")
    end_sec: float = Field(description="Timestamp final do trecho defeituoso em segundos")
    reason: str = Field(description="Explicação rápida do motivo do descarte (ex: repetiu a frase, gaguejou, espirrou)")


class SpeechAnalysisResult(BaseModel):
    has_speech: bool = Field(description="Se há voz humana identificável no áudio")
    transcription_summary: str = Field(description="Resumo do conteúdo falado")
    cuts_to_remove: List[SpeechCorrection] = Field(
        default_factory=list,
        description="Lista de trechos com gagueiras, repetições de frases ou ruídos para remover da edição final"
    )


class AudioSpeechCleaner:
    """
    Módulo de Limpeza e Refinamento de Fala:
    1. Detecta zonas de silêncio (pausas na fala) com FFmpeg silencedetect.
    2. Identifica erros de fala, repetições, gagueiras, espirros ou ruídos anômalos
       usando o Gemini com o áudio extraído.
    3. Retorna os intervalos limpos e válidos para a montagem final.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.log = log_fn or (lambda msg: None)
        self.client = genai.Client(api_key=api_key) if api_key else None

    def detect_silence_intervals(
        self,
        video_path: Path,
        noise_threshold_db: float = -32.0,
        min_silence_duration: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        Executa silencedetect do FFmpeg para encontrar todos os intervalos de silêncio no vídeo.
        Retorna uma lista de tuplas [(start, end), ...] com os intervalos de silêncio.
        """
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-af",
            f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}",
            "-f",
            "null",
            "-",
        ]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        stderr_output = res.stderr

        silence_starts: List[float] = []
        silences: List[Tuple[float, float]] = []

        for line in stderr_output.splitlines():
            if "silence_start:" in line:
                m = re.search(r"silence_start:\s*([0-9.]+)", line)
                if m:
                    silence_starts.append(float(m.group(1)))
            elif "silence_end:" in line:
                m = re.search(r"silence_end:\s*([0-9.]+)", line)
                if m and silence_starts:
                    start = silence_starts.pop(0)
                    end = float(m.group(1))
                    silences.append((round(start, 2), round(end, 2)))

        return silences

    def extract_clean_voice_intervals(
        self,
        total_duration: float,
        silence_intervals: List[Tuple[float, float]],
        padding_sec: float = 0.08,
        min_segment_dur: float = 0.4,
    ) -> List[Tuple[float, float]]:
        """
        Inverte os intervalos de silêncio para obter apenas os intervalos onde há fala/som ativo,
        adicionando uma pequena margem (padding) para não cortar respirações naturais ou consoantes.
        """
        if not silence_intervals:
            return [(0.0, round(total_duration, 2))]

        voice_intervals: List[Tuple[float, float]] = []
        cur_pos = 0.0

        for s_start, s_end in sorted(silence_intervals):
            # Intervalo sonoro antes do início do silêncio
            seg_start = cur_pos
            seg_end = min(total_duration, s_start + padding_sec)

            if seg_end - seg_start >= min_segment_dur:
                voice_intervals.append((round(seg_start, 2), round(seg_end, 2)))

            cur_pos = max(0.0, s_end - padding_sec)

        # Trecho final após o último silêncio
        if total_duration - cur_pos >= min_segment_dur:
            voice_intervals.append((round(cur_pos, 2), round(total_duration, 2)))

        return voice_intervals

    def analyze_speech_mistakes_with_gemini(
        self,
        audio_path: Path,
    ) -> SpeechAnalysisResult:
        """
        Envia o áudio para o Gemini para identificar erros de fala, repetições,
        gagueiras, espirros ou falsas partidas com os timestamps exatos a remover.
        """
        if not self.client:
            return SpeechAnalysisResult(has_speech=True, transcription_summary="API Key não fornecida", cuts_to_remove=[])

        try:
            self.log(f"Enviando áudio de '{audio_path.name}' para Gemini auditar erros de fala...")
            uploaded_file = self.client.files.upload(file=str(audio_path))

            prompt = (
                "Você é um editor de vídeo e áudio profissional focado em edição de fala/podcast/vlog.\n"
                "Ouça atentamente o áudio gravado e analise:\n"
                "1. Se há alguém falando.\n"
                "2. Identifique e marque com precisão cirúrgica de timestamps (em segundos):\n"
                "   - Gagueiras e travamentos severos\n"
                "   - Repetições de frases (quando a pessoa erra uma frase, recomeça e refaz o take)\n"
                "   - Pigarros, tosse, espirros ou barulhos anômalos que devem ser cortados\n"
                "3. Para cada erro encontrado, indique `start_sec` e `end_sec` do trecho exato que deve ser REMOVIDO na edição.\n"
                "Se a fala estiver fluida e sem erros, deixe `cuts_to_remove` vazio."
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SpeechAnalysisResult,
                    temperature=0.1,
                ),
            )

            # Limpeza do arquivo no Google AI Studio
            try:
                self.client.files.delete(name=uploaded_file.name)
            except Exception:
                pass

            result = json.loads(response.text)
            return SpeechAnalysisResult(**result)
        except Exception as exc:
            self.log(f"Aviso na análise de erros com Gemini ({exc}). Prosseguindo sem cortes de fala.")
            return SpeechAnalysisResult(has_speech=True, transcription_summary="Falha na análise", cuts_to_remove=[])

    def subtract_intervals(
        self,
        base_intervals: List[Tuple[float, float]],
        remove_intervals: List[Tuple[float, float]],
        min_duration: float = 0.4,
    ) -> List[Tuple[float, float]]:
        """
        Subtrai trechos com defeitos (gagueira, repetição, espirros) dos trechos válidos de fala.
        """
        current = list(base_intervals)

        for r_start, r_end in remove_intervals:
            new_current: List[Tuple[float, float]] = []
            for b_start, b_end in current:
                # Caso 1: Sem sobreposição
                if r_end <= b_start or r_start >= b_end:
                    new_current.append((b_start, b_end))
                else:
                    # Sobreposição: divide ou encurta
                    if r_start > b_start and (r_start - b_start) >= min_duration:
                        new_current.append((b_start, round(r_start, 2)))
                    if r_end < b_end and (b_end - r_end) >= min_duration:
                        new_current.append((round(r_end, 2), b_end))
            current = new_current

        return sorted(current, key=lambda x: x[0])
