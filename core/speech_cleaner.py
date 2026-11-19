import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Compatibilidade entre PyAV >= 14/19 e faster-whisper (PyAV removeu o argumento metadata_errors em versões recentes)
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _safe_av_open
except Exception:
    pass


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
        whisper_model_size: str = "base",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.log = log_fn or (lambda msg: None)
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.whisper_model_size = whisper_model_size
        self._local_whisper_model = None

    def _get_whisper_model(self):
        """Inicializa o modelo local Faster-Whisper sob demanda."""
        if self._local_whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                self.log(f"Carregando modelo local Faster-Whisper ('{self.whisper_model_size}') na memória...")
                # Tenta GPU CUDA; se não estiver configurado com cuDNN, cai graciosamente para CPU multithread
                try:
                    self._local_whisper_model = WhisperModel(self.whisper_model_size, device="cuda", compute_type="float16")
                    self.log("Faster-Whisper rodando via aceleração por GPU (CUDA)!")
                except Exception:
                    self._local_whisper_model = WhisperModel(self.whisper_model_size, device="cpu", compute_type="int8")
                    self.log("Faster-Whisper rodando via CPU local (int8 quantizado).")
            except Exception as exc:
                self.log(f"Aviso ao carregar Faster-Whisper local: {exc}")
        return self._local_whisper_model

    def analyze_speech_mistakes_local(self, audio_path: Path) -> SpeechAnalysisResult:
        """
        Transcreve o áudio 100% OFFLINE com Faster-Whisper e detecta gagueiras
        e repetições de frases utilizando alinhamento temporal por palavra.
        """
        model = self._get_whisper_model()
        if model is None:
            return SpeechAnalysisResult(has_speech=True, transcription_summary="Whisper local indisponível", cuts_to_remove=[])

        try:
            self.log(f"Transcrevendo localmente com Faster-Whisper: '{audio_path.name}'...")
            segments, info = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="pt",
                vad_filter=True,
            )

            all_words = []
            full_text_parts = []
            for segment in segments:
                full_text_parts.append(segment.text)
                if segment.words:
                    for w in segment.words:
                        clean_w = re.sub(r"[^\wáéíóúâêîôûãõç]", "", w.word.lower().strip())
                        if clean_w:
                            all_words.append({
                                "word": clean_w,
                                "raw": w.word,
                                "start": round(w.start, 2),
                                "end": round(w.end, 2),
                            })

            cuts_to_remove: List[SpeechCorrection] = []

            # 1. Detecção de Gagueira / Palavras Imediatamente Repetidas
            # Exemplo: "eu eu vou" -> remove o primeiro "eu"
            i = 0
            while i < len(all_words) - 1:
                cur = all_words[i]
                nxt = all_words[i + 1]
                if cur["word"] == nxt["word"] and (nxt["start"] - cur["end"]) < 1.0:
                    cuts_to_remove.append(
                        SpeechCorrection(
                            error_type="gagueira",
                            start_sec=cur["start"],
                            end_sec=cur["end"] + 0.05,
                            reason=f"Gagueira: palavra repetida '{cur['word']}'",
                        )
                    )
                    i += 1
                i += 1

            # 2. Detecção de Falsas Partidas / Frases Repetidas (n-gramas)
            # Exemplo: "nós vamos fazer [pausa] nós vamos fazer agora..." -> descarta a primeira tentativa
            n = 3
            if len(all_words) >= n * 2:
                for idx in range(len(all_words) - (n * 2) + 1):
                    seq1 = [w["word"] for w in all_words[idx : idx + n]]
                    # Procura se a mesma sequência de 3 palavras é reiniciada logo em seguida (em até 8 segundos)
                    for ahead in range(idx + n, min(len(all_words) - n + 1, idx + n + 12)):
                        seq2 = [w["word"] for w in all_words[ahead : ahead + n]]
                        if seq1 == seq2:
                            err_start = all_words[idx]["start"]
                            err_end = all_words[ahead - 1]["end"] + 0.1
                            if err_end - err_start >= 0.5:
                                cuts_to_remove.append(
                                    SpeechCorrection(
                                        error_type="repeticao",
                                        start_sec=err_start,
                                        end_sec=err_end,
                                        reason=f"Take refeito: repetiu a frase '{' '.join(seq1)}'",
                                    )
                                )
                            break

            summary = " ".join(full_text_parts)[:150]
            self.log(f"Transcrição local concluída: {len(all_words)} palavras detectadas, {len(cuts_to_remove)} imperfeições identificadas.")
            return SpeechAnalysisResult(
                has_speech=len(all_words) > 0,
                transcription_summary=summary,
                cuts_to_remove=cuts_to_remove,
            )
        except Exception as exc:
            self.log(f"Aviso na análise local Faster-Whisper: {exc}")
            return SpeechAnalysisResult(has_speech=True, transcription_summary="Erro no Whisper", cuts_to_remove=[])


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
        Conta com retry automático e fallback para modelos alternativos caso o Google reporte 503 (alta demanda).
        """
        if not self.client:
            return SpeechAnalysisResult(has_speech=True, transcription_summary="API Key não fornecida", cuts_to_remove=[])

        candidate_models = [self.model_name]
        for fallback in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        uploaded_file = None
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

            last_error = None
            for model_to_try in candidate_models:
                for attempt in range(2):
                    try:
                        if model_to_try != self.model_name:
                            self.log(f"Tentando modelo alternativo ({model_to_try})...")

                        response = self.client.models.generate_content(
                            model=model_to_try,
                            contents=[uploaded_file, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=SpeechAnalysisResult,
                                temperature=0.1,
                            ),
                        )
                        result = json.loads(response.text)
                        return SpeechAnalysisResult(**result)
                    except Exception as err:
                        last_error = err
                        err_str = str(err)
                        if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                            import time
                            time.sleep(2.0)
                            continue
                        else:
                            break

            raise last_error or RuntimeError("Todos os modelos falharam na análise de fala.")

        except Exception as exc:
            self.log(f"Aviso na análise de erros com Gemini ({exc}). Prosseguindo com os cortes de silêncio.")
            return SpeechAnalysisResult(has_speech=True, transcription_summary="Falha na análise", cuts_to_remove=[])
        finally:
            if uploaded_file and self.client:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

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
