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


class VoiceSegmentAI(BaseModel):
    start_sec: float = Field(description="Timestamp exato onde o piloto começou a falar em segundos")
    end_sec: float = Field(description="Timestamp exato onde o piloto terminou de falar em segundos")
    spoken_text: str = Field(description="O que foi dito nesse trecho")
    is_valid_take: bool = Field(description="True se o take for bom; False se for gagueira ou erro para descartar", default=True)


class GeminiSpeechDetectionResult(BaseModel):
    has_speech: bool = Field(description="True se o piloto fala algo inteligível no vídeo")
    summary: str = Field(description="Resumo geral do que o piloto falou no vídeo")
    voice_segments: List[VoiceSegmentAI] = Field(
        default_factory=list,
        description="Lista de todos os trechos onde há voz humana ativa do piloto falando",
    )


class SpeechAnalysisResult(BaseModel):
    has_speech: bool = Field(description="Se há voz humana identificável no áudio")
    transcription_summary: str = Field(description="Resumo do conteúdo falado")
    cuts_to_remove: List[SpeechCorrection] = Field(
        default_factory=list,
        description="Lista de trechos com gagueiras, repetições de frases ou ruídos para remover da edição final"
    )
    voice_intervals: List[Tuple[float, float]] = Field(
        default_factory=list,
        description="Intervalos de fala aprovados",
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
        whisper_model_size: str = "tiny",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.log = log_fn or (lambda msg: None)
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.whisper_model_size = whisper_model_size
        self._local_whisper_model = None

    def _get_whisper_model(self):
        """Inicializa o modelo local Faster-Whisper sob demanda com controle estrito de threads de CPU."""
        if self._local_whisper_model is None:
            try:
                import os
                from faster_whisper import WhisperModel
                # Limita a 4 threads de CPU no máximo para deixar os outros 28 núcleos livres para a interface e SO
                num_cpu_threads = min(4, max(1, (os.cpu_count() or 4) // 4))

                self.log(f"Carregando modelo local Faster-Whisper ('{self.whisper_model_size}')...")
                try:
                    self._local_whisper_model = WhisperModel(
                        self.whisper_model_size,
                        device="cuda",
                        compute_type="float16",
                    )
                    self.log("Faster-Whisper rodando via aceleração por GPU (CUDA - RTX)!")
                except Exception as cuda_err:
                    self.log(f"Fallback para CPU ({num_cpu_threads} threads suaves): {cuda_err}")
                    self._local_whisper_model = WhisperModel(
                        self.whisper_model_size,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=num_cpu_threads,
                        num_workers=1,
                    )
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
            self.log(f"Transcrevendo áudio com Faster-Whisper ({self.whisper_model_size}): '{audio_path.name}'...")
            segments, info = model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language="pt",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
                beam_size=1,  # Greedy search ultra leve (reduz uso de CPU/RAM em 80%)
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


    def extract_voice_isolated_audio(self, video_path: Path, output_audio: Path) -> Path:
        """
        Isola as frequências da voz humana e filtra o barulho do motor da moto e do vento:
        - highpass=f=200: remove o grave profundo do motor e o ronco do escapamento (< 200 Hz)
        - lowpass=f=3600: remove ruídos agudos de atrito e assobio de vento (> 3.6 kHz)
        - afftdn=nr=18:nf=-30: aplica redução adaptativa de ruído de fundo constante (FFT Denoise)
        - arnndn: filtro neural se disponível ou afftdn
        """
        filter_chain = "highpass=f=200,lowpass=f=3600,afftdn=nr=18:nf=-28,volume=2.0"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            filter_chain,
            "-b:a",
            "96k",
            str(output_audio),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_audio

    def detect_silence_intervals(
        self,
        audio_or_video_path: Path,
        noise_threshold_db: float = -28.0,
        min_silence_duration: float = 0.6,
    ) -> List[Tuple[float, float]]:
        """
        Executa silencedetect do FFmpeg no áudio com filtro passa-faixa de voz.
        Retorna uma lista de tuplas [(start, end), ...] com os intervalos de silêncio.
        """
        filter_str = f"highpass=f=200,lowpass=f=3500,silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}"
        cmd = [
            "ffmpeg",
            "-i",
            str(audio_or_video_path),
            "-af",
            filter_str,
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
        padding_start: float = 0.25,
        padding_end: float = 0.35,
        min_segment_dur: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        Inverte os intervalos de silêncio para obter apenas os intervalos onde há fala.
        Aplica margem generosa (padding de segurança):
        - padding_start: 0.25s antes da primeira palavra (evita comer o início da frase/respiração)
        - padding_end: 0.35s após a última palavra (evita comer a última sílaba da fala)
        """
        if not silence_intervals:
            return [(0.0, round(total_duration, 2))]

        voice_intervals: List[Tuple[float, float]] = []
        cur_pos = 0.0

        for s_start, s_end in sorted(silence_intervals):
            # Intervalo sonoro antes do início do silêncio
            seg_start = max(0.0, cur_pos - padding_start) if cur_pos > 0 else 0.0
            seg_end = min(total_duration, s_start + padding_end)

            if seg_end - seg_start >= min_segment_dur:
                voice_intervals.append((round(seg_start, 2), round(seg_end, 2)))

            cur_pos = max(0.0, s_end - padding_start)

        # Trecho final após o último silêncio
        if total_duration - cur_pos >= min_segment_dur:
            seg_start = max(0.0, cur_pos - padding_start)
            voice_intervals.append((round(seg_start, 2), round(total_duration, 2)))

        # Funde intervalos com sobreposição gerada pelas margens (merge overlapping)
        merged: List[Tuple[float, float]] = []
        for start, end in sorted(voice_intervals, key=lambda x: x[0]):
            if not merged:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                if start <= prev_end + 0.15:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))

        return merged

    def detect_voice_intervals_with_gemini(
        self,
        audio_path: Path,
        total_duration: float,
        padding_start: float = 0.35,
        padding_end: float = 0.45,
    ) -> List[Tuple[float, float]]:
        """
        Usa o raciocínio acústico da IA multimodal do Gemini para diferenciar voz humana
        do ruído de motor de moto e vento, identificando exatamente os trechos onde há fala.
        """
        if not self.client:
            return [(0.0, total_duration)]

        uploaded_file = None
        candidate_models = [self.model_name]
        for fallback in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.5-flash-lite"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        try:
            self.log(f"🧠 Enviando áudio para IA do Gemini identificar trechos onde você fala (ignorando moto/vento)...")
            uploaded_file = self.client.files.upload(file=str(audio_path))

            prompt = (
                "Você é um editor de áudio especializado em Motovlog.\n"
                "Neste áudio há muito barulho constante de motor de moto, escapamento e vento.\n"
                "Sua missão é ignorar completamente o som da moto/vento e identificar com precisão temporal os momentos em que o PILOTO ESTÁ FALANDO.\n"
                "Para cada trecho onde o piloto fala, informe:\n"
                "- start_sec: segundo exato onde começa a fala\n"
                "- end_sec: segundo exato onde termina a fala\n"
                "- spoken_text: o que ele disse\n"
                "- is_valid_take: false se foi uma gagueira severa ou recomeço de frase, true se for fala válida.\n"
                "Se não houver voz humana falada em nenhum momento, retorne has_speech = false e lista vazia."
            )

            last_error = None
            for model_to_try in candidate_models:
                for attempt in range(2):
                    try:
                        response = self.client.models.generate_content(
                            model=model_to_try,
                            contents=[uploaded_file, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=GeminiSpeechDetectionResult,
                                temperature=0.1,
                            ),
                        )
                        result = json.loads(response.text)
                        segments = result.get("voice_segments") or []
                        raw_intervals: List[Tuple[float, float]] = []

                        for seg in segments:
                            if seg.get("is_valid_take", True):
                                s_start = max(0.0, float(seg["start_sec"]) - padding_start)
                                s_end = min(total_duration, float(seg["end_sec"]) + padding_end)
                                if s_end - s_start >= 0.5:
                                    raw_intervals.append((round(s_start, 2), round(s_end, 2)))

                        # Mescla sobreposições
                        merged: List[Tuple[float, float]] = []
                        for start, end in sorted(raw_intervals, key=lambda x: x[0]):
                            if not merged:
                                merged.append((start, end))
                            else:
                                prev_start, prev_end = merged[-1]
                                if start <= prev_end + 0.2:
                                    merged[-1] = (prev_start, max(prev_end, end))
                                else:
                                    merged.append((start, end))

                        self.log(f"🧠 IA do Gemini localizou {len(segments)} trecho(s) de fala no motovlog.")
                        for s in segments:
                            status_icon = "✅" if s.get("is_valid_take", True) else "❌ (descartado)"
                            self.log(f"   {status_icon} [{s.get('start_sec', 0):.1f}s - {s.get('end_sec', 0):.1f}s]: '{s.get('spoken_text', '')}'")

                        if not raw_intervals:
                            self.log("⚠️ Nenhuma fala clara detectada pelo Gemini. Mantendo o vídeo original sem cortes.")
                            return [(0.0, total_duration)]

                        return merged
                    except Exception as err:
                        last_error = err
                        import time
                        time.sleep(2.0)
                        continue

            raise last_error or RuntimeError("Falha na chamada Gemini")
        except Exception as exc:
            self.log(f"Aviso no Gemini Speech Detection ({exc}). Mantendo áudio completo.")
            return [(0.0, total_duration)]
        finally:
            if uploaded_file and self.client:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

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
