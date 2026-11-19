import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from utils.config import ensure_opencv_compatible_video


class SegmentEvaluation(BaseModel):
    segment_id: str = Field(description="Identificador do segmento enviado")
    visual_interest_score: float = Field(description="Nota de 0 a 10 para interesse visual, composição e ação")
    energy_level: str = Field(description="Nível de energia da cena: low, medium ou high")
    recommended_role: str = Field(description="Papel ideal na edição: hook, buildup, climax, b-roll ou outro")
    caption_suggestion: str = Field(description="Frase curta opcional (2 a 5 palavras) descrevendo a ação ou impacto")
    description: str = Field(description="Descrição curta do conteúdo visual do trecho")


class BatchSegmentEvaluation(BaseModel):
    evaluations: List[SegmentEvaluation]


class FootageIndexer:
    """
    Módulo 3 (Indexação dos Arquivos Brutos):
    Analisa os vídeos da pasta de entrada com OpenCV (nitidez/Laplaciano, iluminação,
    estabilidade/movimento e centro de ação) e Gemini Vision para pontuar e catalogar
    os melhores trechos com timestamps exatos.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", log_fn: Optional[Callable[[str], None]] = None):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.log = log_fn or (lambda msg: None)

    def _compute_folder_fingerprint(self, video_files: List[Path]) -> str:
        """Gera uma assinatura única baseada nos arquivos de vídeo brutos (caminho, tamanho e mtime)."""
        hasher = hashlib.sha256()
        for vf in sorted(video_files, key=lambda p: str(p)):
            stat = vf.stat()
            hasher.update(f"{vf.name}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8"))
        return hasher.hexdigest()[:16]

    def index_footage_folder(
        self,
        video_files: List[Path],
        target_avg_duration: float = 2.5,
        temp_dir: Optional[Path] = None,
        use_cache: bool = True,
        valid_intervals_map: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Varre todos os vídeos de entrada (respeitando a ordem fornecida),
        divide em janelas candidatas apenas dentro dos intervalos sonoros válidos
        (sem silêncio, sem gagueira/erros), mede nitidez/iluminação/estabilidade
        com OpenCV e classifica os melhores segmentos com Gemini Vision.
        Se já houver cache válido na pasta de entrada, reaproveita a leitura instantaneamente.
        """
        if not video_files:
            raise ValueError("Lista de vídeos de entrada vazia.")

        input_dir = video_files[0].parent
        cache_dir = input_dir / ".ai_editor_cache"
        fingerprint = self._compute_folder_fingerprint(video_files)
        # Se houver filtro de intervalos válidos, compõe a chave de cache
        cache_suffix = f"_{len(valid_intervals_map)}" if valid_intervals_map else ""
        cache_file = cache_dir / f"footage_index_{fingerprint}{cache_suffix}.json"

        if use_cache and cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                segments = cached_data.get("segments") or []
                # Verifica se todos os arquivos referenciados no cache ainda existem
                if segments and all(Path(s["source_path"]).exists() for s in segments):
                    self.log(
                        f"⚡ Cache encontrado! Reaproveitando leitura e notas de {len(segments)} trechos dos vídeos brutos "
                        f"(sem precisar reprocessar no OpenCV/Gemini)."
                    )
                    return segments
            except Exception as exc:
                self.log(f"Aviso ao ler cache de vídeos brutos (reindexando): {exc}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        all_candidates: List[Dict[str, Any]] = []
        window_sec = float(np.clip(target_avg_duration * 1.35, 1.5, 5.0))

        for vid_idx, video_path in enumerate(video_files):
            self.log(f"[{vid_idx + 1}/{len(video_files)}] Analisando arquivo bruto ordenado com OpenCV: {video_path.name}")
            # Salva eventuais conversões AV1 -> H.264 no cache_dir persistente para continuar válido entre execuções
            compatible_path = ensure_opencv_compatible_video(video_path, cache_dir, log_fn=self.log)
            allowed_ranges = (
                valid_intervals_map.get(video_path.name) or valid_intervals_map.get(str(video_path))
                if valid_intervals_map
                else None
            )
            segments = self._scan_video_opencv(
                compatible_path,
                window_sec=window_sec,
                display_name=video_path.name,
                allowed_intervals=allowed_ranges,
            )
            all_candidates.extend(segments)

        if not all_candidates:
            raise RuntimeError("Nenhum trecho válido pôde ser extraído dos vídeos de entrada.")

        # Ordena preliminarmente pela pontuação técnica do OpenCV e seleciona os melhores candidatos para o Gemini Vision
        all_candidates.sort(key=lambda x: x["technical_score"], reverse=True)
        top_candidates = all_candidates[:24]

        self.log(f"Avaliando semanticamente os {len(top_candidates)} melhores trechos com Gemini Vision ({self.model_name})...")
        enriched_candidates = self._evaluate_with_gemini_vision(top_candidates)

        # Combina nota técnica (40%) + nota semântica do Gemini (60%)
        for item in enriched_candidates:
            item["final_score"] = round(0.4 * item["technical_score"] + 0.6 * item.get("visual_interest_score", 7.0), 2)

        # Mantém também os demais candidatos avaliados tecnicamente (útil para vídeos longos com muitos cortes)
        remaining_candidates = all_candidates[24:]
        for item in remaining_candidates:
            item.pop("_thumbnail", None)
            item["final_score"] = item["technical_score"]

        full_catalog = enriched_candidates + remaining_candidates
        full_catalog.sort(key=lambda x: x["final_score"], reverse=True)

        try:
            cache_file.write_text(
                json.dumps({"fingerprint": fingerprint, "segments": full_catalog}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.log(f"💾 Leitura dos vídeos brutos salva em cache ({cache_file.name}) para reaproveitamento futuro.")
        except Exception as exc:
            self.log(f"Aviso: não foi possível salvar cache em disco: {exc}")

        self.log(f"Indexação concluída: {len(full_catalog)} trechos catalogados e ranqueados.")
        return full_catalog

    def _scan_video_opencv(
        self,
        video_path: Path,
        window_sec: float = 2.5,
        display_name: Optional[str] = None,
        allowed_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Divide um vídeo em segmentos e calcula:
        - Nitidez (variância do Laplaciano)
        - Exposição / Iluminação ideal (evita sub/superexposição)
        - Movimento / Estabilidade (diferença entre frames adjacentes: penaliza tremores extremos ou imagem congelada)
        - Centro de gravidade da ação (coordenada X normalizada para Smart Crop)
        Se `allowed_intervals` for fornecido, apenas trechos contidos dentro dos intervalos são extraídos.
        """
        shown_name = display_name or video_path.name
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.log(f"Aviso: Não foi possível abrir {shown_name}, ignorando.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        if duration < 0.6:
            cap.release()
            return []

        step_sec = max(1.0, window_sec * 0.85)
        segments: List[Dict[str, Any]] = []
        seg_counter = 0

        # Se houver intervalos permitidos de fala contínua, iteramos sobre cada bloco permitido
        target_windows: List[Tuple[float, float]] = []
        if allowed_intervals:
            for int_start, int_end in allowed_intervals:
                c_start = int_start
                while c_start < int_end:
                    c_end = min(int_end, c_start + window_sec)
                    if c_end - c_start >= 0.5:
                        target_windows.append((c_start, c_end))
                    c_start += step_sec
        else:
            cur_start = 0.0
            while cur_start < duration:
                cur_end = min(duration, cur_start + window_sec)
                if cur_end - cur_start < 0.6 and cur_start > 0:
                    break
                target_windows.append((cur_start, cur_end))
                cur_start += step_sec

        for cur_start, cur_end in target_windows:
            # Amostra 4 frames dentro da janela
            sample_times = np.linspace(cur_start + 0.05, cur_end - 0.05, num=4)
            sharpness_vals = []
            brightness_vals = []
            motion_vals = []
            focus_x_vals = []
            prev_gray = None
            mid_pil_image: Optional[Image.Image] = None

            for s_idx, t in enumerate(sample_times):
                frame_idx = int(min(total_frames - 1, max(0, t * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                small = cv2.resize(frame, (320, int(320 * height / max(1, width))))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                # 1. Nitidez via Laplaciano
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                sharpness_vals.append(lap_var)

                # 2. Iluminação média (0..255)
                mean_luma = float(np.mean(gray))
                brightness_vals.append(mean_luma)

                # 3. Movimento e centro de atenção horizontal
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion_vals.append(float(np.mean(diff)))
                    col_energy = np.sum(diff, axis=0).astype(np.float32)
                    if np.sum(col_energy) > 1e-3:
                        x_coords = np.linspace(0.0, 1.0, len(col_energy))
                        center_x = float(np.sum(x_coords * col_energy) / np.sum(col_energy))
                        focus_x_vals.append(center_x)
                prev_gray = gray

                if s_idx == 1:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    pil_img.thumbnail((512, 512))
                    mid_pil_image = pil_img

            if sharpness_vals and mid_pil_image is not None:
                avg_sharp = float(np.mean(sharpness_vals))
                avg_luma = float(np.mean(brightness_vals))
                avg_motion = float(np.mean(motion_vals)) if motion_vals else 4.0
                focus_x = float(np.clip(np.mean(focus_x_vals), 0.25, 0.75)) if focus_x_vals else 0.5

                # Normaliza notas de 0 a 10
                sharp_score = float(np.clip(avg_sharp / 35.0, 1.5, 10.0))
                # Iluminação ideal entre 70 e 185
                luma_penalty = abs(avg_luma - 128.0) / 128.0
                luma_score = float(np.clip(10.0 * (1.0 - luma_penalty * 0.85), 1.0, 10.0))
                # Estabilidade: movimento moderado (2 a 18) é ótimo; > 32 indica tremor brusco
                if avg_motion > 32.0:
                    stability_score = max(1.0, 10.0 - (avg_motion - 32.0) * 0.3)
                elif avg_motion < 0.8:
                    stability_score = 6.0
                else:
                    stability_score = 9.2

                tech_score = round(0.40 * sharp_score + 0.30 * luma_score + 0.30 * stability_score, 2)
                energy_est = "high" if avg_motion >= 11.0 else ("medium" if avg_motion >= 4.0 else "low")

                seg_id = f"{Path(shown_name).stem}_seg{seg_counter:02d}"
                segments.append(
                    {
                        "segment_id": seg_id,
                        "source_path": str(video_path),
                        "file_name": shown_name,
                        "start_sec": round(cur_start, 2),
                        "end_sec": round(cur_end, 2),
                        "duration_sec": round(cur_end - cur_start, 2),
                        "width": width,
                        "height": height,
                        "sharpness": round(avg_sharp, 1),
                        "brightness": round(avg_luma, 1),
                        "motion": round(avg_motion, 2),
                        "focus_x": round(focus_x, 3),
                        "technical_score": tech_score,
                        "energy_level": energy_est,
                        "recommended_role": "buildup",
                        "visual_interest_score": tech_score,
                        "caption_suggestion": "",
                        "description": f"Trecho de {shown_name} ({cur_start:.1f}s-{cur_end:.1f}s)",
                        "_thumbnail": mid_pil_image,
                    }
                )
                seg_counter += 1

        cap.release()
        return segments

    def _evaluate_with_gemini_vision(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Envia os thumbnails dos melhores candidatos ao Gemini Vision em lote para classificação estética e narrativa."""
        prompt_lines = [
            "Você é um Editor de Vídeo Sênior selecionando os melhores takes brutos (highlights).",
            "Avalie cada um dos segmentos abaixo observando o frame correspondente na mesma ordem:",
        ]

        contents: List[Any] = []
        for idx, cand in enumerate(candidates):
            prompt_lines.append(
                f"- Imagem {idx + 1}: segment_id='{cand['segment_id']}' | arquivo='{cand['file_name']}' | "
                f"intervalo={cand['start_sec']}s-{cand['end_sec']}s | nitidez={cand['sharpness']} | movimento={cand['motion']}"
            )

        prompt_lines.append(
            "\nPara cada `segment_id`, atribua `visual_interest_score` (0 a 10), `energy_level` (low, medium, high), "
            "`recommended_role` (hook, buildup, climax, b-roll, outro), `caption_suggestion` (frase curta de 2 a 4 palavras em português) e `description`."
        )

        contents.append("\n".join(prompt_lines))
        for cand in candidates:
            if cand.get("_thumbnail") is not None:
                contents.append(cand["_thumbnail"])

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchSegmentEvaluation,
                    temperature=0.2,
                ),
            )
            parsed = json.loads(response.text)
            eval_map = {ev["segment_id"]: ev for ev in parsed.get("evaluations", [])}

            for cand in candidates:
                ev = eval_map.get(cand["segment_id"])
                if ev:
                    cand["visual_interest_score"] = float(ev.get("visual_interest_score", cand["technical_score"]))
                    cand["energy_level"] = ev.get("energy_level", cand["energy_level"])
                    cand["recommended_role"] = ev.get("recommended_role", "buildup")
                    cand["caption_suggestion"] = ev.get("caption_suggestion", "")
                    cand["description"] = ev.get("description", cand["description"])
        except Exception as exc:
            self.log(f"Aviso na classificação Gemini Vision (mantendo pontuação técnica do OpenCV): {exc}")

        # Remove objetos PIL temporários para serialização limpa
        for cand in candidates:
            cand.pop("_thumbnail", None)

        return candidates
