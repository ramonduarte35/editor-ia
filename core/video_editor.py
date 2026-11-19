import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL.Image

# Compatibilidade entre Pillow >= 10.0 e MoviePy 1.0.3 (que referencia Image.ANTIALIAS)
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    VideoFileClip,
    concatenate_videoclips,
    afx,
    vfx,
)


class VideoEditor:
    """
    Módulo 4 (Montagem e Renderização):
    Monta a timeline seguindo o JSON do Blueprint de Edição usando MoviePy e FFmpeg:
    - Sincroniza os cortes com o ritmo identificado (BPM/beats e duração dos planos)
    - Aplica Smart Crop para o aspect ratio escolhido (9:16, 16:9, 1:1)
    - Aplica tratamento de cor (saturação, contraste, brilho, temperatura/LUT)
    - Aplica zoom dinâmico e legendas estilizadas (renderizadas via Pillow sem depender do ImageMagick)
    - Aplica crossfades ou cortes secos e normaliza o áudio final (loudnorm via FFmpeg)
    """

    def __init__(self, log_fn: Optional[Callable[[str], None]] = None):
        self.log = log_fn or (lambda msg: None)

    def render_edited_video(
        self,
        blueprint: Dict[str, Any],
        indexed_segments: List[Dict[str, Any]],
        audio_files: List[Path],
        reference_video_path: Optional[Path],
        output_dir: Path,
        target_resolution: Tuple[int, int],
        quality_cfg: Dict[str, Any],
        temp_dir: Path,
        use_reference_audio_if_no_music: bool = True,
    ) -> Path:
        """Executa o pipeline completo de montagem e exportação do vídeo final."""
        scale = float(quality_cfg.get("scale", 1.0))
        target_w = int(round(target_resolution[0] * scale / 2.0) * 2)
        target_h = int(round(target_resolution[1] * scale / 2.0) * 2)

        cut_slots = blueprint.get("cut_slots") or []
        # Se não houver cut_slots (modo sem vídeo de referência), cria slots sequenciais para todos os segmentos limpos
        if not cut_slots:
            cut_slots = [
                {
                    "slot_index": i,
                    "duration_sec": float(seg["duration_sec"]),
                    "shot_role": seg.get("recommended_role", "buildup"),
                    "energy_level": seg.get("energy_level", "medium"),
                    "transition_in": "hard_cut",
                }
                for i, seg in enumerate(indexed_segments)
            ]
            self.log(f"Modo Fala Contínua: Montando timeline sequencial completa com todos os {len(cut_slots)} trechos limpos.")
        else:
            self.log(f"Pareando {len(cut_slots)} cortes do Blueprint com os melhores takes brutos...")
        timeline_plan = self._match_segments_to_slots(cut_slots, indexed_segments, blueprint)

        color_spec = blueprint.get("color_grading") or {}
        sub_spec = blueprint.get("subtitle_style") or {}
        framing_style = blueprint.get("framing_style", "dynamic_zoom")
        zoom_intensity = float(blueprint.get("dynamic_zoom_intensity", 0.04))
        crossfade_sec = float(blueprint.get("crossfade_duration", 0.0))
        recommended_transition = blueprint.get("recommended_transition", "hard_cut")

        opened_source_clips: List[VideoFileClip] = []
        processed_clips: List[ Any ] = []

        # Otimização de Performance: Se o color grading for neutro (1.0), pula processamento pesado pixel a pixel
        has_color_grading = (
            abs(float(color_spec.get("saturation", 1.0)) - 1.0) > 0.05
            or abs(float(color_spec.get("contrast", 1.0)) - 1.0) > 0.05
            or abs(float(color_spec.get("brightness", 0.0))) > 0.04
            or abs(float(color_spec.get("warmth", 0.0))) > 0.04
            or str(color_spec.get("preset_name", "")).lower() == "bw_punch"
        )
        has_subtitles = bool(sub_spec.get("enabled", False))
        has_zoom = framing_style == "dynamic_zoom" and zoom_intensity > 0.01

        # Cache de clips abertos por arquivo de origem para evitar reabrir o mesmo vídeo dezenas de vezes
        clip_cache: Dict[str, VideoFileClip] = {}

        try:
            for idx, plan_item in enumerate(timeline_plan):
                seg = plan_item["segment"]
                slot = plan_item["slot"]
                clip_dur = float(plan_item["duration_sec"])

                src_path = seg["source_path"]
                start_t = float(seg["start_sec"])
                end_t = min(float(seg["end_sec"]), start_t + clip_dur)
                if end_t - start_t < 0.45:
                    end_t = min(float(seg["end_sec"]), start_t + 0.6)

                self.log(
                    f"Montando Take {idx + 1}/{len(timeline_plan)}: {seg['file_name']} "
                    f"({start_t:.2f}s -> {end_t:.2f}s)"
                )

                if src_path not in clip_cache:
                    src_clip = VideoFileClip(src_path)
                    clip_cache[src_path] = src_clip
                    opened_source_clips.append(src_clip)
                else:
                    src_clip = clip_cache[src_path]

                safe_end = min(src_clip.duration - 0.02, end_t)
                safe_start = max(0.0, min(start_t, safe_end - 0.4))
                sub = src_clip.subclip(safe_start, safe_end)

                # 1. Smart Crop e Redimensionamento para a proporção escolhida
                sub = self._apply_smart_crop(sub, target_w, target_h, focus_x=float(seg.get("focus_x", 0.5)))

                # 2. Efeito de Enquadramento Dinâmico, Cor e Legenda (apenas se realmente ativos)
                caption_text = plan_item.get("caption_text", "")
                if has_color_grading or (has_subtitles and caption_text) or has_zoom:
                    sub = self._apply_visual_styling(
                        clip=sub,
                        color_spec=color_spec,
                        framing_style=framing_style,
                        zoom_intensity=zoom_intensity,
                        caption_text=caption_text,
                        sub_spec=sub_spec,
                        target_w=target_w,
                        target_h=target_h,
                    )

                # 3. Transição de entrada (crossfade ou flash branco)
                transition_type = slot.get("transition_in", recommended_transition)
                if idx > 0 and transition_type == "crossfade" and crossfade_sec > 0.05:
                    fade_d = min(crossfade_sec, sub.duration * 0.35)
                    sub = sub.crossfadein(fade_d)
                elif idx > 0 and transition_type == "flash":
                    sub = sub.fx(vfx.fadein, 0.12, initial_color=[255, 255, 255])

                processed_clips.append(sub)

            if not processed_clips:
                raise RuntimeError("Nenhum clipe pôde ser montado na timeline.")

            self.log("Concatenando timeline principal no MoviePy...")
            if recommended_transition == "crossfade" and crossfade_sec > 0.05 and len(processed_clips) > 1:
                pad = -min(crossfade_sec, 0.25)
                final_video = concatenate_videoclips(processed_clips, method="compose", padding=pad)
            else:
                final_video = concatenate_videoclips(processed_clips, method="compose")

            # 4. Tratamento e Sincronização de Áudio
            external_audio_clip: Optional[AudioFileClip] = None
            if audio_files:
                self.log(f"Aplicando trilha de áudio da pasta de entrada: {audio_files[0].name}")
                external_audio_clip = AudioFileClip(str(audio_files[0]))
            elif use_reference_audio_if_no_music and reference_video_path and reference_video_path.exists():
                try:
                    ref_audio_clip = AudioFileClip(str(reference_video_path))
                    if ref_audio_clip.duration and ref_audio_clip.duration > 1.0:
                        self.log("Usando trilha rítmica do vídeo de referência como guia musical...")
                        external_audio_clip = ref_audio_clip
                except Exception:
                    external_audio_clip = None

            if external_audio_clip is not None:
                if external_audio_clip.duration < final_video.duration:
                    external_audio_clip = afx.audio_loop(external_audio_clip, duration=final_video.duration)
                else:
                    external_audio_clip = external_audio_clip.subclip(0, final_video.duration)
                external_audio_clip = external_audio_clip.fx(afx.audio_fadeout, min(1.2, final_video.duration * 0.15))
                final_video = final_video.set_audio(external_audio_clip)

            # 5. Exportação para arquivo temporário e normalização de áudio via FFmpeg (EBU R128 loudnorm)
            raw_render_path = temp_dir / "rendered_timeline_raw.mp4"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_output_path = output_dir / f"video_editado_{timestamp_str}.mp4"

            self.log(f"Renderizando vídeo ({target_w}x{target_h}) com MoviePy + FFmpeg...")
            # Limita a 4 threads de renderização do FFmpeg para não afogar a CPU nem travar o SO
            final_video.write_videofile(
                str(raw_render_path),
                fps=30,
                codec="libx264",
                audio_codec="aac",
                preset=quality_cfg.get("preset", "ultrafast"),
                bitrate=quality_cfg.get("bitrate", "4500k"),
                ffmpeg_params=["-crf", str(quality_cfg.get("crf", "23")), "-pix_fmt", "yuv420p"],
                threads=4,
                logger=None,
            )

            if external_audio_clip is not None:
                external_audio_clip.close()
            final_video.close()

            self.log("Normalizando áudio final (EBU R128 Loudness Normalization) via FFmpeg...")
            self._normalize_audio_ffmpeg(raw_render_path, final_output_path)

            self.log(f"Vídeo final exportado com sucesso em: {final_output_path}")
            return final_output_path

        finally:
            for clip in opened_source_clips:
                try:
                    clip.close()
                except Exception:
                    pass

    def _match_segments_to_slots(
        self,
        cut_slots: List[Dict[str, Any]],
        indexed_segments: List[Dict[str, Any]],
        blueprint: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Associa cada slot do Blueprint ao segmento bruto mais compatível em energia,
        papel narrativo e duração, ajustando a duração aos beats (BPM) da referência.
        """
        used_ids = set()
        plan: List[Dict[str, Any]] = []
        bpm = float(blueprint.get("bpm") or 120.0)
        beat_interval = 60.0 / max(40.0, min(220.0, bpm))

        sub_spec = blueprint.get("subtitle_style") or {}
        overlay_phrases = list(sub_spec.get("overlay_phrases") or [])

        for idx, slot in enumerate(cut_slots):
            desired_dur = float(slot.get("duration_sec", blueprint.get("avg_shot_duration", 2.0)))
            # Quantiza duração para múltiplo da batida musical (beat sync)
            beats_count = max(1, round(desired_dur / beat_interval))
            synced_dur = float(np.clip(beats_count * beat_interval, 0.65, 5.5))

            desired_energy = slot.get("energy_level", "medium")
            desired_role = slot.get("shot_role", "buildup")

            best_seg = None
            best_match_score = -1.0

            for seg in indexed_segments:
                seg_id = seg["segment_id"]
                reuse_penalty = 4.5 if seg_id in used_ids else 0.0
                dur_diff_penalty = 0.8 if seg["duration_sec"] < synced_dur * 0.75 else 0.0
                energy_bonus = 1.5 if seg.get("energy_level") == desired_energy else 0.0
                role_bonus = 1.2 if seg.get("recommended_role") == desired_role else 0.0

                score = float(seg.get("final_score", 7.0)) + energy_bonus + role_bonus - reuse_penalty - dur_diff_penalty
                if score > best_match_score:
                    best_match_score = score
                    best_seg = seg

            if best_seg is None:
                best_seg = indexed_segments[idx % len(indexed_segments)]

            used_ids.add(best_seg["segment_id"])

            caption_text = ""
            if sub_spec.get("enabled", False):
                if idx < len(overlay_phrases) and overlay_phrases[idx]:
                    caption_text = overlay_phrases[idx]
                elif best_seg.get("caption_suggestion"):
                    caption_text = best_seg["caption_suggestion"]

            plan.append(
                {
                    "slot": slot,
                    "segment": best_seg,
                    "duration_sec": min(synced_dur, float(best_seg["duration_sec"])),
                    "caption_text": caption_text,
                }
            )

        return plan

    def _apply_smart_crop(self, clip: VideoFileClip, target_w: int, target_h: int, focus_x: float = 0.5) -> VideoFileClip:
        """Redimensiona e aplica recorte inteligente (Smart Crop) preservando o centro de ação."""
        src_w, src_h = clip.size
        target_ratio = target_w / float(target_h)
        src_ratio = src_w / float(src_h)

        if abs(src_ratio - target_ratio) < 0.02:
            return clip.resize(newsize=(target_w, target_h))

        if src_ratio > target_ratio:
            # Vídeo original é mais largo que o alvo (ex: 16:9 -> 9:16): ajusta altura e recorta largura no focus_x
            resized = clip.resize(height=target_h)
            new_w, _ = resized.size
            center_x = int(np.clip(focus_x * new_w, target_w / 2.0, new_w - target_w / 2.0))
            x1 = max(0, int(center_x - target_w / 2.0))
            return resized.crop(x1=x1, y1=0, width=target_w, height=target_h)
        else:
            # Vídeo original é mais alto que o alvo: ajusta largura e recorta altura centralizada
            resized = clip.resize(width=target_w)
            _, new_h = resized.size
            y1 = max(0, int((new_h - target_h) / 2.0))
            return resized.crop(x1=0, y1=y1, width=target_w, height=target_h)

    def _apply_visual_styling(
        self,
        clip: VideoFileClip,
        color_spec: Dict[str, Any],
        framing_style: str,
        zoom_intensity: float,
        caption_text: str,
        sub_spec: Dict[str, Any],
        target_w: int,
        target_h: int,
    ) -> VideoFileClip:
        """
        Aplica Color Grading (saturação, contraste, brilho, balanço quente/frio),
        Zoom Dinâmico progressivo e overlay de legenda estilizada frame a frame.
        """
        sat = float(np.clip(color_spec.get("saturation", 1.1), 0.5, 1.6))
        con = float(np.clip(color_spec.get("contrast", 1.06), 0.7, 1.5))
        bri = float(np.clip(color_spec.get("brightness", 0.01), -0.25, 0.25))
        warmth = float(np.clip(color_spec.get("warmth", 0.0), -0.25, 0.25))
        preset_name = str(color_spec.get("preset_name", "vibrant")).lower()

        dur = max(0.5, float(clip.duration or 2.0))
        do_zoom = framing_style == "dynamic_zoom" and zoom_intensity > 0.005

        # Pré-renderiza camada RGBA da legenda com Pillow (se ativa) para máxima velocidade
        subtitle_rgba: Optional[np.ndarray] = None
        if caption_text and sub_spec.get("enabled", False):
            subtitle_rgba = self._create_subtitle_overlay_rgba(
                text=caption_text,
                width=target_w,
                height=target_h,
                position=sub_spec.get("position", "bottom"),
                primary_hex=sub_spec.get("primary_color_hex", "#FFFFFF"),
                stroke_hex=sub_spec.get("stroke_color_hex", "#000000"),
            )

        def process_frame(get_frame, t):
            frame = get_frame(t)
            img = frame.astype(np.float32)

            # 1. Tratamento de cor (LUT paramétrica)
            if preset_name == "bw_punch":
                luma = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                img[:, :, 0] = luma
                img[:, :, 1] = luma
                img[:, :, 2] = luma
            else:
                if abs(sat - 1.0) > 0.01:
                    luma = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])[:, :, np.newaxis]
                    img = luma + (img - luma) * sat

                if abs(warmth) > 0.01:
                    img[:, :, 0] += warmth * 35.0  # Canal R
                    img[:, :, 2] -= warmth * 35.0  # Canal B

            if abs(con - 1.0) > 0.01 or abs(bri) > 0.005:
                img = (img - 128.0) * con + 128.0 + (bri * 255.0)

            out = np.clip(img, 0, 255).astype(np.uint8)

            # 2. Zoom dinâmico progressivo
            if do_zoom:
                progress = float(np.clip(t / dur, 0.0, 1.0))
                z_factor = 1.0 + zoom_intensity * progress
                if z_factor > 1.003:
                    h, w = out.shape[:2]
                    crop_w = int(w / z_factor)
                    crop_h = int(h / z_factor)
                    x1 = (w - crop_w) // 2
                    y1 = (h - crop_h) // 2
                    cropped = out[y1 : y1 + crop_h, x1 : x1 + crop_w]
                    out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            # 3. Composição alfa da legenda estilizada
            if subtitle_rgba is not None:
                alpha = (subtitle_rgba[:, :, 3:4].astype(np.float32)) / 255.0
                fg_rgb = subtitle_rgba[:, :, :3].astype(np.float32)
                out = np.clip(out.astype(np.float32) * (1.0 - alpha) + fg_rgb * alpha, 0, 255).astype(np.uint8)

            return out

        return clip.fl(process_frame)

    def _create_subtitle_overlay_rgba(
        self,
        text: str,
        width: int,
        height: int,
        position: str = "bottom",
        primary_hex: str = "#FFFFFF",
        stroke_hex: str = "#000000",
    ) -> np.ndarray:
        """Gera uma imagem transparente RGBA com legenda estilo Reels/Shorts usando Pillow."""
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        font_size = max(28, int(min(width, height) * 0.055))
        font = None
        for font_candidate in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]:
            if Path(font_candidate).exists():
                try:
                    font = ImageFont.truetype(font_candidate, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        clean_text = text.strip().upper()
        bbox = draw.textbbox((0, 0), clean_text, font=font, stroke_width=4)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = max(20, (width - text_w) // 2)
        if position == "center":
            y = (height - text_h) // 2
        elif position == "top":
            y = int(height * 0.14)
        else:
            y = int(height * 0.78)

        # Caixa semi-transparente suave para legibilidade
        pad_x, pad_y = 22, 12
        draw.rounded_rectangle(
            [x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y],
            radius=14,
            fill=(0, 0, 0, 135),
        )

        draw.text(
            (x, y),
            clean_text,
            font=font,
            fill=primary_hex if primary_hex.startswith("#") else "#FFFFFF",
            stroke_width=4,
            stroke_fill=stroke_hex if stroke_hex.startswith("#") else "#000000",
        )
        return np.array(canvas)

    def _normalize_audio_ffmpeg(self, input_mp4: Path, output_mp4: Path) -> None:
        """
        Aplica equalização de voz e normalização EBU R128 via FFmpeg:
        - highpass=f=80: corta frequências subsônicas/estrondos e vibrações de mesa
        - equalizer=f=3000:t=q:w=1.5:g=2.0: leve realce de clareza e inteligibilidade na voz
        - loudnorm=I=-16:TP=-1.5:LRA=11: normalização de loudness padrão para plataformas digitais
        """
        try:
            audio_filter = "highpass=f=80,equalizer=f=3000:t=q:w=1.5:g=2.0,loudnorm=I=-16:TP=-1.5:LRA=11"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_mp4),
                "-c:v",
                "copy",
                "-af",
                audio_filter,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(output_mp4),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode != 0 or not output_mp4.exists():
                # Se falhar ou não tiver áudio, apenas copia o arquivo original renderizado
                input_mp4.replace(output_mp4)
        except Exception:
            if input_mp4.exists():
                input_mp4.replace(output_mp4)
