import base64
import io
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

tab1, tab2, tab3 = st.tabs(["Перевод текста на изображении", "Перевод текста", "История переводов"])

with tab1: 
    GEMINI_MODEL = "gemini-3.8-flash"
    LANGUAGES = {
        "Русский": "Russian",
        "Українська": "Ukrainian",
        "English": "English",
    }
    MAX_AI_IMAGE_BYTES = 900 * 1024
    MAX_AI_IMAGE_SIDE = 20000


    def compress_image_for_ai(image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail(
            (MAX_AI_IMAGE_SIDE, MAX_AI_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )

        while True:
            for quality in (82, 70, 58):
                output = io.BytesIO()
                image.save(output, format="JPEG", quality=quality, optimize=True)
                compressed = output.getvalue()
                if len(compressed) <= MAX_AI_IMAGE_BYTES:
                    return compressed

            if max(image.size) <= 640:
                raise ValueError(
                    "Изображение слишком большое для отправки. "
                    "Загрузите файл меньшего размера."
                )
            image.thumbnail(
                (int(image.width * 0.8), int(image.height * 0.8)),
                Image.Resampling.LANCZOS,
            )


    def extract_translations(image_bytes, target_language):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Добавьте GEMINI_API_KEY в файл .env.")

        compressed_image = compress_image_for_ai(image_bytes)
        encoded_image = base64.b64encode(compressed_image).decode("ascii")
        prompt = (
            "Read every visible text block in this image and translate it into "
            f"{target_language}. Return only JSON in this format: "
            '{"items":[{"bbox":[x1,y1,x2,y2],"translation":"translated text"}]}. '
            "Coordinates must be integers from 0 to 1000, normalized against image "
            "width and height, with bbox as [left, top, right, bottom]. "
            "Include each separate text block once. Preserve numbers, names, and "
            "meaning. If there is no readable text, return {\"items\":[]}."
        )
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={"x-goog-api-key": api_key},
                json={
                    "model": GEMINI_MODEL,
                    "input": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "data": encoded_image,
                            "mime_type": "image/jpeg",
                        },
                    ],
                    "store": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            response_data = response.json()
            content = "".join(
                part["text"]
                for step in response_data["steps"]
                if step.get("type") == "model_output"
                for part in step.get("content", [])
                if part.get("type") == "text" and isinstance(part.get("text"), str)
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("Истекло время ожидания ответа ИИ.") from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 401:
                message = "Ключ Gemini недействителен. Проверьте GEMINI_API_KEY в .env."
            elif status == 404:
                details = ""
                if exc.response is not None:
                    try:
                        error_data = exc.response.json().get("error", {})
                        details = error_data.get("message", "")
                    except (ValueError, AttributeError):
                        details = ""
                message = (
                    "Gemini вернул 404. Проверьте, что GEMINI_API_KEY создан в "
                    "Google AI Studio и что GEMINI_MODEL — доступная модель. "
                    f"Ответ Google: {details[:300]}"
                )
            elif status == 413:
                message = (
                    "Изображение слишком большое для сервиса ИИ. "
                    "Попробуйте файл меньшего размера."
                )
            elif status == 403:
                message = (
                    "Google Gemini запретил запрос. Проверьте, что Gemini API включен "
                    "для проекта и ключ имеет доступ."
                )
            elif status == 429:
                message = "Превышен лимит Gemini API. Проверьте квоту или попробуйте позже."
            else:
                message = f"Сервис ИИ вернул ошибку HTTP {status or 'неизвестен'}."
            raise RuntimeError(message) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Не удалось подключиться к ИИ: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("ИИ вернул ответ в неизвестном формате.") from exc

        if not isinstance(content, str):
            raise ValueError("ИИ вернул ответ не в текстовом формате.")
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("ИИ не вернул корректный JSON с переводом.") from exc

        items = result.get("items") if isinstance(result, dict) else None
        if not isinstance(items, list):
            raise ValueError("В ответе ИИ отсутствует список переведенного текста.")

        valid_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            translation = item.get("translation")
            if (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, int) and 0 <= value <= 1000 for value in bbox)
                and bbox[0] < bbox[2]
                and bbox[1] < bbox[3]
                and isinstance(translation, str)
                and translation.strip()
            ):
                valid_items.append(
                    {"bbox": bbox, "translation": translation.strip()}
                )
        return valid_items


    def load_font(size):
        for font_path in (
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()


    def wrap_text(text, draw, font, max_width):
        lines = []
        for paragraph in text.splitlines() or [text]:
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            line = words[0]
            for word in words[1:]:
                candidate = f"{line} {word}"
                if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    line = candidate
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines


    def replace_text_in_image(image_bytes, translations):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        boxes = []
        for item in translations:
            x1, y1, x2, y2 = item["bbox"]
            left = max(0, round(x1 * width / 1000))
            top = max(0, round(y1 * height / 1000))
            right = min(width, round(x2 * width / 1000))
            bottom = min(height, round(y2 * height / 1000))
            pad = max(2, round(min(width, height) * 0.006))
            left, top = max(0, left - pad), max(0, top - pad)
            right, bottom = min(width, right + pad), min(height, bottom + pad)
            if right <= left or bottom <= top:
                continue
            cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)
            boxes.append((left, top, right, bottom, item["translation"]))

        if not boxes:
            return image

        inpaint_radius = max(3, round(min(width, height) * 0.015))
        cleaned = cv2.inpaint(
            cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR),
            mask,
            inpaint_radius,
            cv2.INPAINT_TELEA,
        )
        translated_image = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(translated_image)

        for left, top, right, bottom, translation in boxes:
            box_width, box_height = right - left, bottom - top
            font = load_font(max(8, min(64, box_height)))
            margin = max(2, round(min(box_width, box_height) * 0.08))
            inner_width = max(1, box_width - margin * 2)
            inner_height = max(1, box_height - margin * 2)

            while font.size > 8:
                lines = wrap_text(translation, draw, font, inner_width)
                line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 2
                if (
                    max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
                    <= inner_width
                    and len(lines) * line_height <= inner_height
                ):
                    break
                font = load_font(font.size - 1)

            lines = wrap_text(translation, draw, font, inner_width)
            line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 2
            background = image_array[top:bottom, left:right].mean(axis=(0, 1))
            text_color = (20, 20, 20) if background.mean() > 140 else (250, 250, 250)
            y = top + margin
            for line in lines:
                if y + line_height > bottom:
                    break
                draw.text((left + margin, y), line, font=font, fill=text_color)
                y += line_height

        return translated_image


    st.set_page_config(page_title="Перевод текста на изображении", page_icon="🌍")
    st.title("Перевод текста на изображении")
    st.write("Загрузите изображение: ИИ переведет текст и заменит его на копии.")

    uploaded_file = st.file_uploader(
        "Изображение",
        type=["png", "jpg", "jpeg", "webp"],
    )
    target_language = st.selectbox("Перевести на", list(LANGUAGES))

    if uploaded_file:
        try:
            original = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
        except (OSError, ValueError) as exc:
            st.error(f"Не удалось открыть изображение: {exc}")
            st.stop()
        normalized_image = io.BytesIO()
        original.save(normalized_image, format="PNG")
        image_bytes = normalized_image.getvalue()
        original_column, translated_column = st.columns(2)
        with original_column:
            st.subheader("Оригинал")
            st.image(original, use_container_width=True)

        if st.button("Перевести изображение", type="primary"):
            try:
                with st.spinner("Распознаю текст и перевожу..."):
                    translations = extract_translations(
                        image_bytes, LANGUAGES[target_language]
                    )
                    if not translations:
                        raise ValueError("На изображении не удалось найти текст.")
                    translated_image = replace_text_in_image(image_bytes, translations)
                    output = io.BytesIO()
                    translated_image.save(output, format="PNG")
                    st.session_state["translated_image"] = output.getvalue()
                    st.session_state["translated_filename"] = (
                        f"translated_{Path(uploaded_file.name).stem}.png"
                    )
                    st.session_state["translation_key"] = (
                        uploaded_file.name,
                        target_language,
                    )
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
            except (OSError, cv2.error) as exc:
                st.error(f"Не удалось обработать изображение: {exc}")

        if st.session_state.get("translation_key") == (
            uploaded_file.name,
            target_language,
        ):
            with translated_column:
                st.subheader(f"Перевод: {target_language}")
                st.image(st.session_state["translated_image"], use_container_width=True)
            st.download_button(
                "Скачать переведенное изображение",
                data=st.session_state["translated_image"],
                file_name=st.session_state["translated_filename"],
                mime="image/png",
            )
with tab2:
    st.set_page_config(page_title="Перевод текста", page_icon="🌍")
    st.title("Перевод текста")
    st.write("Введите текст: ИИ переведет его на выбранный язык.")

    target_language_text = st.selectbox("Перевести на", list(LANGUAGES), key="text_lang")
    input_text = st.text_area("Введите текст для перевода", height=200)

    if st.button("Перевести текст", type="primary"):
        if not input_text.strip():
            st.error("Пожалуйста, введите текст для перевода.")
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("Добавьте GEMINI_API_KEY в файл .env.")
            else:
                prompt = (
                    f"Translate the following text into {LANGUAGES[target_language_text]}:\n\n"
                    f"{input_text}"
                )
                try:
                    response = requests.post(
                        "https://generativelanguage.googleapis.com/v1beta/interactions",
                        headers={"x-goog-api-key": api_key},
                        json={
                            "model": GEMINI_MODEL,
                            "input": [{"type": "text", "text": prompt}],
                            "store": False,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    content = "".join(
                        part["text"]
                        for step in response_data["steps"]
                        if step.get("type") == "model_output"
                        for part in step.get("content", [])
                        if part.get("type") == "text" and isinstance(part.get("text"), str)
                    )
                    st.subheader(f"Перевод на {target_language_text}")
                    st.text_area(
                        "Результат перевода",
                        value=content.strip(),
                        height=200,
                        disabled=True,
                    )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Не удалось подключиться к ИИ: {exc}")
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    st.error(f"ИИ вернул ответ в неизвестном формате: {exc}")
with tab3:
    st.set_page_config(page_title="История переводов", page_icon="📜")
    st.title("История переводов")
    st.write("Здесь отображается история ваших переводов текста и изображений.")

    if "translation_history" not in st.session_state:
        st.session_state["translation_history"] = []

    if st.session_state.get("translation_key"):
        translation_entry = {
            "file_name": st.session_state["translation_key"][0],
            "target_language": st.session_state["translation_key"][1],
            "image": st.session_state.get("translated_image"),
        }
        if translation_entry not in st.session_state["translation_history"]:
            st.session_state["translation_history"].append(translation_entry)

    if st.session_state["translation_history"]:
        for entry in reversed(st.session_state["translation_history"]):
            st.subheader(f"Файл: {entry['file_name']}")
            st.write(f"Переведено на: {entry['target_language']}")
            if entry.get("image"):
                st.image(entry["image"], use_container_width=True)
            else:
                st.write("Нет изображения для отображения.")
    else:
        st.write("История переводов пуста.")