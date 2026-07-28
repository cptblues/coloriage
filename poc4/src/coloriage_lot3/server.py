"""Petit serveur local pour brancher l'interface web au moteur Python."""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .export import build_stats, export_result
from .pipeline import PipelineConfig, _load_and_normalize, run_pipeline
from .subject import generate_ai_mask, mask_overlay


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 40 * 1024 * 1024
DEFAULT_MAX_SIDE = 1200
DEFAULT_MAX_SIDE_LIMIT = 2400
DEFAULT_MAX_DETAIL_ZONES = 64
DEFAULT_MAX_DETAIL_POINTS = 12000
AUTO_MAX_SIDE_VALUE = "auto"


class RequestError(Exception):
    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def _clamp_int(value: object, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    return _clamp_int(os.getenv(name), minimum, maximum, default)


def _auto_max_side_for_image(
    source_path: Path,
    default: int,
    maximum: int,
) -> int:
    try:
        with Image.open(source_path) as image:
            width, height = image.size
    except (OSError, ValueError):
        return max(256, min(maximum, default))

    long_side = max(width, height)
    if long_side <= 900:
        target = min(default, 900)
    elif long_side >= 2400:
        target = max(default, 1600)
    else:
        target = default
    return max(256, min(maximum, target))


def _resolve_max_side(
    value: object,
    source_path: Path,
    default: int,
    maximum: int,
) -> int:
    if value is None:
        return _auto_max_side_for_image(source_path, default, maximum)
    if isinstance(value, str) and value.strip().lower() == AUTO_MAX_SIDE_VALUE:
        return _auto_max_side_for_image(source_path, default, maximum)
    return _clamp_int(value, 256, maximum, default)


def _data_url(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_to_data_url(image: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as handle:
        image.save(handle.name, "PNG")
        return _data_url("image/png", Path(handle.name).read_bytes())


def _file_to_data_url(path: Path, mime_type: str = "image/png") -> str:
    return _data_url(mime_type, path.read_bytes())


def _decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise RequestError("Image encodée invalide.")
    header, data = data_url.split(",", 1)
    if ";base64" not in header:
        raise RequestError("Seules les images base64 sont acceptées.")
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        raise RequestError("Image base64 invalide.") from exc


def _write_image_payload(
    payload: dict[str, Any],
    directory: Path,
    max_image_bytes: int,
) -> Path:
    image = payload.get("image")
    if not isinstance(image, dict):
        raise RequestError("Le champ image est obligatoire.")
    data_url = image.get("dataUrl")
    if not isinstance(data_url, str):
        raise RequestError("Le champ image.dataUrl est obligatoire.")
    name = str(image.get("name") or "source.png")
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".png"
    data = _decode_data_url(data_url)
    if len(data) > max_image_bytes:
        raise RequestError(
            "Image trop volumineuse.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    path = directory / f"source{suffix}"
    path.write_bytes(data)
    return path


def _write_optional_data_url(
    data_url: object,
    directory: Path,
    filename: str,
    max_bytes: int,
) -> Path | None:
    if not isinstance(data_url, str) or not data_url:
        return None
    data = _decode_data_url(data_url)
    if len(data) > max_bytes:
        raise RequestError(
            "Fichier joint trop volumineux.",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    path = directory / filename
    path.write_bytes(data)
    return path


def _complexity_config(value: object) -> dict[str, float | int]:
    if value == "simple":
        return {
            "superpixels": 620,
            "min_region_area_mm2": 14.0,
            "subject_min_region_area_mm2": 8.0,
            "background_min_region_area_mm2": 36.0,
            "background_superpixel_ratio": 0.35,
            "background_smoothing_radius": 4,
            "detail_min_region_area_mm2": 3.5,
            "contour_smoothing_iterations": 2,
        }
    if value == "detaille":
        return {
            "superpixels": 1350,
            "min_region_area_mm2": 6.0,
            "subject_min_region_area_mm2": 4.0,
            "background_min_region_area_mm2": 20.0,
            "background_superpixel_ratio": 0.55,
            "background_smoothing_radius": 2,
            "detail_min_region_area_mm2": 2.0,
            "contour_smoothing_iterations": 1,
        }
    return {
        "superpixels": 900,
        "min_region_area_mm2": 9.0,
        "subject_min_region_area_mm2": 6.0,
        "background_min_region_area_mm2": 28.0,
        "background_superpixel_ratio": 0.45,
        "background_smoothing_radius": 3,
        "detail_min_region_area_mm2": 3.0,
        "contour_smoothing_iterations": 1,
    }


def _format_config(value: object) -> str:
    return "a3" if str(value).lower() == "a3" else "a4"


def _orientation_config(value: object) -> str:
    return "landscape" if str(value).lower() in {"paysage", "landscape"} else "portrait"


def _palette_layout_config(value: object) -> str:
    return "inline" if str(value).lower() == "inline" else "separate"


def _smooth_points(
    points: list[tuple[float, float]],
    *,
    closed: bool,
    iterations: int = 2,
) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    smoothed = points
    for _ in range(iterations):
        next_points: list[tuple[float, float]] = []
        if not closed:
            next_points.append(smoothed[0])
        count = len(smoothed)
        pair_count = count if closed else count - 1
        for index in range(pair_count):
            start = smoothed[index]
            end = smoothed[(index + 1) % count]
            next_points.append(
                (
                    0.75 * start[0] + 0.25 * end[0],
                    0.75 * start[1] + 0.25 * end[1],
                )
            )
            next_points.append(
                (
                    0.25 * start[0] + 0.75 * end[0],
                    0.25 * start[1] + 0.75 * end[1],
                )
            )
        if not closed:
            next_points.append(smoothed[-1])
        smoothed = next_points
    return smoothed


def _draw_detail_mask(
    source_path: Path,
    zones: object,
    directory: Path,
    max_side: int,
    max_zones: int = DEFAULT_MAX_DETAIL_ZONES,
    max_points: int = DEFAULT_MAX_DETAIL_POINTS,
) -> Path | None:
    if not isinstance(zones, list) or not zones:
        return None
    if len(zones) > max_zones:
        raise RequestError("Trop de zones détaillées.")
    normalized_rgb, _metadata = _load_and_normalize(source_path, max_side)
    height, width = normalized_rgb.shape[:2]
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    drew_zone = False

    total_points = 0
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        raw_points = zone.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            continue
        total_points += len(raw_points)
        if total_points > max_points:
            raise RequestError("Le tracé contient trop de points.")
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            try:
                x = max(0.0, min(1.0, float(point["x"]))) * width
                y = max(0.0, min(1.0, float(point["y"]))) * height
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x, y))
        if not points:
            continue
        try:
            radius = max(0.006, min(0.12, float(zone.get("radius", 0.027))))
        except (TypeError, ValueError):
            radius = 0.027
        line_width = max(8, round(radius * min(width, height) * 2.4))
        mode = zone.get("mode")
        if mode == "outline" and len(points) >= 3:
            smoothed = _smooth_points(points, closed=True)
            draw.polygon(smoothed, fill=255)
            draw.line(
                [*smoothed, smoothed[0]],
                fill=255,
                width=max(3, line_width // 4),
                joint="curve",
            )
        elif len(points) == 1:
            x, y = points[0]
            half = line_width / 2
            draw.ellipse((x - half, y - half, x + half, y + half), fill=255)
        else:
            draw.line(
                _smooth_points(points, closed=False),
                fill=255,
                width=line_width,
                joint="curve",
            )
        drew_zone = True

    if not drew_zone:
        return None
    path = directory / "detail-mask.png"
    image.save(path)
    return path


def _palette_payload(result: Any) -> list[dict[str, object]]:
    return [
        {
            "number": index + 1,
            "hex": "#{:02X}{:02X}{:02X}".format(*(int(value) for value in rgb)),
            "rgb": [int(value) for value in rgb],
        }
        for index, rgb in enumerate(result.palette_rgb)
    ]


def _mask_payload(
    normalized_rgb: Any,
    source_metadata: dict[str, Any],
    mask: Any | None,
    subject_metadata: dict[str, Any],
) -> dict[str, Any]:
    control_image = (
        Image.fromarray(mask_overlay(normalized_rgb, mask), "RGB")
        if mask is not None
        else Image.fromarray(normalized_rgb, "RGB")
    )
    return {
        "ok": True,
        "source": source_metadata,
        "subject": subject_metadata,
        "subjectMaskImage": (
            _image_to_data_url(Image.fromarray((mask.astype("uint8") * 255), "L"))
            if mask is not None
            else None
        ),
        "maskControlImage": _image_to_data_url(control_image),
    }


def _config_from_payload(
    payload: dict[str, Any],
    subject_mask_path: Path | None,
    detail_mask_path: Path | None,
    max_side: int,
) -> PipelineConfig:
    preset = _complexity_config(payload.get("complexity"))
    return PipelineConfig(
        colors=_clamp_int(payload.get("colors"), 2, 40, 24),
        max_side=max_side,
        sample_pixels=_clamp_int(payload.get("samplePixels"), 10_000, 300_000, 100_000),
        segmentation="slic",
        superpixels=int(preset["superpixels"]),
        smoothing_radius=1,
        min_region_area_mm2=float(preset["min_region_area_mm2"]),
        page_format=_format_config(payload.get("format")),
        orientation=_orientation_config(payload.get("orientation")),
        palette_layout=_palette_layout_config(payload.get("paletteLayout")),
        subject_mode="manual" if subject_mask_path else "none",
        subject_mask_path=str(subject_mask_path) if subject_mask_path else None,
        ai_model=str(payload.get("aiModel") or "birefnet-general"),
        subject_color_ratio=0.68,
        subject_min_region_area_mm2=float(preset["subject_min_region_area_mm2"]),
        background_min_region_area_mm2=float(
            preset["background_min_region_area_mm2"]
        ),
        background_superpixel_ratio=float(preset["background_superpixel_ratio"]),
        background_smoothing_radius=int(preset["background_smoothing_radius"]),
        detail_mask_path=str(detail_mask_path) if detail_mask_path else None,
        detail_min_region_area_mm2=float(preset["detail_min_region_area_mm2"]),
        auto_tune=True,
        contour_smoothing_iterations=int(preset["contour_smoothing_iterations"]),
        title=str(payload.get("title") or "Mon coloriage mystère"),
    )


def _result_payload(
    result: Any,
    paths: dict[str, Path],
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "stats": stats,
        "actualColors": int(len(result.palette_rgb)),
        "regionsAfter": len(result.regions_after),
        "palette": _palette_payload(result),
        "coloringPreviewImage": _file_to_data_url(paths["coloring_preview"]),
        "modelPreviewImage": _file_to_data_url(paths["model_print_preview"]),
        "palettePageImage": (
            _file_to_data_url(paths["palette_page"])
            if "palette_page" in paths
            else None
        ),
        "pdfDocument": _file_to_data_url(
            paths["pdf_document"],
            mime_type="application/pdf",
        ),
        "maskControlImage": (
            _file_to_data_url(paths["mask_control"])
            if "mask_control" in paths
            else None
        ),
        "detailMaskImage": (
            _file_to_data_url(paths["detail_mask"])
            if "detail_mask" in paths
            else None
        ),
    }


class EngineHandler(BaseHTTPRequestHandler):
    server_version = "ColoriageEngine/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[engine] {self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("content-length") or "0")
        if content_length <= 0:
            raise RequestError("Corps JSON manquant.")
        max_request_bytes = int(
            getattr(self.server, "max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)
        )
        if content_length > max_request_bytes:
            raise RequestError(
                "Requête trop volumineuse.",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RequestError("JSON invalide.") from exc

    def _max_image_bytes(self) -> int:
        return int(getattr(self.server, "max_image_bytes", DEFAULT_MAX_IMAGE_BYTES))

    def _request_max_side(self, payload: dict[str, Any], source_path: Path) -> int:
        max_side_limit = int(
            getattr(self.server, "max_side_limit", DEFAULT_MAX_SIDE_LIMIT)
        )
        default_max_side = int(
            getattr(self.server, "default_max_side", DEFAULT_MAX_SIDE)
        )
        return _resolve_max_side(
            payload.get("maxSide"),
            source_path,
            default_max_side,
            max_side_limit,
        )

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self._send_json({"ok": False, "error": "Route inconnue."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/mask":
                self._handle_mask()
            elif self.path == "/generate":
                self._handle_generate()
            else:
                self._send_json(
                    {"ok": False, "error": "Route inconnue."},
                    HTTPStatus.NOT_FOUND,
                )
        except RequestError as exc:
            self._send_json({"ok": False, "error": str(exc)}, exc.status)
        except RuntimeError as exc:
            self._send_json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        except Exception as exc:
            self._send_json(
                {"ok": False, "error": f"Erreur moteur : {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_mask(self) -> None:
        payload = self._read_json()
        with tempfile.TemporaryDirectory(prefix="coloriage-mask-") as temp_dir:
            root = Path(temp_dir)
            source_path = _write_image_payload(
                payload,
                root,
                self._max_image_bytes(),
            )
            max_side = self._request_max_side(payload, source_path)
            normalized_rgb, metadata = _load_and_normalize(source_path, max_side)
            try:
                mask, subject_metadata = generate_ai_mask(
                    normalized_rgb,
                    str(payload.get("aiModel") or "birefnet-general"),
                )
            except ValueError as exc:
                self._send_json(
                    _mask_payload(
                        normalized_rgb,
                        metadata,
                        None,
                        {
                            "mode": "none",
                            "fallback": "global",
                            "reason": "no_reliable_subject",
                            "message": (
                                "Aucun sujet isolable fiable détecté. "
                                "Le coloriage sera généré globalement."
                            ),
                            "detail": str(exc),
                        },
                    )
                )
                return
            self._send_json(
                _mask_payload(normalized_rgb, metadata, mask, subject_metadata)
            )

    def _handle_generate(self) -> None:
        payload = self._read_json()
        with tempfile.TemporaryDirectory(prefix="coloriage-generate-") as temp_dir:
            root = Path(temp_dir)
            source_path = _write_image_payload(
                payload,
                root,
                self._max_image_bytes(),
            )
            subject_mask_path = _write_optional_data_url(
                payload.get("subjectMaskImage"),
                root,
                "subject-mask.png",
                self._max_image_bytes(),
            )
            max_side = self._request_max_side(payload, source_path)
            detail_mask_path = _draw_detail_mask(
                source_path,
                payload.get("detailZones"),
                root,
                max_side,
                max_zones=int(
                    getattr(self.server, "max_detail_zones", DEFAULT_MAX_DETAIL_ZONES)
                ),
                max_points=int(
                    getattr(self.server, "max_detail_points", DEFAULT_MAX_DETAIL_POINTS)
                ),
            )
            config = _config_from_payload(
                payload,
                subject_mask_path,
                detail_mask_path,
                max_side,
            )
            result = run_pipeline(source_path, config)
            output = root / "output"
            paths = export_result(result, output, overwrite=True)
            stats = build_stats(result)
            self._send_json(_result_payload(result, paths, stats))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serveur local du moteur Nuance.")
    parser.add_argument("--host", default=os.getenv("COLORIAGE_ENGINE_HOST", DEFAULT_HOST))
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("COLORIAGE_ENGINE_PORT", DEFAULT_PORT, 1, 65535),
    )
    parser.add_argument(
        "--max-request-mb",
        type=int,
        default=_env_int("COLORIAGE_MAX_REQUEST_MB", 64, 1, 512),
    )
    parser.add_argument(
        "--max-image-mb",
        type=int,
        default=_env_int("COLORIAGE_MAX_IMAGE_MB", 40, 1, 256),
    )
    parser.add_argument(
        "--default-max-side",
        type=int,
        default=_env_int("COLORIAGE_DEFAULT_MAX_SIDE", DEFAULT_MAX_SIDE, 256, 2400),
    )
    parser.add_argument(
        "--max-side-limit",
        type=int,
        default=_env_int("COLORIAGE_MAX_SIDE_LIMIT", DEFAULT_MAX_SIDE_LIMIT, 256, 4000),
    )
    parser.add_argument(
        "--max-detail-zones",
        type=int,
        default=_env_int(
            "COLORIAGE_MAX_DETAIL_ZONES",
            DEFAULT_MAX_DETAIL_ZONES,
            1,
            512,
        ),
    )
    parser.add_argument(
        "--max-detail-points",
        type=int,
        default=_env_int(
            "COLORIAGE_MAX_DETAIL_POINTS",
            DEFAULT_MAX_DETAIL_POINTS,
            100,
            200000,
        ),
    )
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), EngineHandler)
    server.max_request_bytes = args.max_request_mb * 1024 * 1024
    server.max_image_bytes = args.max_image_mb * 1024 * 1024
    server.default_max_side = args.default_max_side
    server.max_side_limit = args.max_side_limit
    server.max_detail_zones = args.max_detail_zones
    server.max_detail_points = args.max_detail_points
    print(f"Nuance engine listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
