from __future__ import annotations

import csv
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = font(30, True)
FONT_LABEL = font(22, True)
FONT_SMALL = font(18)


def add_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=(30, 30, 30)):
    draw.text(xy, text, font=fnt, fill=fill)


def open_png(path: Path, size: tuple[int, int] = (210, 210)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS)


def draw_cell(
    canvas: Image.Image,
    img: Image.Image,
    x: int,
    y: int,
    header: str,
    subheader: str = "",
):
    draw = ImageDraw.Draw(canvas)
    canvas.paste(img, (x, y + 42))
    draw.rectangle((x, y + 42, x + img.width - 1, y + 42 + img.height - 1), outline=(150, 150, 150), width=1)
    add_text(draw, (x, y), header, FONT_LABEL)
    if subheader:
        add_text(draw, (x, y + 24), subheader, FONT_SMALL, fill=(75, 75, 75))


def read_rows() -> list[dict[str, str]]:
    with (ROOT / "temporal_perturbation_results.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_row(rows: list[dict[str, str]], sample_index: int, perturbation: str, severity: str) -> dict[str, str]:
    for row in rows:
        if (
            int(row["sample_index"]) == sample_index
            and row["perturbation"] == perturbation
            and row["severity"] == severity
        ):
            return row
    raise ValueError(f"No row for sample={sample_index}, perturbation={perturbation}, severity={severity}")


def rel_path(value: str) -> Path:
    return ROOT.parents[2] / value.replace("\\", "/")


def extract_frames(video_path: Path, n: int = 6, size: tuple[int, int] = (180, 180)) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [round(i * (total - 1) / (n - 1)) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS))
    cap.release()
    return frames


def make_temporal_sequence(rows: list[dict[str, str]]) -> Path:
    row = get_row(rows, 683, "reverse", "0.25")
    clean_frames = extract_frames(rel_path(row["clean_video_path"]))
    reversed_frames = extract_frames(rel_path(row["perturbed_video_path"]))

    margin, label_w, gap = 34, 270, 14
    frame_w, frame_h = clean_frames[0].size
    width = margin * 2 + label_w + len(clean_frames) * frame_w + (len(clean_frames) - 1) * gap
    height = 80 + 2 * frame_h + 84
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    row_y = [36, 36 + frame_h + 62]
    labels = ["Original", "Reversed"]
    for frames, y, label in zip([clean_frames, reversed_frames], row_y, labels):
        add_text(draw, (margin, y + frame_h // 2 - 14), label, FONT_LABEL)
        for i, frame in enumerate(frames):
            x = margin + label_w + i * (frame_w + gap)
            canvas.paste(frame, (x, y))
            draw.rectangle((x, y, x + frame_w - 1, y + frame_h - 1), outline=(135, 135, 135), width=1)
            add_text(draw, (x + 58, y + frame_h + 8), f"t{i + 1}", FONT_SMALL, fill=(80, 80, 80))

    add_text(
        draw,
        (margin, height - 42),
        f"Case 0X53491FA3CDC950C1, reverse severity 0.25; |Delta EF error| = {abs(float(row['delta_ef_abs_error_pct'])):.2f} pp",
        FONT_SMALL,
        fill=(70, 70, 70),
    )
    out = OUT / "figure_1_original_vs_reversed_temporal_sequence.png"
    canvas.save(out, dpi=(300, 300))
    return out


def make_segmentation_examples(rows: list[dict[str, str]]) -> Path:
    cases = [
        (683, "reverse", "0.25", "Case A"),
        (680, "attention_guided_mask", "0.25", "Case B"),
        (731, "attention_guided_mask", "0.25", "Case C"),
    ]
    tile = (184, 184)
    margin, gap_x, gap_y = 34, 18, 70
    row_label_w = 92
    cols = 4
    width = margin * 2 + row_label_w + cols * tile[0] + (cols - 1) * gap_x
    height = 58 + len(cases) * (tile[1] + gap_y) + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["GT ED", "Pred ED", "GT ES", "Pred ES"]
    fields = ["clean_gt_ed_frame_path", "clean_clean_pred_ed_frame_path", "clean_gt_es_frame_path", "clean_clean_pred_es_frame_path"]

    for r, (sample, perturbation, severity, label) in enumerate(cases):
        row = get_row(rows, sample, perturbation, severity)
        y = 40 + r * (tile[1] + gap_y)
        add_text(draw, (margin, y + 72), label, FONT_LABEL)
        for c, (field, header) in enumerate(zip(fields, headers)):
            x = margin + row_label_w + c * (tile[0] + gap_x)
            img = open_png(rel_path(row[field]), tile)
            draw_cell(canvas, img, x, y, header)
        add_text(draw, (margin + row_label_w, y + tile[1] + 48), row["file_name"].replace(".avi", ""), FONT_SMALL, fill=(70, 70, 70))

    out = OUT / "figure_2_segmentation_overlay_examples.png"
    canvas.save(out, dpi=(300, 300))
    return out


def make_attention_visualization(rows: list[dict[str, str]]) -> Path:
    row = get_row(rows, 680, "attention_guided_mask", "0.25")
    tile = (215, 215)
    margin, gap_x, gap_y = 34, 22, 76
    row_label_w = 52
    cols = 4
    width = margin * 2 + row_label_w + cols * tile[0] + (cols - 1) * gap_x
    height = 72 + 2 * (tile[1] + gap_y)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    columns = [
        ("Clean GT", ("clean_gt_ed_frame_path", "clean_gt_es_frame_path")),
        ("Clean Pred", ("clean_clean_pred_ed_frame_path", "clean_clean_pred_es_frame_path")),
        ("Masked Input", ("perturbed_gt_ed_frame_path", "perturbed_gt_es_frame_path")),
        ("Masked Pred", ("perturbed_perturbed_pred_ed_frame_path", "perturbed_perturbed_pred_es_frame_path")),
    ]
    phases = [("ED", 56), ("ES", 56 + tile[1] + gap_y)]
    for phase, y in phases:
        add_text(draw, (margin, y + 90), phase, FONT_LABEL)
    for c, (header, fields) in enumerate(columns):
        x = margin + row_label_w + c * (tile[0] + gap_x)
        for (_, y), field in zip(phases, fields):
            img = open_png(rel_path(row[field]), tile)
            draw_cell(canvas, img, x, y, header if y == phases[0][1] else "")

    out = OUT / "figure_3_attention_perturbation_visualization.png"
    canvas.save(out, dpi=(300, 300))
    return out


def main() -> None:
    rows = read_rows()
    outputs = [
        make_temporal_sequence(rows),
        make_segmentation_examples(rows),
        make_attention_visualization(rows),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
