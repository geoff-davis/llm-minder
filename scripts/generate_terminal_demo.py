"""Generate the deterministic README terminal-demo GIF.

Maintainer command:

    uv run --with pillow python scripts/generate_terminal_demo.py

The rendered transcript is the standard output of the credential-free
``examples/example_cooldown_resume.py``; rerun it and update ``OUTPUT_LINES``
if its output changes. Pillow is a generation-time tool, not a package dependency.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets" / "v0.20-quickstart.gif"

# Each frame shows the first N lines of the example's real output.
OUTPUT_LINES = [
    "$ uv run python examples/example_cooldown_resume.py",
    "run 1: 20 items, 5 workers, JSONL checkpoint",
    "429 from the provider: all 5 workers pause 1.0s",
    "cooldown over: workers ramp back up, the item retries",
    "process killed (simulated crash): 12 of 20 items checkpointed",
    "",
    "run 2: same command, ResumePolicy.REUSE_SUCCESSES",
    "replayed 12 from the checkpoint, called the client 8 times",
    "Items:     20 total — 20 succeeded, 0 failed",
]
FRAMES = [OUTPUT_LINES[:n] for n in [2, 3, 4, 5, 7, 8, 9]]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _render(lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (1040, 520), "#10151c")
    draw = ImageDraw.Draw(image)
    title_font = _font(21)
    body_font = _font(19)
    draw.rounded_rectangle((10, 10, 1030, 510), radius=14, fill="#17202a", outline="#34495e")
    draw.ellipse((30, 29, 45, 44), fill="#ff5f57")
    draw.ellipse((53, 29, 68, 44), fill="#febc2e")
    draw.ellipse((76, 29, 91, 44), fill="#28c840")
    draw.text((112, 27), "llm-minder · local demo", font=title_font, fill="#c9d1d9")
    y = 76
    for index, line in enumerate(lines):
        color = "#7ee787" if index == 0 else "#d6deeb"
        if line.startswith(("429", "cooldown")):
            color = "#f2cc60"
        elif line.startswith("process killed"):
            color = "#ff7b72"
        elif line.startswith("replayed"):
            color = "#79c0ff"
        elif line.startswith("Items:"):
            color = "#7ee787"
        draw.text((34, y), line, font=body_font, fill=color)
        y += 40
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    images = [_render(lines) for lines in FRAMES]
    images[0].save(
        OUTPUT,
        save_all=True,
        append_images=images[1:],
        duration=[900, 1_400, 1_200, 1_600, 1_000, 1_400, 3_500],
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
