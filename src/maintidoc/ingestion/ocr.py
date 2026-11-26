"""Optional local OCR.

Two needs from the brief: scanned pages with no text layer, and text printed
*inside* figures — a label on a diagram is often the answer to the question,
and it is invisible to a text-only index.

Tesseract is driven directly as a subprocess rather than through `pytesseract`,
which is itself only a subprocess wrapper. That removes a dependency and, more
usefully, lets the binary live anywhere: it is frequently installed outside
PATH on Windows.

OCR is off unless an engine is both requested and actually usable. Text it
produces is tagged `ExtractionMethod.OCR` so a reader — and the evaluation —
can tell it apart from a real text layer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Protocol

DEFAULT_TIMEOUT = 60
# Common Windows install locations, tried when nothing is configured.
CANDIDATE_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\tesseract\tesseract.exe",
)


class OcrEngine(Protocol):
    name: str

    def available(self) -> bool: ...

    def read(self, image: bytes, *, language: str = "fra") -> str: ...


class NullOcr:
    """Default engine: does nothing, and says so."""

    name = "none"

    def available(self) -> bool:
        return False

    def read(self, image: bytes, *, language: str = "fra") -> str:
        return ""


def find_tesseract(explicit: str | Path | None = None) -> Path | None:
    """Locate the binary: explicit setting, then PATH, then usual suspects."""
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None
    found = shutil.which("tesseract")
    if found:
        return Path(found)
    return next((Path(p) for p in CANDIDATE_PATHS if Path(p).is_file()), None)


class TesseractOcr:
    """Local Tesseract. No network, no third party service.

    `available()` does not merely check that the file exists: a Tesseract
    missing its DLLs is present and still cannot run. The binary is actually
    executed once, and the result cached.
    """

    name = "tesseract"

    def __init__(
        self,
        binary: str | Path | None = None,
        *,
        tessdata_dir: str | Path | None = None,
        min_characters: int = 12,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.binary = find_tesseract(binary)
        self.tessdata_dir = Path(tessdata_dir) if tessdata_dir else None
        self.min_characters = min_characters
        self.timeout = timeout

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.tessdata_dir:
            env["TESSDATA_PREFIX"] = str(self.tessdata_dir)
        return env

    @lru_cache(maxsize=1)  # noqa: B019 - one engine instance per process
    def _probe(self) -> bool:
        if self.binary is None:
            return False
        try:
            result = subprocess.run(
                [str(self.binary), "--version"],
                capture_output=True,
                timeout=15,
                env=self._environment(),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        # A broken install exits non-zero, or starts and prints nothing.
        return result.returncode == 0 and bool(result.stdout.strip())

    def available(self) -> bool:
        return self._probe()

    def languages(self) -> list[str]:
        if not self.available():
            return []
        assert self.binary is not None
        result = subprocess.run(
            [str(self.binary), "--list-langs"],
            capture_output=True,
            text=True,
            timeout=15,
            env=self._environment(),
            check=False,
        )
        lines = (result.stdout or "").splitlines()
        return [line.strip() for line in lines[1:] if line.strip()]

    def read(self, image: bytes, *, language: str = "fra") -> str:
        if not self.available():
            return ""
        assert self.binary is not None
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "page.png"
            source.write_bytes(image)
            try:
                result = subprocess.run(
                    [str(self.binary), str(source), "stdout", "-l", language],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=self._environment(),
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return ""
        if result.returncode != 0:
            return ""
        cleaned = " ".join((result.stdout or "").split())
        # Below a few characters the output is noise, not content.
        return cleaned if len(cleaned) >= self.min_characters else ""


def default_engine(
    enabled: bool = False,
    *,
    binary: str | Path | None = None,
    tessdata_dir: str | Path | None = None,
) -> OcrEngine:
    """Tesseract when asked for and actually usable, otherwise nothing."""
    if not enabled:
        return NullOcr()
    engine = TesseractOcr(binary, tessdata_dir=tessdata_dir)
    return engine if engine.available() else NullOcr()
