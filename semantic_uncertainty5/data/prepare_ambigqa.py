#!/usr/bin/env python3
"""Download and safely prepare the official AmbigQA development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import BinaryIO, Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile


OFFICIAL_ARCHIVE_URL = (
    "https://nlp.cs.washington.edu/ambigqa/data/ambignq_light.zip"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "3f5dada69dec05cef1533a64945cd7bafde1aa94b0cdd6fa9a22f881206220db"
)
OFFICIAL_HOST = "nlp.cs.washington.edu"
DEV_MEMBER_NAME = "dev_light.json"
EXPECTED_EXAMPLES = 2002
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_DEV_BYTES = 16 * 1024 * 1024
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "ambig_qa" / DEV_MEMBER_NAME


class PreparationError(RuntimeError):
    """Raised when downloaded data fails an integrity or format check."""


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_official_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != OFFICIAL_HOST:
        raise PreparationError(
            f"Refusing non-official download URL: {url!r}. "
            f"Expected HTTPS on {OFFICIAL_HOST}."
        )


def download_official_archive(
    destination: Path,
    *,
    timeout: float = 60.0,
    opener: Callable[..., BinaryIO] = urlopen,
) -> str:
    """Download the pinned official archive to *destination*.

    ``opener`` is injectable so the download logic can be unit-tested without
    network access.
    """

    _require_official_https_url(OFFICIAL_ARCHIVE_URL)
    request = Request(
        OFFICIAL_ARCHIVE_URL,
        headers={"User-Agent": "ShaQ-AmbigQA-preparer/1.0"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with opener(request, timeout=timeout) as response:
        final_url = response.geturl()
        _require_official_https_url(final_url)

        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_ARCHIVE_BYTES:
            raise PreparationError(
                f"Archive is larger than the {MAX_ARCHIVE_BYTES}-byte limit."
            )

        total = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise PreparationError(
                        f"Archive exceeded the {MAX_ARCHIVE_BYTES}-byte limit."
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

    digest = sha256_file(destination)
    if digest != OFFICIAL_ARCHIVE_SHA256:
        raise PreparationError(
            "Official archive checksum mismatch: "
            f"expected {OFFICIAL_ARCHIVE_SHA256}, received {digest}."
        )
    return digest


def verify_official_archive(path: Path) -> str:
    """Verify that *path* is the pinned official AmbigNQ light archive."""

    if not path.is_file():
        raise PreparationError(f"Archive does not exist or is not a file: {path}")
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise PreparationError(
            f"Archive is larger than the {MAX_ARCHIVE_BYTES}-byte limit."
        )
    digest = sha256_file(path)
    if digest != OFFICIAL_ARCHIVE_SHA256:
        raise PreparationError(
            "Archive checksum mismatch: "
            f"expected {OFFICIAL_ARCHIVE_SHA256}, received {digest}."
        )
    return digest


def _safe_dev_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    candidates: list[zipfile.ZipInfo] = []

    for info in archive.infolist():
        normalized_name = info.filename.replace("\\", "/")
        member_path = PurePosixPath(normalized_name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise PreparationError(f"Unsafe ZIP member path: {info.filename!r}")

        unix_mode = info.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise PreparationError(
                f"ZIP archive contains a symbolic link: {info.filename!r}"
            )

        if not info.is_dir() and member_path.name == DEV_MEMBER_NAME:
            candidates.append(info)

    if len(candidates) != 1:
        raise PreparationError(
            f"Expected exactly one {DEV_MEMBER_NAME!r} member; "
            f"found {len(candidates)}."
        )

    member = candidates[0]
    if member.file_size > MAX_DEV_BYTES:
        raise PreparationError(
            f"{DEV_MEMBER_NAME} is larger than the {MAX_DEV_BYTES}-byte limit."
        )
    if member.compress_size == 0 and member.file_size != 0:
        raise PreparationError(f"Invalid compressed size for {DEV_MEMBER_NAME}.")
    if member.compress_size and member.file_size / member.compress_size > 100:
        raise PreparationError(f"Suspicious compression ratio for {DEV_MEMBER_NAME}.")
    return member


def validate_dev_json(path: Path) -> tuple[int, int, int]:
    """Validate the official split and return total/ambiguous/unambiguous counts."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"Invalid {DEV_MEMBER_NAME}: {exc}") from exc

    if not isinstance(records, list) or len(records) != EXPECTED_EXAMPLES:
        actual = len(records) if isinstance(records, list) else type(records).__name__
        raise PreparationError(
            f"Expected {EXPECTED_EXAMPLES} development examples; found {actual}."
        )

    ambiguous = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PreparationError(f"Record {index} is not a JSON object.")
        if not isinstance(record.get("id"), str) or not record["id"]:
            raise PreparationError(f"Record {index} has no valid string id.")
        if not isinstance(record.get("question"), str) or not record["question"]:
            raise PreparationError(f"Record {index} has no valid question.")
        annotations = record.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            raise PreparationError(f"Record {index} has no annotation list.")
        if not all(isinstance(annotation, dict) for annotation in annotations):
            raise PreparationError(f"Record {index} contains an invalid annotation.")
        if any(
            annotation.get("type") == "multipleQAs" for annotation in annotations
        ):
            ambiguous += 1

    unambiguous = len(records) - ambiguous
    if ambiguous == 0 or unambiguous == 0:
        raise PreparationError("Development data does not contain both ambiguity classes.")
    return len(records), ambiguous, unambiguous


def extract_dev_json(
    archive_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Safely copy only ``dev_light.json`` from *archive_path* to *output_path*."""

    if archive_path.resolve() == output_path.resolve():
        raise PreparationError("Archive and output paths must be different.")
    if output_path.exists() and not overwrite:
        raise PreparationError(
            f"Output already exists: {output_path}. Pass --force to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            member = _safe_dev_member(archive)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_path.parent,
            )
            temporary_path = Path(temporary_name)
            bytes_written = 0
            with os.fdopen(file_descriptor, "wb") as destination:
                with archive.open(member, "r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > MAX_DEV_BYTES:
                            raise PreparationError(
                                f"{DEV_MEMBER_NAME} exceeded the extraction size limit."
                            )
                        destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())

        if bytes_written != member.file_size:
            raise PreparationError(
                f"Extracted size mismatch: expected {member.file_size}, "
                f"received {bytes_written}."
            )
        counts = validate_dev_json(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
        return counts
    except (OSError, zipfile.BadZipFile) as exc:
        raise PreparationError(f"Could not read AmbigNQ archive: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official AmbigNQ light archive and safely extract "
            "its 2,002-example dev_light.json split."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing official ambignq_light.zip instead of downloading it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace --output if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()

    try:
        if args.archive is not None:
            archive_path = args.archive.expanduser().resolve()
            digest = verify_official_archive(archive_path)
            counts = extract_dev_json(
                archive_path,
                output_path,
                overwrite=args.force,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="shaq-ambigqa-") as temp_dir:
                archive_path = Path(temp_dir) / "ambignq_light.zip"
                digest = download_official_archive(archive_path)
                counts = extract_dev_json(
                    archive_path,
                    output_path,
                    overwrite=args.force,
                )
    except PreparationError as exc:
        raise SystemExit(f"error: {exc}") from exc

    total, ambiguous, unambiguous = counts
    print(f"Verified archive SHA-256: {digest}")
    print(f"Wrote: {output_path}")
    print(
        f"Validated {total} examples "
        f"({ambiguous} ambiguous, {unambiguous} unambiguous)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
