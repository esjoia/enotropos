"""Tests for enotropos — extraction / discovery logic."""
import tempfile
from pathlib import Path

from winegpt.extract import discover_gis


def test_discover_gis_basic() -> None:
    """discover_gis should find DOP_* and IGP_* folders with PDFs."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Create a valid GI folder
        gi_folder = root / "DOP_Rioja"
        gi_folder.mkdir()
        (gi_folder / "DOP_Rioja.pdf").touch()

        # Create a folder without PDFs (should be skipped)
        empty_folder = root / "DOP_Empty"
        empty_folder.mkdir()

        # Create a non-GI folder (should be skipped)
        other = root / "Info"
        other.mkdir()
        (other / "notes.txt").touch()

        # Create an IGP folder
        igp_folder = root / "IGP_Castilla"
        igp_folder.mkdir()
        (igp_folder / "IGP_Castilla.pdf").touch()

        gis = discover_gis(root)
        assert len(gis) == 2

        gi_names = {g["folder_name"] for g in gis}
        assert "DOP_Rioja" in gi_names
        assert "IGP_Castilla" in gi_names

        rioja = next(g for g in gis if g["folder_name"] == "DOP_Rioja")
        assert rioja["gi_type"] == "DOP"
        assert rioja["display_name"] == "Rioja"
        assert rioja["pdfs"] == ["DOP_Rioja.pdf"]


def test_discover_gis_no_pdfs() -> None:
    """Folders without PDFs should be excluded."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "DOP_Empty").mkdir()
        gis = discover_gis(root)
        assert len(gis) == 0


def test_discover_gis_skips_non_gi() -> None:
    """Non-DOP/IGP folders should be skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "Info").mkdir()
        (root / "Info" / "test.pdf").touch()
        gis = discover_gis(root)
        assert len(gis) == 0


def test_discover_gis_multiple_pdfs() -> None:
    """A GI folder with multiple PDFs (e.g., translations)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gi = root / "DOP_Rioja"
        gi.mkdir()
        (gi / "DOP_Rioja.pdf").touch()
        (gi / "DOP_Rioja_EN.pdf").touch()
        (gi / "notes.txt").touch()  # non-PDF, should be ignored

        gis = discover_gis(root)
        assert len(gis) == 1
        assert len(gis[0]["pdfs"]) == 2
