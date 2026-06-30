"""Tests for enotropos — parent chunk persistence."""
import tempfile
from pathlib import Path

from winegpt.parents import delete_parents, load_parent, load_parents, save_parents


def test_save_and_load_parent() -> None:
    """Saving parents and loading one back should preserve content."""
    with tempfile.TemporaryDirectory() as tmp:
        # Override PARENTS_DIR to a temp location
        from winegpt import parents as parents_module

        original_dir = parents_module.PARENTS_DIR
        parents_module.PARENTS_DIR = Path(tmp)

        try:
            parents = [
                {
                    "parent_id": "Espanya__DOP_Rioja__DOP_Rioja__parent_0",
                    "section": "Varietats",
                    "markdown": "## Varietats\n\nTempranillo",
                },
            ]
            save_parents("Espanya", "DOP_Rioja", "DOP_Rioja", parents)

            loaded = load_parent("Espanya__DOP_Rioja__DOP_Rioja__parent_0")
            assert loaded is not None
            assert loaded["section"] == "Varietats"
            assert "Tempranillo" in loaded["markdown"]
        finally:
            parents_module.PARENTS_DIR = original_dir


def test_load_parents_batch() -> None:
    """load_parents should retrieve multiple parents in one call."""
    with tempfile.TemporaryDirectory() as tmp:
        from winegpt import parents as parents_module

        original_dir = parents_module.PARENTS_DIR
        parents_module.PARENTS_DIR = Path(tmp)

        try:
            parents = [
                {
                    "parent_id": "Coneixement__enologia__doc__parent_0",
                    "section": "Titol",
                    "markdown": "Text A",
                },
                {
                    "parent_id": "Coneixement__enologia__doc__parent_1",
                    "section": "Titol 2",
                    "markdown": "Text B",
                },
            ]
            save_parents("Coneixement", "enologia", "doc", parents)

            result = load_parents({
                "Coneixement__enologia__doc__parent_0",
                "Coneixement__enologia__doc__parent_1",
            })
            assert len(result) == 2
            assert result["Coneixement__enologia__doc__parent_0"]["markdown"] == "Text A"
            assert result["Coneixement__enologia__doc__parent_1"]["markdown"] == "Text B"
        finally:
            parents_module.PARENTS_DIR = original_dir


def test_delete_parents() -> None:
    """delete_parents should remove the whole country directory."""
    with tempfile.TemporaryDirectory() as tmp:
        from winegpt import parents as parents_module

        original_dir = parents_module.PARENTS_DIR
        parents_module.PARENTS_DIR = Path(tmp)

        try:
            save_parents("Espanya", "DOP_X", "DOP_X", [{
                "parent_id": "Espanya__DOP_X__DOP_X__parent_0",
                "section": "S",
                "markdown": "M",
            }])
            country_dir = Path(tmp) / "Espanya"
            assert country_dir.exists()

            delete_parents("Espanya")
            assert not country_dir.exists()
        finally:
            parents_module.PARENTS_DIR = original_dir
