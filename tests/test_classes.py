"""Tests for semantic class definitions and lookups."""

from rohbau_annotator.classes import (
    CLASS_BY_ID,
    CLASS_BY_NAME,
    CLASSES,
    NUM_CLASSES,
    SemanticClass,
    class_color_norm,
)


class TestClassDefinitions:
    def test_class_count(self):
        """14 total entries: unlabeled(0) + 13 annotatable classes."""
        assert len(CLASSES) == 14
        assert NUM_CLASSES == 13

    def test_ids_are_sequential(self):
        ids = [c.id for c in CLASSES]
        assert ids == list(range(14))

    def test_unlabeled_is_zero(self):
        assert CLASSES[0].id == 0
        assert CLASSES[0].name == "unlabeled"
        assert CLASSES[0].color == (0, 0, 0)

    def test_all_names_unique(self):
        names = [c.name for c in CLASSES]
        assert len(names) == len(set(names))

    def test_all_colors_are_valid_rgb(self):
        for cls in CLASSES:
            assert len(cls.color) == 3
            for channel in cls.color:
                assert 0 <= channel <= 255


class TestLookupDicts:
    def test_lookup_by_id(self):
        wall = CLASS_BY_ID[1]
        assert wall.name == "wall"
        assert isinstance(wall, SemanticClass)

    def test_lookup_by_name(self):
        floor = CLASS_BY_NAME["floor"]
        assert floor.id == 2

    def test_all_classes_in_both_dicts(self):
        for cls in CLASSES:
            assert CLASS_BY_ID[cls.id] is cls
            assert CLASS_BY_NAME[cls.name] is cls

    def test_unknown_id_returns_none(self):
        assert CLASS_BY_ID.get(999) is None

    def test_unknown_name_returns_none(self):
        assert CLASS_BY_NAME.get("nonexistent") is None


class TestClassColorNorm:
    def test_normalized_range(self):
        for cls in CLASSES:
            r, g, b = class_color_norm(cls)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_black_stays_zero(self):
        unlabeled = CLASS_BY_ID[0]
        assert class_color_norm(unlabeled) == (0.0, 0.0, 0.0)

    def test_specific_color(self):
        wall = CLASS_BY_ID[1]
        r, g, b = class_color_norm(wall)
        assert r == 174 / 255.0
        assert g == 199 / 255.0
        assert b == 232 / 255.0


class TestSemanticClassFrozen:
    def test_frozen_dataclass(self):
        """SemanticClass instances should be immutable."""
        import pytest

        cls = CLASSES[0]
        with pytest.raises(AttributeError):
            cls.id = 99
