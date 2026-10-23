import pytest
from pathlib import Path
from src.TextSummarizer.utils.common import (
    read_yaml,
    create_directories,
    get_size,
    save_json,
    load_json
)


def test_create_directories(temp_dir):
    d1 = temp_dir / "dir1"
    d2 = temp_dir / "dir2" / "subdir"
    create_directories([d1, d2])
    assert d1.exists() and d1.is_dir()
    assert d2.exists() and d2.is_dir()


def test_read_yaml_valid(temp_dir):
    yaml_file = temp_dir / "test.yaml"
    with open(yaml_file, "w") as f:
        f.write("key: value\nnumber: 42\n")
    data = read_yaml(yaml_file)
    assert data.key == "value"
    assert data.number == 42


def test_read_yaml_empty(temp_dir):
    yaml_file = temp_dir / "empty.yaml"
    yaml_file.touch()
    with pytest.raises(ValueError):
        read_yaml(yaml_file)


def test_read_yaml_missing(temp_dir):
    with pytest.raises(FileNotFoundError):
        read_yaml(temp_dir / "non_existent.yaml")


def test_get_size(temp_dir):
    f = temp_dir / "sample.txt"
    f.write_text("Hello " * 500)
    size_str = get_size(f)
    assert "KB" in size_str


def test_save_and_load_json(temp_dir):
    json_path = temp_dir / "metrics.json"
    data = {"rouge1": 0.45, "status": "passed"}
    save_json(json_path, data)
    assert json_path.exists()

    loaded = load_json(json_path)
    assert loaded.rouge1 == 0.45
    assert loaded.status == "passed"
