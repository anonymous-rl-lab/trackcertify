"""CLI tests."""
import json

from track_certify.cli import main


def test_cli_demo(capsys):
    rc = main(["demo", "--r", "0.8", "--delta", "0.05", "--seed", "3"])
    out = capsys.readouterr().out
    assert rc == 0 and "certificate=" in out and "CORRECT" in out


def test_cli_envelope_feasible_and_refusal(capsys):
    rc = main(["envelope", "--n0", "500000", "--delta", "0.01", "--d", "3",
               "--c-max", "1.0"])
    assert rc == 0 and "feasible" in capsys.readouterr().out
    rc = main(["envelope", "--n0", "100", "--delta", "0.01", "--d", "3",
               "--c-max", "1.0"])
    assert rc == 1 and "REFUSAL" in capsys.readouterr().out


def test_cli_tstar(tmp_path, capsys):
    spec = {"sigma": [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "graphs": {"G12": [0, 1, 2], "G21": [1, 0, 2]},
            "hypothesis": {"graph": "G12", "targets": [0, 2]},
            "amplitudes": [1.0, 1.0]}
    f = tmp_path / "spec.json"
    f.write_text(json.dumps(spec))
    rc = main(["tstar", str(f), "--delta", "0.01"])
    out = capsys.readouterr().out
    assert rc == 0 and "T*" in out and "coupling tax" in out
