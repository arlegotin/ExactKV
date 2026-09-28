from __future__ import annotations

def test_missing_sysctl_is_unknown(monkeypatch):
    from kvrefine import doctor

    monkeypatch.setattr(doctor, "_run_command", lambda args: (1, "", "denied"))
    report = doctor.collect_environment(smoke=False)
    assert report["ram_bytes"] is None
    assert report["unavailable"]["ram_bytes"]
    assert report["application_budget_bytes"] is None


def test_help_is_offline(monkeypatch, capsys):
    from kvrefine import cli

    def forbid_download(*args, **kwargs):
        raise AssertionError("help attempted a network request")

    monkeypatch.setattr("urllib.request.urlopen", forbid_download)
    assert cli.main(["--help"]) == 0
    assert "doctor" in capsys.readouterr().out


def test_doctor_reports_missing_critical_ram_as_unsupported(tmp_path, monkeypatch):
    from kvrefine import cli, doctor

    monkeypatch.setattr(doctor, "_run_command", lambda args: (1, "", "denied"))
    out = tmp_path / "environment.json"
    assert cli.main(["doctor", "--out", str(out)]) == 4
    assert out.is_file()
