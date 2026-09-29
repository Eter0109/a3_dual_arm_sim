"""Command dispatch contracts; no simulator or model construction is required."""

from types import SimpleNamespace

import pytest

from a3_dual_arm_sim.cli import main as cli


@pytest.mark.parametrize("arguments", [["--help"], *[[group, "--help"] for group in cli.COMMANDS]])
def test_group_help_does_not_import_command_modules(monkeypatch, capsys, arguments):
    def unexpected_import(_name):
        pytest.fail("Help must not load simulator or model command modules")

    monkeypatch.setattr(cli.importlib, "import_module", unexpected_import)
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 0
    assert "a3-sim" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("group", "name", "command"),
    [(group, name, command) for group, entries in cli.COMMANDS.items()
     for name, command in entries.items()],
)
def test_selected_command_receives_exact_remaining_arguments(monkeypatch, group, name, command):
    calls = []

    def fake_import(module_name):
        calls.append(module_name)

        def selected_main(argv):
            calls.append(argv)
            return 1

        return SimpleNamespace(main=selected_main)

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    options = ["--seed", "12", "--task", "pack five cookies"]
    assert cli.main([group, name, *options]) == 1
    assert calls == [f"a3_dual_arm_sim.cli.{command.module}", [*command.prefix, *options]]


def test_command_help_is_forwarded_to_selected_parser(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda _name: SimpleNamespace(main=lambda argv: calls.append(argv)))
    assert cli.main(["agent", "run", "--help"]) == 0
    assert calls == [["--help"]]


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["sim"], ["sim", "collect-grasp"]])
def test_missing_and_removed_commands_are_rejected_without_import(monkeypatch, arguments):
    monkeypatch.setattr(cli.importlib, "import_module",
                        lambda _name: pytest.fail("Invalid commands must never be imported"))
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 2


def test_environment_command_handles_explicit_help(capsys):
    from a3_dual_arm_sim.cli.diagnostics.environment import main

    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    assert "a3-sim inspect environment" in capsys.readouterr().out
