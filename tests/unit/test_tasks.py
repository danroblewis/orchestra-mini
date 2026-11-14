"""The owner's tasks: the built-in worker, and the facts the Dashboard's box makes."""

from orchestra import tasks
from orchestra.config.schema import Machine
from orchestra.project import load


def test_the_worker_is_built_in_unless_the_config_names_one(make_project):
    (w,) = load(make_project.repo).config.machines
    assert w.name == "worker" and w.on == ["owner.task.requested", "owner.task.replied"]
    assert w.command.startswith("orchestra agent ") and tasks.PROMPT.is_file()
    own = Machine(name="worker", command="./mine.sh", on=["x"])
    assert tasks.with_builtin([own]) == [own]


def test_a_request_is_a_new_task_and_a_reply_stays_on_it():
    ev, why = tasks.request("  Add a README\nthat says what this is  ")
    assert why is None and ev["type"] == "owner.task.requested" and ev["subject"].startswith("task/")
    assert ev["data"]["title"] == "Add a README"
    assert ev["data"]["waiting_on"] == "worker"
    reply, _ = tasks.request("yes, MIT", ev["subject"])
    assert reply["type"] == "owner.task.replied" and reply["subject"] == ev["subject"]


def test_refused_requests():
    assert tasks.request("   ")[1] == "say what you want done"
    assert tasks.request("x" * (tasks.TEXT_MAX + 1))[1].startswith("at most")
    assert tasks.request("hi", "github/pull/1")[1] == "replies go to tasks only"
    assert tasks.title("a" * 200).endswith("…") and len(tasks.title("a" * 200)) == tasks.TITLE_MAX
