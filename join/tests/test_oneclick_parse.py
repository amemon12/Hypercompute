"""Invite parsing, redaction and the invite prompt (PACKAGING-SPEC 2.4 step 4, 2.6). Synthetic tickets only."""

from __future__ import annotations

import pytest
from oneclick import read_invite
from oneclick_core import Stop
from oneclick_invite import InviteError, make_redactor, normalize_code, parse_invite, redact_generic
from oneclick_testkit import CODE, INVITE_LINE, TICKET, make_ctx


def test_full_invite_line() -> None:
    inv = parse_invite(INVITE_LINE)
    assert (inv.ticket, inv.code, inv.tail) == (TICKET, CODE, TICKET[-4:])


@pytest.mark.parametrize("text", [f"{TICKET} {CODE}", f"{CODE} {TICKET}", f"  {CODE}\t{TICKET}\n"])
def test_two_bare_tokens_in_either_order(text: str) -> None:
    inv = parse_invite(text)
    assert (inv.ticket, inv.code) == (TICKET, CODE)


@pytest.mark.parametrize("pasted", ["abcde-fghjk", "ABCDEFGHJK", "abcdefghjk", "ABCDE FGHJK".replace(" ", "")])
def test_code_forms_normalize(pasted: str) -> None:
    inv = parse_invite(f"uv run hypercompute join --ticket {TICKET} --code {pasted}")
    assert inv.code == CODE
    assert inv.pasted_code == pasted
    assert set(inv.code_forms()) == {pasted, CODE, "ABCDEFGHJK"}


def test_quoted_tokens_are_accepted() -> None:
    assert parse_invite(f"uv run hypercompute join --ticket '{TICKET}' --code \"{CODE}\"").ticket == TICKET


@pytest.mark.parametrize("raw", ["ABCDE-FGHJ", "ABCDE-FGHJKL", "ABCDO-FGHJK", "ABCD0-FGHJK", "ABCDI-FGHJ1", ""])
def test_bad_codes(raw: str) -> None:
    assert normalize_code(raw) is None


def test_missing_code() -> None:
    with pytest.raises(InviteError) as err:
        parse_invite(f"uv run hypercompute join --ticket {TICKET}")
    assert not err.value.lan


def test_lan_invite_is_flagged() -> None:
    with pytest.raises(InviteError) as err:
        parse_invite(f"uv run hypercompute join --host 192.168.1.20:8000 --code {CODE}")
    assert err.value.lan


def test_short_ticket_is_rejected() -> None:
    with pytest.raises(InviteError):
        parse_invite(f"uv run hypercompute join --ticket {'a' * 20} --code {CODE}")


def test_paste_over_8_kib_is_rejected() -> None:
    with pytest.raises(InviteError):
        parse_invite(f"uv run hypercompute join --ticket {'a' * 9 * 1024} --code {CODE}")


@pytest.mark.parametrize("text", ["hello", f"{TICKET}", f"{CODE} {CODE}", f"{TICKET} {TICKET}", "a b c"])
def test_other_shapes_are_rejected(text: str) -> None:
    with pytest.raises(InviteError):
        parse_invite(text)


def test_ticket_with_symbols_is_rejected() -> None:
    with pytest.raises(InviteError):
        parse_invite(f"{TICKET[:-1]}/ {CODE}")


def test_ticket_check_failure_is_an_invite_error() -> None:
    def broken(_ticket: str) -> None:
        raise RuntimeError("bad ticket bytes")

    with pytest.raises(InviteError):
        parse_invite(INVITE_LINE, broken)


def test_redaction_masks_ticket_codes_and_hashes() -> None:
    redact = make_redactor(TICKET, [CODE, "ABCDEFGHJK", "abcde-fghjk"])
    sha = "0123456789abcdef" * 4
    line = f"join --ticket {TICKET} --code {CODE} raw ABCDEFGHJK low abcde-fghjk sha {sha} multi-level offload"
    out = redact(line)
    assert TICKET not in out and CODE not in out and "ABCDEFGHJK" not in out and "abcde-fghjk" not in out
    assert sha not in out and "<r>" in out
    assert out.count("<ticket>") == 1 and out.count("<code>") == 3
    assert "multi-level offload" in out


def test_generic_redaction_masks_shapes_but_keeps_words() -> None:
    out = redact_generic(f"{INVITE_LINE} pre-fill multi-level XYZ23-ABCDE")
    assert TICKET not in out and CODE not in out
    assert "pre-fill multi-level" in out
    assert "<code>" in out and "<r>" in out


def test_read_invite_prints_only_the_tail(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, secrets=[INVITE_LINE])
    inv = read_invite(ctx)
    assert inv.ticket == TICKET
    text = "\n".join(out)
    assert f"ticket ...{TICKET[-4:]} ({len(TICKET)} characters), code hidden" in text
    assert TICKET not in text and CODE not in text


def test_read_invite_retries_three_times(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, secrets=["nope", "still nope", "no"])
    with pytest.raises(Stop):
        read_invite(ctx)
    assert sum("does not look like a Hypercompute invite" in line for line in out) == 3


def test_read_invite_recovers_on_second_try(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, secrets=["nope", INVITE_LINE])
    assert read_invite(ctx).code == CODE


def test_lan_invite_stops_without_retry(tmp_path) -> None:
    ctx, _ = make_ctx(tmp_path, secrets=[f"uv run hypercompute join --host 10.0.0.2:8000 --code {CODE}", INVITE_LINE])
    with pytest.raises(Stop, match="LAN invite"):
        read_invite(ctx)


def test_hc_invite_is_used_once_and_never_echoed(tmp_path) -> None:
    ctx, out = make_ctx(tmp_path, env={"HC_INVITE": INVITE_LINE})
    assert read_invite(ctx).ticket == TICKET
    assert all(TICKET not in line and CODE not in line for line in out)
    bad, _ = make_ctx(tmp_path, env={"HC_INVITE": "garbage"})
    with pytest.raises(Stop):
        read_invite(bad)


def test_ticket_check_runs_on_real_runs_only(tmp_path) -> None:
    seen: list[str] = []
    ctx, _ = make_ctx(tmp_path, secrets=[INVITE_LINE], ticket_check=seen.append)
    read_invite(ctx)
    dry, out = make_ctx(tmp_path, secrets=[INVITE_LINE], ticket_check=seen.append, dry_run=True)
    read_invite(dry)
    assert seen == [TICKET]
    assert any("skipped the local ticket check" in line for line in out)
