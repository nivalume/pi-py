"""Tests for the AWS plumbing behind the Bedrock adapter (SigV4, event streams, credentials)."""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from pi_ai.utils import aws
from pi_ai.utils.aws import (
    AwsCredentials,
    AwsCredentialsError,
    EventStreamDecoder,
    EventStreamError,
    derive_signing_key,
    encode_event_stream_message,
    load_shared_profiles,
    resolve_aws_credentials,
    resolve_profile_region,
    sign_request,
)

# -- SigV4 (vectors from the AWS docs / aws-sig-v4-test-suite) --------------------------------------

EXAMPLE = AwsCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
TEST_SUITE_DATE = datetime(2015, 8, 30, 12, 36, 0, tzinfo=UTC)


def _signature(headers: dict[str, str]) -> str:
    return headers["authorization"].rsplit("Signature=", 1)[1]


def test_signing_key_example():
    key = derive_signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "20120215", "us-east-1", "iam")
    assert key.hex() == "f4780e2d9f65fa895f9c67b32ce1baf0b0d8a43505a000a1a9e090d414db404d"


@pytest.mark.parametrize(
    ("method", "url", "headers", "body", "credentials", "service", "signed", "signature"),
    [
        # get-vanilla
        (
            "GET",
            "https://example.amazonaws.com/",
            {"Host": "example.amazonaws.com"},
            b"",
            EXAMPLE,
            "service",
            "host;x-amz-date",
            "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        ),
        # post-vanilla
        (
            "POST",
            "https://example.amazonaws.com/",
            {},
            b"",
            EXAMPLE,
            "service",
            "host;x-amz-date",
            "5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
        ),
        # get-vanilla-query-order-key-case
        (
            "GET",
            "https://example.amazonaws.com/?Param2=value2&Param1=value1",
            {},
            b"",
            EXAMPLE,
            "service",
            "host;x-amz-date",
            "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500",
        ),
        # post-x-www-form-urlencoded
        (
            "POST",
            "https://example.amazonaws.com/",
            {"Content-Type": "application/x-www-form-urlencoded"},
            b"Param1=value1",
            EXAMPLE,
            "service",
            "content-type;host;x-amz-date",
            "ff11897932ad3f4e8b18135d722051e5ac45fc38421b1da7b9d196a0fe09473a",
        ),
        # IAM ListUsers example from the SigV4 documentation
        (
            "GET",
            "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
            {"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
            b"",
            EXAMPLE,
            "iam",
            "content-type;host;x-amz-date",
            "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7",
        ),
        # post-sts-token/post-sts-header-before
        (
            "POST",
            "https://example.amazonaws.com/",
            {},
            b"",
            AwsCredentials(
                EXAMPLE.access_key_id,
                EXAMPLE.secret_access_key,
                "AQoDYXdzEPT//////////wEXAMPLEtc764bNrC9SAPBSM22wDOk4x4HIZ8j4FZTwdQWLWsKWHGBuFqwAeMicRXmxfpSPfIeoIYRqTflfKD8YUuwthAx7mSEI/qkPpKPi/kMcGdQrmGdeehM4IC1NtBmUpp2wUE8phUZampKsburEDy0KPkyQDYwT7WZ0wq5VSXDvp75YU9HFvlRd8Tx6q6fE8YQcHNVXAkiY9q6d+xo0rKwT38xVqr7ZD0u0iPPkUL64lIZbqBAz+scqKmlzm8FDrypNC9Yjc8fPOLn9FX9KSYvKTr4rvx3iSIlTJabIQwj2ICCR/oLxBA==",
            ),
            "service",
            "host;x-amz-date;x-amz-security-token",
            "85d96828115b5dc0cfc3bd16ad9e210dd772bbebba041836c64533a82be05ead",
        ),
    ],
)
def test_sigv4_test_vectors(method, url, headers, body, credentials, service, signed, signature):
    result = sign_request(
        method,
        url,
        headers,
        body,
        credentials,
        region="us-east-1",
        service=service,
        now=TEST_SUITE_DATE,
        apply_checksum=False,
    )
    assert f"SignedHeaders={signed}," in result["authorization"]
    assert _signature(result) == signature
    assert result["x-amz-date"] == "20150830T123600Z"


def test_sigv4_matches_the_aws_js_sdk_for_bedrock_requests():
    # Produced by @smithy/signature-v4 (the AWS JS SDK signer): double-encoded ARN path, session
    # token, x-amz-content-sha256, header whitespace normalization, unsigned user-agent.
    body = b'{"messages":[{"role":"user","content":[{"text":"hi"}]}]}'
    result = sign_request(
        "POST",
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123%3Aapplication-inference-profile%2Fabc/converse-stream",
        {"content-type": "application/json", "x-custom": "  a   b ", "user-agent": "pi"},
        body,
        AwsCredentials("AKID", "SECRET", "TOKEN"),
        region="us-west-2",
        service="bedrock",
        now=datetime(2026, 10, 18, 6, 0, 0, tzinfo=UTC),
    )
    assert result["x-amz-content-sha256"] == "c3335e07d6e89650bf94cb98664be69f8774dbc3198df221f113db0a59ea14f0"
    assert result["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKID/20261018/us-west-2/bedrock/aws4_request, "
        "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token;x-custom, "
        "Signature=c069fae1b619d5fcd409574647d8bd2d83c5209e40ebe093649af90cc4c61334"
    )


def test_sign_request_replaces_stale_auth_headers():
    result = sign_request(
        "POST",
        "http://127.0.0.1:8080/x",
        {"Authorization": "old", "X-Amz-Date": "old"},
        b"",
        EXAMPLE,
        region="r",
        service="s",
        now=TEST_SUITE_DATE,
    )
    assert result["host"] == "127.0.0.1:8080"
    assert "Authorization" not in result and "X-Amz-Date" not in result
    assert result["authorization"].startswith("AWS4-HMAC-SHA256 ")


# -- event stream codec -----------------------------------------------------------------------------

# Encoded by @smithy/eventstream-codec (the AWS JS SDK codec).
JS_EVENT_FRAME = bytes.fromhex(
    "0000009f00000057c37babff0d3a6d6573736167652d747970650700056576656e740b3a6576656e742d74797065070011636f6e74656e74"
    "426c6f636b44656c74610d3a636f6e74656e742d747970650700106170706c69636174696f6e2f6a736f6e7b22636f6e74656e74426c6f63"
    "6b496e646578223a302c2264656c7461223a7b2274657874223a226869227d2c2270223a2261626364227dcf16db44"
)
JS_ALL_HEADER_TYPES_FRAME = bytes.fromhex(
    "00000050000000402bed5132017400016601016904fffffffb0262790207017303012c0362696e0600030102030175096f1c2b9e1c2d4e5f"
    "8a9b0c1d2e3f4a5b027473080000018bcfe5687b5cc9322f"
)


def test_encoder_matches_the_js_codec():
    encoded = encode_event_stream_message(
        {":message-type": "event", ":event-type": "contentBlockDelta", ":content-type": "application/json"},
        b'{"contentBlockIndex":0,"delta":{"text":"hi"},"p":"abcd"}',
    )
    assert encoded == JS_EVENT_FRAME


def test_decodes_all_header_types_bytewise():
    decoder = EventStreamDecoder()
    messages = []
    for index in range(len(JS_ALL_HEADER_TYPES_FRAME)):
        messages.extend(decoder.feed(JS_ALL_HEADER_TYPES_FRAME[index : index + 1]))
    decoder.finish()
    assert len(messages) == 1
    assert messages[0].payload == b""
    assert messages[0].headers == {
        "t": True,
        "f": False,
        "i": -5,
        "by": 7,
        "s": 300,
        "bin": b"\x01\x02\x03",
        "u": uuid.UUID("6f1c2b9e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"),
        "ts": 1700000000123,
    }


def test_round_trip_multiple_messages_in_one_chunk():
    frames = encode_event_stream_message({"a": "1"}, b"one") + encode_event_stream_message({"b": 2**40}, b"two")
    decoder = EventStreamDecoder()
    messages = decoder.feed(frames + frames[:5])
    assert [(m.headers, m.payload) for m in messages] == [({"a": "1"}, b"one"), ({"b": 2**40}, b"two")]
    with pytest.raises(EventStreamError, match="incomplete message"):
        decoder.finish()


def test_checksum_mismatches_are_detected():
    frame = bytearray(JS_EVENT_FRAME)
    frame[9] ^= 1
    with pytest.raises(EventStreamError, match="prelude checksum"):
        EventStreamDecoder().feed(bytes(frame))
    frame = bytearray(JS_EVENT_FRAME)
    frame[40] ^= 1
    with pytest.raises(EventStreamError, match="message checksum"):
        EventStreamDecoder().feed(bytes(frame))


# -- shared files and credential chain --------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(aws, "_botocore_available", lambda: False)
    aws.clear_credentials_cache()
    yield
    aws.clear_credentials_cache()


def test_shared_profiles_merge_config_and_credentials(tmp_path):
    (tmp_path / "config").write_text(
        "[default]\nregion = us-west-1\n\n[profile dev]\nregion=eu-west-1 # comment\ns3 =\n  max_concurrent_requests = 5\n"
        "[sso-session corp]\nsso_region = us-east-1\n[ignored]\nregion = nope\n"
    )
    (tmp_path / "credentials").write_text("; comment\n[dev]\naws_access_key_id = AKIDDEV\naws_secret_access_key = s\n")
    profiles = load_shared_profiles()
    assert profiles["default"] == {"region": "us-west-1"}
    assert profiles["dev"] == {
        "region": "eu-west-1",
        "s3.max_concurrent_requests": "5",
        "aws_access_key_id": "AKIDDEV",
        "aws_secret_access_key": "s",
    }
    assert "ignored" not in profiles and "corp" not in profiles
    assert resolve_profile_region("dev") == "eu-west-1"
    assert resolve_profile_region(None) == "us-west-1"


async def test_env_keys_are_skipped_for_explicit_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDENV")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    (tmp_path / "credentials").write_text(
        "[p]\naws_access_key_id = AKIDFILE\naws_secret_access_key = s2\naws_session_token = t\n"
    )
    assert (await resolve_aws_credentials()).access_key_id == "AKIDENV"
    assert await resolve_aws_credentials(profile="p") == AwsCredentials("AKIDFILE", "s2", "t")


async def test_no_credentials():
    with pytest.raises(AwsCredentialsError, match="Could not load credentials from any providers"):
        await resolve_aws_credentials()


async def test_container_credentials(mock_server, monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/credentials/abc")
    (tmp_path / "token").write_text("secret-token\n")
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", str(tmp_path / "token"))
    mock_server.add_json(
        {"AccessKeyId": "AKIDECS", "SecretAccessKey": "s", "Token": "t", "Expiration": "2099-01-01T00:00:00Z"}
    )
    client = mock_server.client()
    credentials = await resolve_aws_credentials(http_client=client)
    assert credentials.access_key_id == "AKIDECS"
    assert credentials.session_token == "t"
    assert credentials.expiration == datetime(2099, 1, 1, tzinfo=UTC).timestamp()
    assert mock_server.last.url == "http://169.254.170.2/v2/credentials/abc"
    assert mock_server.last.headers["authorization"] == "secret-token"
    # Cached until shortly before expiration.
    assert (await resolve_aws_credentials(http_client=client)).access_key_id == "AKIDECS"
    assert len(mock_server.requests) == 1


async def test_container_full_uri_must_be_local_or_https(monkeypatch):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://evil.example.com/creds")
    with pytest.raises(AwsCredentialsError, match="Unsupported host"):
        await resolve_aws_credentials()


STS_RESPONSE = """<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleWithWebIdentityResult>
    <Credentials>
      <AccessKeyId>ASIASTS</AccessKeyId>
      <SecretAccessKey>stssecret</SecretAccessKey>
      <SessionToken>ststoken</SessionToken>
      <Expiration>2099-01-01T00:00:00Z</Expiration>
    </Credentials>
  </AssumeRoleWithWebIdentityResult>
</AssumeRoleWithWebIdentityResponse>"""


async def test_web_identity_token_file(mock_server, monkeypatch, tmp_path):
    (tmp_path / "jwt").write_text("eyJ.token\n")
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(tmp_path / "jwt"))
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123:role/r")
    monkeypatch.setenv("AWS_ROLE_SESSION_NAME", "sess")
    mock_server.add(httpx.Response(200, text=STS_RESPONSE))
    credentials = await resolve_aws_credentials(region="eu-west-1", http_client=mock_server.client())
    assert credentials == AwsCredentials(
        "ASIASTS", "stssecret", "ststoken", datetime(2099, 1, 1, tzinfo=UTC).timestamp()
    )
    request = mock_server.last
    assert request.url == "https://sts.eu-west-1.amazonaws.com/"
    assert "authorization" not in request.headers
    form = dict(pair.split("=", 1) for pair in request.body.decode().split("&"))
    assert form["Action"] == "AssumeRoleWithWebIdentity"
    assert form["RoleSessionName"] == "sess"
    assert form["WebIdentityToken"] == "eyJ.token"


async def test_assume_role_with_source_profile(mock_server, tmp_path):
    (tmp_path / "config").write_text(
        "[profile admin]\nrole_arn = arn:aws:iam::123:role/admin\nsource_profile = base\nexternal_id = ext\n"
    )
    (tmp_path / "credentials").write_text("[base]\naws_access_key_id = AKIDBASE\naws_secret_access_key = s\n")
    mock_server.add(httpx.Response(200, text=STS_RESPONSE.replace("WithWebIdentity", "")))
    credentials = await resolve_aws_credentials(profile="admin", http_client=mock_server.client())
    assert credentials.access_key_id == "ASIASTS"
    request = mock_server.last
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDBASE/")
    assert "/us-east-1/sts/aws4_request" in request.headers["authorization"]
    assert b"Action=AssumeRole&" in request.body and b"ExternalId=ext" in request.body


async def test_sts_errors_are_reported(mock_server, tmp_path):
    (tmp_path / "config").write_text("[profile admin]\nrole_arn = arn:aws:iam::123:role/admin\nsource_profile = base\n")
    (tmp_path / "credentials").write_text("[base]\naws_access_key_id = A\naws_secret_access_key = s\n")
    mock_server.add(
        httpx.Response(
            403, text="<ErrorResponse><Error><Code>AccessDenied</Code><Message>no</Message></Error></ErrorResponse>"
        )
    )
    with pytest.raises(AwsCredentialsError, match="STS AssumeRole failed: AccessDenied: no"):
        await resolve_aws_credentials(profile="admin", http_client=mock_server.client())


async def test_credential_process(tmp_path):
    script = tmp_path / "creds.py"
    script.write_text(
        'import json; print(json.dumps({"Version": 1, "AccessKeyId": "AKIDPROC", "SecretAccessKey": "s"}))'
    )
    (tmp_path / "config").write_text(f"[profile proc]\ncredential_process = {sys.executable} {script}\n")
    assert (await resolve_aws_credentials(profile="proc")).access_key_id == "AKIDPROC"


async def test_sso_profiles_require_botocore(tmp_path):
    (tmp_path / "config").write_text("[profile sso]\nsso_session = corp\nsso_account_id = 1\n")
    with pytest.raises(AwsCredentialsError, match="install botocore"):
        await resolve_aws_credentials(profile="sso")


async def test_instance_metadata_v2(mock_server, monkeypatch):
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "false")
    responses: list[Any] = [
        httpx.Response(200, text="imds-token"),
        httpx.Response(200, text="my-role\n"),
        httpx.Response(200, json={"Code": "Success", "AccessKeyId": "AKIDIMDS", "SecretAccessKey": "s", "Token": "t"}),
    ]
    for response in responses:
        mock_server.add(response)
    credentials = await resolve_aws_credentials(http_client=mock_server.client())
    assert credentials.access_key_id == "AKIDIMDS"
    assert [r.method for r in mock_server.requests] == ["PUT", "GET", "GET"]
    assert mock_server.requests[2].url == "http://169.254.169.254/latest/meta-data/iam/security-credentials/my-role"
    assert mock_server.requests[2].headers["x-aws-ec2-metadata-token"] == "imds-token"


async def test_botocore_chain_is_preferred_when_available(monkeypatch):
    calls: list[Any] = []

    def fake(profile: str | None) -> AwsCredentials:
        calls.append(profile)
        return AwsCredentials("AKIDBOTO", "s")

    monkeypatch.setattr(aws, "_botocore_credentials", fake)
    assert (await resolve_aws_credentials(profile="p", use_botocore=True)).access_key_id == "AKIDBOTO"
    assert calls == ["p"]
