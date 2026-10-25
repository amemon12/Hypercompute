"""Pool Room bridge: runs next to the Hypercompute gateway, claims prompts from Convex and streams answers back.

It only makes outbound HTTPS calls (to https://NAME.convex.site/bridge/*) and loopback calls to the gateway. The
gateway owner token is read from /api/owner/bootstrap on 127.0.0.1 and never sent anywhere else.
"""

__version__ = "0.1.0"
