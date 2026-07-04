"""Tool modules — import to trigger auto-registration in the tool registry.

Every tool module in this package registers itself at import time by
calling ``tool_registry.register(name, schema, handler)`` at module
level.  Importing this package ensures all tools are available before
any Gemini session starts.
"""

from __future__ import annotations

# Import all tool modules so they register themselves.
# Order does not matter — each module is self-contained.
from voice_agent.tools import appointment as _appointment  # noqa: F401
from voice_agent.tools import faq as _faq  # noqa: F401
from voice_agent.tools import sms as _sms  # noqa: F401
from voice_agent.tools import transfer as _transfer  # noqa: F401
