"""Bot package for Discord integration and command handling."""

from .bot import DiscordBot
from .commands import setup_commands

__all__ = ['DiscordBot', 'setup_commands']