# Diana Bot Control Center

Run `start_gui.bat` on Windows, or `python gui.py`.

The GUI is intentionally a thin launcher/control layer. It installs dependencies, edits the existing connection/persona files, starts/stops `bot.py`, and displays the bot's current coarse processing stage. It does not redesign Diana's conversation, memory, Discord, image, or evolution architecture.

## Credential separation

The existing three API-key files remain independent:

- `config/google_api_key_chat.txt`
- `config/google_api_key_evolution.txt`
- `config/google_api_key_image.txt`

Discord remains `config/discord_token.txt`. The GUI preserves extra lines in the evolution key file when editing its first-line key, so the evolution model stored on line 2 is not destroyed.

## GUI-managed non-secret settings

`config/gui_config.json` stores Local/Web/Image endpoints and model names plus temperature/thinking-token settings. `bot.py` only reads the settings needed to honor the GUI; its main control flow remains unchanged.

## Persona

The Character page supports the current Diana persona, a blank schema-compatible template, and three LLM-generated candidates. Persona generation uses the configured Local API and does not add a new routing layer.
