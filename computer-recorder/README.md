# Human Activity Recording Tool

## Installation

Install from source for now. As of now, we've only tested MacOS:

```bash
pip install -e .
```

Make sure to enable recording on your Mac: go to System Preferences $\rightarrow$ Privacy & Security $\rightarrow$ Accessibility, allow recording for the app that you use to edit the code, e.g., vscode.

## Usage

```bash
crec
```

The recorded actions and screenshots will be saved in `~/Downloads/records`.

### Scroll Filtering Options

To reduce unnecessary scroll logging, you can configure scroll filtering parameters:

```bash
# More aggressive filtering (fewer scroll events logged)
crec --scroll-debounce 1.0 \
    --scroll-min-distance 10.0 \
    --scroll-max-frequency 5 \
    --scroll-session-timeout 3.0

# Less filtering (more scroll events logged)
crec --scroll-debounce 0.2 \
    --scroll-min-distance 2.0 \
    --scroll-max-frequency 20 \
    --scroll-session-timeout 1.0
```

**Scroll filtering parameters:**
- `--scroll-debounce`: Minimum time between scroll events (default: 0.5 seconds)
- `--scroll-min-distance`: Minimum scroll distance to log (default: 5.0 pixels)
- `--scroll-max-frequency`: Maximum scroll events per second (default: 10)
- `--scroll-session-timeout`: Scroll session timeout (default: 2.0 seconds)
