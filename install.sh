#!/bin/sh
# NOVA-EVO installer for Linux and macOS: finds Python 3.10+ and hands over to install.py.
#   ./install.sh            ./install.sh --cpu            ./install.sh --tests
cd "$(dirname "$0")" || exit 1
for py in python3 python python3.13 python3.12 python3.11 python3.10; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        exec "$py" install.py "$@"
    fi
done
echo "Python 3.10 or newer was not found."
echo "  Debian/Ubuntu:  sudo apt install python3 python3-venv"
echo "  Fedora:         sudo dnf install python3"
echo "  macOS:          brew install python   (or https://www.python.org/downloads/)"
exit 1
