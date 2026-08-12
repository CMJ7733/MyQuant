"""
Package management utilities for Famou 2.0.

Handles automatic installation of Python packages required by generated programs.
"""

import os
import subprocess
import sys
from typing import List, Optional, Set


# Global cache of installed packages (shared across all environments)
_INSTALLED_PACKAGES: Set[str] = set()


def install_packages(
    packages: List[str],
    quiet: bool = False,
) -> bool:
    """
    Install Python packages to a target directory using pip.

    Packages are installed to an experiment-specific directory to avoid
    polluting the user's Python environment. The target directory is
    added to PYTHONPATH during execution.

    Uses sys.executable to ensure packages are installed using the
    current Python interpreter (conda, uv, pyenv, venv, etc.).

    Args:
        packages: List of package names to install (e.g., ["numpy", "matplotlib"])
        timeout: Installation timeout in seconds (default: 120)
        quiet: Suppress pip output (default: True)

    Returns:
        True if installation succeeded, False otherwise
    """
    if not packages:
        return True

    # Filter out already installed packages
    global _INSTALLED_PACKAGES
    to_install = [p for p in packages if p not in _INSTALLED_PACKAGES]

    if not to_install:
        return True

    try:
        # Build pip command
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
        ]

        if quiet:
            cmd.append("--quiet")

        cmd.extend(to_install)

        # Run pip install
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            _INSTALLED_PACKAGES.update(to_install)
            return True
        else:
            # Installation failed - log but don't raise
            # Let program execution handle missing dependencies
            print(f"Warning: Package installation failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"Warning: Package installation error: {e}")
        return False

def clear_package_cache():
    """
    Clear the global package installation cache.

    Forces re-installation of all packages on next call to install_packages().

    Useful for testing or when package versions need to be refreshed.

    Examples:
        >>> clear_package_cache()
        >>> install_packages(["numpy"], "./site-packages")  # Will reinstall
    """
    global _INSTALLED_PACKAGES
    _INSTALLED_PACKAGES.clear()
