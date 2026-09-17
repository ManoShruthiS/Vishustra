"""Enable `python -m vishustra_core` to run the dashboard/CLI."""

from vishustra_core.main import main

if __name__ == "__main__":
    raise SystemExit(main())