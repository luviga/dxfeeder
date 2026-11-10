"""Command line entry point of dxfeeder."""

import argparse
import sys


def main(argv=None):
    """Parse the command line, run the chosen subcommand, and return its exit code."""
    parser = argparse.ArgumentParser(prog="dxfeeder", description="Synthetic OpenDSS distribution feeders from open data.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="run the tests and the package checks")
    arguments = parser.parse_args(argv)
    if arguments.command == "check":
        from dxfeeder import check
        return check.main()
    return 2


if __name__ == "__main__":
    sys.exit(main())
