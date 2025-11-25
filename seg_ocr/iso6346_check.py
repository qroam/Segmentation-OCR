#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iso6346_check.py — Compute and validate ISO 6346 container numbers.

Usage:
  # Compute check digit from first 10 chars (4 letters + 6 digits)
  python iso6346_check.py compute MSCU123456
  # → MSCU1234567

  # Validate a full container number (11 chars, last is check digit)
  python iso6346_check.py validate MSCU1234567
  # → VALID (expected 7, got 7)

  # Auto mode: give either 10 or 11 chars
  python iso6346_check.py auto MS CU-123456-7
  # → VALID (expected 7, got 7)

The script is tolerant of spaces and hyphens; case-insensitive.
Implements ISO 6346 check digit: weighted sum with 2^position, mod 11 (10 → 0).
"""

from __future__ import annotations
import argparse
import re
from typing import Tuple

# Mapping per ISO 6346: A=10, B=12, C=13, D=14, E=15, F=16, G=17, H=18,
# I=19, J=20, K=21, L=23, M=24, N=25, O=26, P=27, Q=28, R=29, S=30, T=31,
# U=32, V=34, W=35, X=36, Y=37, Z=38  (11,22,33 skipped)
LETTER_VALUES = {
    **{chr(ord('A') + i): 10 + i for i in range(26)}
}
# Remove the gaps 11, 22, 33 by bumping letters after L and V by +1 respectively
# But it's simpler and clearer to define the exact mapping directly:
LETTER_VALUES = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18,
    'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27,
    'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36,
    'Y': 37, 'Z': 38
}

def normalize(code: str) -> str:
    """Keep only alphanumerics, uppercase, strip spaces/hyphens/etc."""
    return re.sub(r'[^A-Za-z0-9]', '', code).upper()

def char_value(ch: str) -> int:
    """Map a single character to its numeric value per ISO 6346."""
    if ch.isdigit():
        return int(ch)
    if ch in LETTER_VALUES:
        return LETTER_VALUES[ch]
    raise ValueError(f"Invalid character '{ch}' in code. Must be A–Z or 0–9.")

def compute_check_digit(first10: str) -> int:
    """
    Compute ISO 6346 check digit from the first 10 characters.
    first10: 4 letters + 6 digits after normalization.
    Returns: check digit (0–9)
    """
    s = normalize(first10)
    if len(s) != 10:
        raise ValueError(f"Need exactly 10 alphanumerics (got {len(s)}): '{first10}' -> '{s}'")
    # Weighted sum with weights 2^position, position starting at 0 from the left.
    total = 0
    for pos, ch in enumerate(s):
        total += char_value(ch) * (2 ** pos)
    remainder = total % 11
    return 0 if remainder == 10 else remainder

def validate_full(code11: str) -> Tuple[bool, int, int]:
    """
    Validate an 11-char container number.
    Returns (is_valid, expected_check, given_check)
    """
    s = normalize(code11)
    if len(s) != 11:
        raise ValueError(f"Need exactly 11 alphanumerics (got {len(s)}): '{code11}' -> '{s}'")
    expected = compute_check_digit(s[:10])
    given = int(s[-1]) if s[-1].isdigit() else -1
    return (expected == given, expected, given)

def main():
    ap = argparse.ArgumentParser(description="ISO 6346 container check digit tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_compute = sub.add_parser("compute", help="Compute check digit from first 10 chars")
    ap_compute.add_argument("code10", help="First 10 chars (4 letters + 6 digits)")

    ap_validate = sub.add_parser("validate", help="Validate a full 11-char container number")
    ap_validate.add_argument("code11", help="Full 11 chars (with check digit)")

    ap_auto = sub.add_parser("auto", help="Auto-detect: compute (10) or validate (11)")
    ap_auto.add_argument("code", help="Either 10 or 11 alphanumerics; spaces/hyphens OK")

    args = ap.parse_args()

    try:
        if args.cmd == "compute":
            cd = compute_check_digit(args.code10)
            print(normalize(args.code10) + str(cd))
        elif args.cmd == "validate":
            ok, exp, got = validate_full(args.code11)
            s = normalize(args.code11)
            print(f"{s}: {'VALID' if ok else 'INVALID'} (expected {exp}, got {got})")
        elif args.cmd == "auto":
            s = normalize(args.code)
            if len(s) == 10:
                cd = compute_check_digit(s)
                print(s + str(cd))
            elif len(s) == 11:
                ok, exp, got = validate_full(s)
                print(f"{s}: {'VALID' if ok else 'INVALID'} (expected {exp}, got {got})")
            else:
                raise ValueError(f"After normalization you must have 10 or 11 alphanumerics, got {len(s)}: '{s}'")
        else:
            ap.print_help()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
