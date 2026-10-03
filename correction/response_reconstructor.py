"""
response_reconstructor.py  (Phase 6)
======================================
Response Reconstruction Module
---------------------------------
Implements MINIMUM NECESSARY CORRECTION: the original LLM response
is preserved as closely as possible, with only the hallucinated
claim spans surgically replaced.

Primary algorithm:
  1. Start with the original response text.
  2. Collect all CORRECTED claims (sorted by start_char descending so
     that earlier replacements do not shift later offsets).
  3. For each CORRECTED claim, locate the original sentence span
     [start_char, end_char) in the current text and replace it with
     the corrected claim text.
  4. All SUPPORTED and UNVERIFIABLE claim text is left byte-for-byte
     unchanged.

Span-based replacement guarantees:
  - Correct order (no reordering of claims)
  - Correct scope (no surrounding text removed)
  - Safe handling of duplicate claim text (uses offsets, not search)
  - Exact preservation of unchanged passages

Fallback:
  If no original_response is provided (e.g., in unit tests), the
  reconstructor falls back to joining the corrected claim strings as
  a paragraph (the old behaviour).
"""

from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

from correction.claim_corrector import CorrectedClaim, CorrectionStatus


# ------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------

@dataclass
class ReconstructedResponse:
    """
    The final reconstructed response after selective correction.

    Attributes:
        final_text:          The assembled final response text.
        corrected_claims:    Claims that were changed.
        preserved_claims:    Claims that were kept as-is.
        flagged_claims:      Claims with insufficient evidence.
        failed_corrections:  Claims where correction failed.
        total_claims:        Total number of atomic claims.
        correction_report:   Human-readable summary of changes.
    """
    final_text:           str
    corrected_claims:     List[str] = field(default_factory=list)
    preserved_claims:     List[str] = field(default_factory=list)
    flagged_claims:       List[str] = field(default_factory=list)
    failed_corrections:   List[str] = field(default_factory=list)
    total_claims:         int       = 0
    correction_report:    str       = ""


# ------------------------------------------------------------------
# Response Reconstructor
# ------------------------------------------------------------------

class ResponseReconstructor:
    """
    Merges corrected claims into a coherent final response using
    span-level surgery on the original response text.

    Usage:
        reconstructor = ResponseReconstructor()
        response = reconstructor.reconstruct(
            corrected_claims=claims,
            claim_spans=spans,
            original_response=original_text,
        )
        print(response.final_text)
    """

    def reconstruct(
        self,
        corrected_claims: List[CorrectedClaim],
        claim_spans=None,        # List[ClaimSpan] | None
        annotate: bool = False,
        original_response: Optional[str] = None,
    ) -> ReconstructedResponse:
        """
        Assemble corrected claims into the final response.

        When original_response and claim_spans are provided, performs
        span-level surgery: only CORRECTED claim spans are replaced,
        all other text is preserved verbatim.

        When original_response is not provided, falls back to joining
        claim strings as a paragraph (legacy behaviour).

        Args:
            corrected_claims:  List of CorrectedClaim objects from Phase 5.
            claim_spans:       List of ClaimSpan objects from Phase 1
                               (must be parallel to corrected_claims).
            annotate:          If True, embed correction markers in text.
            original_response: Original LLM response for span surgery.

        Returns:
            ReconstructedResponse with final text and statistics.
        """
        if not corrected_claims:
            logger.warning("No corrected claims to reconstruct -- returning empty response.")
            return ReconstructedResponse(
                final_text=original_response or "",
                total_claims=0,
            )

        preserved_claims:   List[str] = []
        corrected_list:     List[str] = []
        flagged_claims:     List[str] = []
        failed_corrections: List[str] = []

        # Classify each claim
        for cc in corrected_claims:
            if cc.status == CorrectionStatus.PRESERVED:
                preserved_claims.append(cc.original_claim)
            elif cc.status == CorrectionStatus.CORRECTED:
                corrected_list.append(
                    f"'{cc.original_claim}' -> '{cc.corrected_claim}'"
                )
            elif cc.status == CorrectionStatus.FLAGGED:
                flagged_claims.append(cc.original_claim)
            else:
                # FAILED or FINAL_VERIFICATION_FAILED
                failed_corrections.append(cc.original_claim)

        # Choose reconstruction strategy
        if original_response and claim_spans:
            final_text = self._span_surgery(
                original_response, corrected_claims, claim_spans, annotate
            )
        elif original_response and any(
            cc.status == CorrectionStatus.CORRECTED for cc in corrected_claims
        ):
            # Fallback: string-search surgery (no exact offsets)
            final_text = self._string_search_surgery(
                original_response, corrected_claims, annotate
            )
        else:
            # Legacy: join claim texts
            parts = self._collect_parts(corrected_claims, annotate)
            final_text = self._join_claims(parts)

        report = self._build_report(
            total=len(corrected_claims),
            preserved=preserved_claims,
            corrected=corrected_list,
            flagged=flagged_claims,
            failed=failed_corrections,
            original_response=original_response,
        )

        logger.info(
            f"Response reconstructed: {len(preserved_claims)} preserved, "
            f"{len(corrected_list)} corrected, "
            f"{len(flagged_claims)} flagged, "
            f"{len(failed_corrections)} failed."
        )

        return ReconstructedResponse(
            final_text=final_text,
            corrected_claims=corrected_list,
            preserved_claims=preserved_claims,
            flagged_claims=flagged_claims,
            failed_corrections=failed_corrections,
            total_claims=len(corrected_claims),
            correction_report=report,
        )

    # ------------------------------------------------------------------
    # Span Surgery (primary path)
    # ------------------------------------------------------------------

    @staticmethod
    def _span_surgery(
        original: str,
        corrected_claims: List[CorrectedClaim],
        claim_spans,              # List[ClaimSpan]
        annotate: bool = False,
    ) -> str:
        """
        Replace ONLY the corrected claim spans in the original response.

        Processing order: descending by start_char so that replacements
        of earlier spans do not shift the offsets of later spans.

        Supported claims are left completely untouched.

        Args:
            original:         The original LLM response text.
            corrected_claims: Parallel list of CorrectedClaim objects.
            claim_spans:      Parallel list of ClaimSpan objects.
            annotate:         If True, wrap replacements with annotation.

        Returns:
            Modified response string.
        """
        result = original

        # Build (start_char, end_char, replacement) triples for CORRECTED claims
        replacements = []
        for cc, span in zip(corrected_claims, claim_spans):
            if cc.status != CorrectionStatus.CORRECTED:
                continue

            replacement = cc.corrected_claim.strip()
            if annotate:
                replacement = f"[CORRECTED: was '{cc.original_claim}'] {replacement}"

            # Ensure replacement ends with sentence termination
            if replacement and not replacement.endswith((".","?","!")):
                replacement += "."

            replacements.append((span.start_char, span.end_char, replacement))

        # Sort descending by start_char to preserve offset validity
        replacements.sort(key=lambda x: x[0], reverse=True)

        for start, end, replacement in replacements:
            # Validate offsets are still within range after previous edits
            # (offsets are relative to the ORIGINAL string, which we keep)
            if start < 0 or end > len(result) or start >= end:
                logger.warning(
                    f"Span [{start},{end}) out of range for result "
                    f"(len={len(result)}); skipping."
                )
                continue
            result = result[:start] + replacement + result[end:]

        return result

    @staticmethod
    def _string_search_surgery(
        original: str,
        corrected_claims: List[CorrectedClaim],
        annotate: bool = False,
    ) -> str:
        """
        Fallback surgery when exact offsets are not available.
        Uses string search to find and replace original claim text.
        """
        result = original

        # Process in reverse order by approximate position in text
        for cc in reversed(corrected_claims):
            if cc.status != CorrectionStatus.CORRECTED:
                continue

            original_span = cc.original_claim.strip()
            corrected_span = cc.corrected_claim.strip()
            if not original_span or original_span == corrected_span:
                continue

            replacement = corrected_span
            if annotate:
                replacement = f"[CORRECTED: was '{original_span}'] {corrected_span}"

            if original_span in result:
                result = result.replace(original_span, replacement, 1)
            else:
                # Case-insensitive fallback
                lower_result = result.lower()
                idx = lower_result.find(original_span.lower())
                if idx >= 0:
                    result = result[:idx] + replacement + result[idx + len(original_span):]
                else:
                    # Append as a correction note (do not silently drop)
                    result = result + f" [{replacement}]"

        return result

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_parts(
        corrected_claims: List[CorrectedClaim],
        annotate: bool,
    ) -> List[str]:
        """Build text parts list for legacy paragraph join."""
        parts = []
        for cc in corrected_claims:
            if cc.status == CorrectionStatus.PRESERVED:
                text = cc.corrected_claim
            elif cc.status == CorrectionStatus.CORRECTED:
                text = cc.corrected_claim
                if annotate:
                    text = f"[CORRECTED: was '{cc.original_claim}'] {text}"
            elif cc.status == CorrectionStatus.FLAGGED:
                text = cc.original_claim
                if annotate:
                    text = f"[UNVERIFIED] {text}"
            else:
                text = cc.original_claim
                if annotate:
                    text = f"[CORRECTION FAILED] {text}"
            parts.append(text)
        return parts

    @staticmethod
    def _join_claims(parts: List[str]) -> str:
        """Join atomic claim sentences into a coherent paragraph."""
        cleaned = []
        for p in parts:
            p = p.strip()
            if p and not p.endswith("."):
                p += "."
            if p:
                cleaned.append(p)
        return " ".join(cleaned)

    @staticmethod
    def _build_report(
        total: int,
        preserved: List[str],
        corrected: List[str],
        flagged: List[str],
        failed: List[str],
        original_response: Optional[str],
    ) -> str:
        """Build a human-readable correction report for transparency."""
        lines = [
            "=" * 60,
            "  CORRECTION REPORT",
            "=" * 60,
            f"  Total claims processed  : {total}",
            f"  SUPPORTED (preserved)   : {len(preserved)}",
            f"  CONTRADICTED (corrected): {len(corrected)}",
            f"  UNVERIFIABLE (flagged)  : {len(flagged)}",
            f"  Failed corrections      : {len(failed)}",
            "=" * 60,
        ]

        if corrected:
            lines.append("\n  CORRECTIONS MADE:")
            for i, c in enumerate(corrected, 1):
                lines.append(f"  {i}. {c}")

        if flagged:
            lines.append("\n  UNVERIFIABLE CLAIMS (insufficient evidence):")
            for i, f_item in enumerate(flagged, 1):
                lines.append(f"  {i}. {f_item}")

        if failed:
            lines.append("\n  FAILED CORRECTIONS (original retained):")
            for i, f_item in enumerate(failed, 1):
                lines.append(f"  {i}. {f_item}")

        lines.append("=" * 60)
        return "\n".join(lines)
