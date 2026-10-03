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

import difflib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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
        rps:                 Response Preservation Score (expected ≈ 1.0).
    """
    final_text:           str
    corrected_claims:     List[str] = field(default_factory=list)
    preserved_claims:     List[str] = field(default_factory=list)
    flagged_claims:       List[str] = field(default_factory=list)
    failed_corrections:   List[str] = field(default_factory=list)
    total_claims:         int       = 0
    correction_report:    str       = ""
    rps:                  float     = 1.0


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
        print(response.rps)
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
            ReconstructedResponse with final text, statistics, and RPS.
        """
        if not corrected_claims:
            logger.warning("No corrected claims to reconstruct -- returning empty response.")
            return ReconstructedResponse(
                final_text=original_response or "",
                total_claims=0,
                rps=1.0,
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
        rps = 1.0
        applied_replacements = []

        if original_response and claim_spans:
            final_text, applied_replacements = self._span_surgery(
                original_response, corrected_claims, claim_spans, annotate
            )
            rps = self.compute_rps(
                original=original_response,
                reconstructed=final_text,
                replacements=applied_replacements,
            )
        elif original_response and any(
            cc.status == CorrectionStatus.CORRECTED for cc in corrected_claims
        ):
            # Fallback: string-search surgery (no exact offsets)
            final_text = self._string_search_surgery(
                original_response, corrected_claims, annotate
            )
            rps = self.compute_rps(
                original=original_response,
                reconstructed=final_text,
            )
        else:
            # Legacy: join claim texts
            parts = self._collect_parts(corrected_claims, annotate)
            final_text = self._join_claims(parts)
            if original_response:
                rps = self.compute_rps(
                    original=original_response,
                    reconstructed=final_text,
                )
            else:
                rps = 1.0

        report = self._build_report(
            total=len(corrected_claims),
            preserved=preserved_claims,
            corrected=corrected_list,
            flagged=flagged_claims,
            failed=failed_corrections,
            original_response=original_response,
            rps=rps,
        )

        logger.info(
            f"Response reconstructed: {len(preserved_claims)} preserved, "
            f"{len(corrected_list)} corrected, "
            f"{len(flagged_claims)} flagged, "
            f"{len(failed_corrections)} failed. RPS={rps:.4f}"
        )

        return ReconstructedResponse(
            final_text=final_text,
            corrected_claims=corrected_list,
            preserved_claims=preserved_claims,
            flagged_claims=flagged_claims,
            failed_corrections=failed_corrections,
            total_claims=len(corrected_claims),
            correction_report=report,
            rps=rps,
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
    ) -> Tuple[str, List[Tuple[int, int, str]]]:
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
            Tuple of (modified_response_str, applied_replacements)
        """
        result = original

        # Build (start_char, end_char, replacement) triples for CORRECTED claims
        raw_replacements = []
        for cc, span in zip(corrected_claims, claim_spans):
            if cc.status != CorrectionStatus.CORRECTED:
                continue

            replacement = cc.corrected_claim.strip()
            if annotate:
                replacement = f"[CORRECTED: was '{cc.original_claim}'] {replacement}"

            orig_slice = original[span.start_char:span.end_char]

            # Preserve punctuation and formatting:
            # If the original span text ended with terminal punctuation, ensure replacement does too.
            if orig_slice.endswith((".", "?", "!")):
                if replacement and not replacement.endswith((".", "?", "!")):
                    replacement += orig_slice[-1]
            else:
                # If original span did NOT end with punctuation, and original text immediately
                # following span.end_char starts with punctuation, avoid duplicating punctuation.
                if span.end_char < len(original) and original[span.end_char] in (".", "?", "!", ",", ";", ":"):
                    replacement = replacement.rstrip(".?!,;:")

            raw_replacements.append((span.start_char, span.end_char, replacement))

        # Sort descending by start_char to preserve offset validity during substitution
        replacements_descending = sorted(raw_replacements, key=lambda x: x[0], reverse=True)

        applied = []
        for start, end, replacement in replacements_descending:
            if start < 0 or end > len(original) or start >= end:
                logger.warning(
                    f"Span [{start},{end}) out of range for original "
                    f"(len={len(original)}); skipping."
                )
                continue
            # Surgical span replacement: response[start:end] = replacement
            result = result[:start] + replacement + result[end:]
            applied.append((start, end, replacement))

        # Sort ascending by start_char for caller / RPS computation
        applied_ascending = sorted(applied, key=lambda x: x[0])
        return result, applied_ascending

    @staticmethod
    def compute_rps(
        original: str,
        reconstructed: str,
        corrected_spans: Optional[List[Tuple[int, int]]] = None,
        replacements: Optional[List[Tuple[int, int, str]]] = None,
    ) -> float:
        """
        Compute Response Preservation Score (RPS):

            RPS = (Unchanged Characters Outside Corrected Spans) /
                  (Total Characters Outside Corrected Spans)

        Expected: RPS ≈ 1.0 (approaching 1.0 indicates perfect preservation).

        Args:
            original:        The original LLM response text.
            reconstructed:   The reconstructed response text.
            corrected_spans: Optional list of (start_char, end_char) intervals for corrected claims.
            replacements:    Optional list of (start_char, end_char, replacement_text).

        Returns:
            RPS float in [0.0, 1.0].
        """
        if not original:
            return 1.0

        if corrected_spans is None:
            if replacements:
                corrected_spans = [(s, e) for s, e, _ in replacements]
            else:
                corrected_spans = []

        if not corrected_spans:
            if original == reconstructed:
                return 1.0
            matcher = difflib.SequenceMatcher(None, original, reconstructed)
            unchanged = sum(m.size for m in matcher.get_matching_blocks())
            return round(unchanged / len(original), 4)

        # Sort and merge any overlapping intervals
        sorted_spans = sorted(corrected_spans, key=lambda x: x[0])
        merged_spans = []
        for s, e in sorted_spans:
            s_clamp = max(0, min(s, len(original)))
            e_clamp = max(s_clamp, min(e, len(original)))
            if not merged_spans:
                merged_spans.append([s_clamp, e_clamp])
            else:
                if s_clamp <= merged_spans[-1][1]:
                    merged_spans[-1][1] = max(merged_spans[-1][1], e_clamp)
                else:
                    merged_spans.append([s_clamp, e_clamp])

        # Extract outside segments from original text
        orig_outside_segments = []
        last_end = 0
        for s, e in merged_spans:
            if s > last_end:
                orig_outside_segments.append(original[last_end:s])
            last_end = max(last_end, e)
        if last_end < len(original):
            orig_outside_segments.append(original[last_end:])

        orig_outside = "".join(orig_outside_segments)
        total_outside = len(orig_outside)
        if total_outside == 0:
            return 1.0

        # Extract outside segments from reconstructed text if exact replacements are provided
        if replacements:
            sorted_reps = sorted(replacements, key=lambda x: x[0])
            recon_outside_segments = []
            recon_pos = 0
            last_orig_end = 0
            for s, e, rep in sorted_reps:
                seg_len = max(0, s - last_orig_end)
                recon_outside_segments.append(reconstructed[recon_pos : recon_pos + seg_len])
                recon_pos += seg_len + len(rep)
                last_orig_end = e
            if last_orig_end < len(original):
                rem_len = len(original) - last_orig_end
                recon_outside_segments.append(reconstructed[recon_pos : recon_pos + rem_len])
            recon_outside = "".join(recon_outside_segments)

            if recon_outside == orig_outside:
                return 1.0

            matcher = difflib.SequenceMatcher(None, orig_outside, recon_outside)
            unchanged = sum(m.size for m in matcher.get_matching_blocks())
            return round(min(unchanged, total_outside) / total_outside, 4)
        else:
            matcher = difflib.SequenceMatcher(None, orig_outside, reconstructed)
            unchanged = sum(m.size for m in matcher.get_matching_blocks())
            return round(min(unchanged, total_outside) / total_outside, 4)

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
        rps: float = 1.0,
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
            f"  Preservation Score (RPS): {rps:.4f}",
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
