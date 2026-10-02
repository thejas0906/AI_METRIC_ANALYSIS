"""
response_reconstructor.py  (Phase 6)
======================================
Response Reconstruction Module
---------------------------------
Merges corrected and preserved claims into a coherent final response.

Takes the list of CorrectedClaim objects (output of Phase 5) and
assembles them back into a readable paragraph or structured text,
distinguishing between:

  ✅ PRESERVED  → claim retained verbatim
  ✏️  CORRECTED  → hallucinated claim replaced with corrected version
  ⚠️  FLAGGED    → claim retained with optional inline annotation
  ❌ FAILED     → claim retained with warning annotation

The reconstructor also generates a structured correction report
summarizing what changed and why — useful for research transparency.

Design Decisions:
-----------------
- We join claims as a paragraph (no bullet points) to produce
  natural-sounding text similar to the original LLM output.
- Annotation mode can be toggled via `annotate=True` to embed
  correction metadata directly into the text (useful for debugging).
- The reconstruction order follows the original claim order to
  maintain narrative coherence.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

from correction.claim_corrector import CorrectedClaim, CorrectionStatus


# ──────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────

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
    corrected_claims:     List[str]    = field(default_factory=list)
    preserved_claims:     List[str]    = field(default_factory=list)
    flagged_claims:       List[str]    = field(default_factory=list)
    failed_corrections:   List[str]    = field(default_factory=list)
    total_claims:         int          = 0
    correction_report:    str          = ""


# ──────────────────────────────────────────────────────────────────
# Response Reconstructor
# ──────────────────────────────────────────────────────────────────

class ResponseReconstructor:
    """
    Merges corrected claims into a coherent final response.

    Usage:
        reconstructor = ResponseReconstructor()
        response = reconstructor.reconstruct(corrected_claims)
        print(response.final_text)
        print(response.correction_report)
    """

    def reconstruct(
        self,
        corrected_claims: List[CorrectedClaim],
        annotate: bool = False,
        original_response: Optional[str] = None,
    ) -> ReconstructedResponse:
        """
        Assemble corrected claims into the final response.

        Strategy (Minimum Necessary Correction):
        - If original_response is provided and corrections exist, use
          span-level surgery: locate the original claim span inside the
          original response and replace ONLY that span with the corrected
          text.  All other text (transitions, punctuation, structure) is
          preserved verbatim.
        - If original_response is not provided, fall back to joining the
          corrected/preserved claim strings as a paragraph.

        Args:
            corrected_claims:  List of CorrectedClaim objects from Phase 5.
            annotate:          If True, embed correction annotations in text.
            original_response: Original LLM response for span-surgery.

        Returns:
            ReconstructedResponse with final text and statistics.
        """
        if not corrected_claims:
            logger.warning("No corrected claims to reconstruct — returning empty response.")
            return ReconstructedResponse(
                final_text="",
                total_claims=0,
            )

        final_parts:        List[str] = []
        preserved_claims:   List[str] = []
        corrected_list:     List[str] = []
        flagged_claims:     List[str] = []
        failed_corrections: List[str] = []

        for cc in corrected_claims:
            status = cc.status

            if status == CorrectionStatus.PRESERVED:
                # Use the original claim text (unchanged)
                text = cc.corrected_claim   # same as original for PRESERVED
                preserved_claims.append(cc.original_claim)

            elif status == CorrectionStatus.CORRECTED:
                # Use the corrected (rewritten) claim
                text = cc.corrected_claim
                corrected_list.append(
                    f"'{cc.original_claim}' → '{cc.corrected_claim}'"
                )
                if annotate:
                    text = f"[CORRECTED: was '{cc.original_claim}'] {text}"

            elif status == CorrectionStatus.FLAGGED:
                # Keep original but annotate if requested
                text = cc.original_claim
                flagged_claims.append(cc.original_claim)
                if annotate:
                    text = f"[UNVERIFIED] {text}"

            else:  # FAILED
                text = cc.original_claim   # revert to original
                failed_corrections.append(cc.original_claim)
                if annotate:
                    text = f"[CORRECTION FAILED] {text}"

            final_parts.append(text)

        # Minimum Necessary Correction: prefer span-level surgery over claim-join
        if original_response and any(
            cc.status == CorrectionStatus.CORRECTED for cc in corrected_claims
        ):
            final_text = self._apply_span_surgery(
                original_response, corrected_claims, annotate
            )
        else:
            final_text = self._join_claims(final_parts)

        # Build correction report
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

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_span_surgery(
        original_response: str,
        corrected_claims: List[CorrectedClaim],
        annotate: bool = False,
    ) -> str:
        """
        Minimum Necessary Correction: surgically replace only the claim
        spans that were corrected, preserving all other text verbatim.

        Algorithm:
        1. Start with the full original response text.
        2. For each CORRECTED claim (processed in reverse order to keep
           character offsets valid after earlier substitutions), find the
           original claim string in the current text using exact string
           search, then simple fuzzy fallback.
        3. Replace the matched span with the corrected text.
        4. Claims with status PRESERVED, FLAGGED, or FAILED are left
           untouched (the whole point of minimum-necessary correction).

        This preserves paragraph structure, transitions, punctuation,
        and the overall wording of the original LLM response.

        Args:
            original_response: The original LLM response text.
            corrected_claims:  List of CorrectedClaim objects.
            annotate:          If True, wrap replacements with annotation.

        Returns:
            Modified response with only hallucinated spans replaced.
        """
        result = original_response

        # Process corrections in reverse so that earlier replacements
        # do not shift the character positions of later ones.
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

            # Strategy 1: exact substring match
            if original_span in result:
                result = result.replace(original_span, replacement, 1)
                continue

            # Strategy 2: case-insensitive match
            lower_result = result.lower()
            lower_span   = original_span.lower()
            idx = lower_result.find(lower_span)
            if idx >= 0:
                result = result[:idx] + replacement + result[idx + len(original_span):]
                continue

            # Strategy 3: partial match on the first 60 characters of the claim
            # (handles cases where punctuation was normalised during extraction)
            partial = original_span[:60].strip()
            if len(partial) > 20 and partial in result:
                end_idx = result.find(partial) + len(partial)
                # Extend to the end of the sentence
                for terminator in ['. ', '? ', '! ', '
']:
                    t_idx = result.find(terminator, end_idx)
                    if 0 < t_idx - end_idx < 80:
                        end_idx = t_idx + 1
                        break
                result = result[:result.find(partial)] + replacement + result[end_idx:]
                continue

            # Strategy 4: fallback — append corrected text in brackets
            # (avoid silently dropping the correction)
            result = result + f" [{replacement}]"

        return result

    @staticmethod
    def _join_claims(parts: List[str]) -> str:
        """
        Join atomic claim sentences into a coherent paragraph.

        Each claim ends with a period (ensured by ClaimExtractor).
        We simply join them with a space to form natural paragraphs.

        Args:
            parts: List of claim strings.

        Returns:
            Paragraph string.
        """
        # Clean up each part and join with spaces
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
        """
        Build a human-readable correction report for transparency.

        Args:
            total:             Total number of claims processed.
            preserved:         List of preserved claim strings.
            corrected:         List of "original → corrected" strings.
            flagged:           List of flagged (insufficient evidence) claims.
            failed:            List of claims where correction failed.
            original_response: Original LLM answer for comparison.

        Returns:
            Formatted correction report string.
        """
        lines = [
            "=" * 60,
            "  CORRECTION REPORT",
            "=" * 60,
            f"  Total claims processed : {total}",
            f"  ✅ Preserved (supported): {len(preserved)}",
            f"  ✏️  Corrected (halluc.)  : {len(corrected)}",
            f"  ⚠️  Flagged (insuff. ev.): {len(flagged)}",
            f"  ❌ Failed corrections   : {len(failed)}",
            "=" * 60,
        ]

        if corrected:
            lines.append("\n  CORRECTIONS MADE:")
            for i, c in enumerate(corrected, 1):
                lines.append(f"  {i}. {c}")

        if flagged:
            lines.append("\n  FLAGGED CLAIMS (insufficient evidence):")
            for i, f in enumerate(flagged, 1):
                lines.append(f"  {i}. {f}")

        if failed:
            lines.append("\n  FAILED CORRECTIONS (original retained):")
            for i, f in enumerate(failed, 1):
                lines.append(f"  {i}. {f}")

        lines.append("=" * 60)
        return "\n".join(lines)
