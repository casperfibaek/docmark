# LLM Formatter Refactoring Plan

## Problem Statement

The current LLM-based Markdown formatter (`enhance_markdown` in `docmark/core/llm.py`) experiences content loss (truncation) when processing long text, particularly when using the `o3-mini` model. This issue seems related to the chunking mechanism, specifically the `_ensure_complete_markdown_formatting` method which uses overlapping fixed-size chunks and `difflib` for stitching.

## Analysis of Current Implementation

- **Relevant Code:** `docmark/core/llm.py`, methods `enhance_markdown`, `_enhance_markdown_in_chunks`, `_process_single_chunk`, `_ensure_complete_markdown_formatting`.
- **Current Strategies:**
  - Semantic Chunking (`_enhance_markdown_in_chunks`): Splits by headers/paragraphs. Potentially fragile.
  - Overlapping Fixed-Size Chunking (`_ensure_complete_markdown_formatting`): Fallback/default for `o3-mini`. Uses fixed character size (1500) and overlap (400), stitching with `difflib`. Suspected source of content loss due to LLM reformatting in overlap zones or incorrect stitching.
- **Potential Issues:**
  - Token limits (`max_completion_tokens`) might be hit if formatted output exceeds the limit.
  - `difflib` stitching might incorrectly remove content.
  - Length-based verification might miss subtle content loss.
  - Fixed character chunking doesn't respect Markdown structure.

## Proposed Refactoring Plan

### Phase 1: Analysis & Design

1. **Deep Dive into `_ensure_complete_markdown_formatting`:** Re-evaluate chunk size, overlap, and `difflib` stitching logic specifically for `o3-mini`.
2. **Token Limit Analysis:** Analyze `max_completion_tokens` (currently 2000) vs. input chunk size (1500 chars) to ensure sufficient headroom for formatted output.
3. **Alternative Chunking Strategy:** Design structure-aware chunking based on logical Markdown elements (paragraphs, headings, lists, code blocks) using *token* counts (via `tiktoken`) instead of character counts.
4. **Simpler Stitching:** Design a simpler stitching mechanism (e.g., concatenation with newline checks) suitable for structure-aware chunks.
5. **Improved Verification:** Enhance `_process_single_chunk` verification to check for the presence of the first/last few non-whitespace characters/words of the original chunk in the output.
6. **Prompt Refinement:** Strengthen LLM prompts to emphasize absolute content preservation, especially near chunk ends.

### Phase 2: Refactoring & Implementation (Code Mode)

1. **Implement Token Counting:** Integrate `tiktoken`.
2. **Implement New Chunking Logic:** Replace/augment existing methods with structure-aware, token-based chunking.
3. **Implement New Stitching Logic:** Replace `difflib` stitching with the simpler approach.
4. **Implement Enhanced Verification:** Add start/end content checks.
5. **Adjust API Parameters:** Fine-tune `max_completion_tokens`.
6. **Update Prompts:** Modify LLM prompts.

### Phase 3: Testing (Code Mode)

1. **Unit Tests:** Test new chunking and stitching functions.
2. **Integration Test:** Test `enhance_markdown` with long documents known to cause issues, using `o3-mini` and the `OPENAI_API_KEY` environment variable.
3. **Content Integrity Check:** Verify output against original documents for content loss.

## Proposed Logic Flow

```mermaid
graph TD
    A[Start enhance_markdown] --> B{Is text long (check token count)?};
    B -- No --> C[Process Full Text with LLM];
    B -- Yes --> D[Chunk Text by Structure (paragraphs, lists, code blocks) respecting token limit];
    D --> E[Initialize Empty Result];
    D --> F{For each chunk};
    F -- Loop --> G[Process Chunk with LLM (using refined prompt & API params)];
    G --> H{Verification Passed? (check start/end content)};
    H -- Yes --> I[Append Formatted Chunk to Result (simple concatenation)];
    H -- No --> J[Append Original Chunk to Result & Log Warning];
    I --> K{More chunks?};
    J --> K;
    K -- Yes --> F;
    K -- No --> L[Return Combined Result];
    C --> L;
```

## API Key Handling

The implementation will rely on the `OPENAI_API_KEY` environment variable being set for testing with the OpenAI provider. The `.keys` file will not be read directly.
