import re
from typing import Optional
from enum import Enum

line_number_pattern = re.compile(r"(?m)^@@ -(\d*?),(\d*?) \+(\d*?),(\d*?) @@")
diff_pattern = re.compile(r"""<NME> (?P<name>.*?)
<BEF> (?P<file>(.|\n)*?)
<MSG> (?P<message>(.|\n)*?)
<DFF> (?P<diff>(.|\n)*)""")
hunk_split_pattern = re.compile(r"(?m)^(@@ .*? @@).*\n")


class DiffState(Enum):
    """
    An Enum keeping track of the validity of the diff data. It is the return of the helper function `verify_diff`.
    Binary codes help internally, as some errors are additive (e.g., can have both invalid text and invalid line num).
    But we convert the binary codes into Enum for better readability.
    """
    VALID = 0b000  # valid diff

    # The following are errors that can still be either ignored or fixed.
    INVALID_TEXT = 0b001  # pre-diff texts cannot be found in the context.
    INVALID_LINE_NUM = 0b010  # the numbers in @@ -x,y +a,b @@ are invalid (but can be parsed).
    INVALID_TEXT_AND_LINE_NUM = 0b011  # both 0b001 and 0b010.

    # The following are format errors that cannot be ignored.
    BAD_FORMAT = 0b100  # cannot be parsed according to <NME> ...\n<BEF> ...\n<MSG> ...\n<DFF> @@ ... @@\n...
    BAD_DIFF_HUNK_FORMAT = 0b101  # diff hunk contains lines whose initial character is not one of ' ', '+', '-'
    BAD_LINE_NUM_FORMAT = 0b110  # the @@ ... @@ bracket can be found but numbers cannot be parsed.
    BAD_HUNK_AND_LINE_FORMAT = 0b111  # both 0b110 and 0b101.


def split_diff(content: str) -> dict:
    """
    Args:
        content: the diff content.

    Returns:
        A dict with potentially 4 items:
            name: the filename
            file: the file content
            message: the diff message
            diff: the diff hunk
        Any key could be missing. That would mean a failure in matching.
    """
    match = diff_pattern.match(content)
    return {} if match is None else match.groupdict()


def parse_line_info(content: str) -> tuple:
    """
    Parse @@ -x,y +a,b @@

    Args:
        the @@ ... @@ line
    Returns:
        (x, y, a, b) as integers
    """
    match = line_number_pattern.match(content)
    if match is None:
        return ()
    match = match.groups()
    if len(match) >= 4:
        # shouldn't be more than 4, but in case of anything weird, we take the first 4 matching elements.
        return tuple([int(num) for num in match][:4])
    else:
        # incorrect format => return nothing
        return ()


def parse_diff_content(hunk: str, separate_lines=False, reject_invalid=False) -> Optional[tuple]:
    """
    Parse a diff content to turn it into (before_diff, after_diff) based on '+', '-' at the beginning of each line.

    Args:
        hunk: the diff content (without "@@ ... @@").
        separate_lines: (Optional) True if return list of lines.
        reject_invalid: (Optional) True if return None for invalid diff hunk (non-empty lines without starting
            with ' ', '-', '+')
    Returns:
        (before_diff, after_diff);
        None if reject_invalid==True and the diff hunk contains invalid format.
    """
    # Remove trailing \n at the beginning and the end.
    hunk = hunk.split('\n')
    before_diff, after_diff = [], []
    for line in hunk:
        if not line:
            continue
        if line[0] == '-' or line[0] == ' ':
            before_diff.append(line[1:])
        if line[0] == '+' or line[0] == ' ':
            after_diff.append(line[1:])
        if reject_invalid:
            if all([line[0] != c for c in [' ', '-', '+']]):
                return None
    if separate_lines:
        return before_diff, after_diff
    else:
        return '\n'.join(before_diff), '\n'.join(after_diff)


def replace_text(text: str, before: str, after: str, start_pointer: int) -> tuple[str, int]:
    """
    Try to match `before` within `text` and replace the content into `after`.
    If not found, return the original text.

    Args:
        text: the original text.
        before: the text to be matched.
        after: the text to be replaced into.
        start_pointer: the index where we start to match (inclusive).
    Returns:
        (diff_result, new_start_pointer)
        the text after the match-and-replace and the new index at the end of the change.
    """
    idx = text.find(before)
    if idx < start_pointer:
        return text, start_pointer
    else:
        # Even if idx + len(before) is out-of-bound, the list slicing would return []
        return text[:idx] + after + text[idx + len(before):], idx + len(after)


def apply_diff(file: str, diff: str, use_line_number=False, allow_add_file=True) -> str:
    """
    Apply the diff to the file content. We try to be lenient and keep applying the patch naively until we cannot.
    (Note: use_line_number=True is somehow slightly slower.)
    (Warning: if use_line_number==False, we could have some problematic cases like, if all lines in diff hunk
        starts with "+", the pre-diff paragraphs relevant to the hunk is empty. Because we only use pre-diff
        paragraphs to match, we would simply match the very beginning.)
    Args:
        file: the file content.
        diff: the diff hunk (containing "@@ -x,y +a,b @@").
        use_line_number: (Optional) use the line numbers in "@@ ... @@" faithfully.
        allow_add_file: (Optional) when file is "ADDFILE" (meaning <BEF> ADDFILE\n... showed up in the diff text),
            we automatically patch the diff by a direct replacement.
    Return:
        the maximally patched file content.
    """
    diff = hunk_split_pattern.split(diff.lstrip().lstrip("\n"))

    # If we use the line numbers, we match-and-replace in a line-by-line fashion.
    file_by_line = file.split('\n') if use_line_number else None
    line_offset = 0  # the offset between pre-/post-patching line numbers

    # If we do not use the line numbers, for multiple diff hunk, we only move forward in a greedy manner.
    patch_pointer = 0

    i = 0 if diff[0] else 1  # We have delimiter at the beginning, causing empty initial string
    while i < len(diff) - 1:  # Need at least a pair of '@@ ... @@' and diff hunk to continue
        # Expect a string with '@@ ... @@' followed by a diff hunk
        line_info = parse_line_info(diff[i])
        diff_content = diff[i + 1]
        i += 2

        # Generate the pre-/post-diff string based on the first character being '+' or '-'
        # (Note: parse_diff_content will strip trailing \n at the beginning and the end)
        parsed_diff = parse_diff_content(diff_content, separate_lines=use_line_number)

        # If we allow the recognition of "ADDFILE" and encounter such file, special treatment is needed.
        if allow_add_file and file == "ADDFILE":
            if use_line_number:
                return parsed_diff[1] if line_info == (0, 0) else ""
            else:
                return parsed_diff[1]

        if use_line_number:
            # If line numbers cannot be parsed, skip.
            if not line_info:
                continue

            # Offset the starting line
            start_idx = line_info[0] + line_offset

            # Match the referred lines with the file context
            referred_lines = file_by_line[start_idx - 1 : start_idx - 1 + line_info[1]]
            valid = all([l1 == l2 for l1, l2 in zip(parsed_diff[0], referred_lines)])

            # If lines fully match and the number of lines is consistent, apply the patch.
            # We ignore the second pair "+a, b" just to be lenient.
            if valid and len(parsed_diff[0]) == line_info[1]:
                # Update the list of lines
                file_by_line = file_by_line[: start_idx - 1] + parsed_diff[1] + \
                               file_by_line[start_idx - 1 + line_info[1] :]
                line_offset += len(parsed_diff[1]) - line_info[1]
        else:
            # Directly (and naively) apply patch by match-and-replace.
            file, patch_pointer = replace_text(file, parsed_diff[0], parsed_diff[1], patch_pointer)

    if use_line_number:
        file = "\n".join(file_by_line)
    return file


def verify_diff(diff_text: str) -> DiffState:
    """
    Verify the validity of a complete diff text.

    Args:
        diff_text: the complete diff text.
            The overall format conforms "<NME> ...\n<BEF> ...\n<MSG> ...\n<DFF> ..." and the text
            after <DFF> has 1 or more lines of "@@ -x,y +a,b @@" followed by the corresponding hunk.
    Returns:
        A DiffState (see above).
    """
    diff_dict = split_diff(diff_text)
    line_offset = 0

    keys = ["name", "file", "message", "diff"]
    for key in keys:
        if key not in diff_dict:
            return DiffState(0b100)  # Invalid overall format

    diff_parts = hunk_split_pattern.split(diff_dict["diff"].lstrip())
    if not diff_parts:
        return DiffState(0b100)  # Invalid overall format

    context_mismatch, line_number_mismatch = False, False
    bad_diff_hunk, bad_line_number = False, False

    i = 0 if diff_parts[0] else 1
    while i < len(diff_parts) - 1:  # Need at least a pair of '@@ ... @@' and diff hunk to continue
        line_info = parse_line_info(diff_parts[i])
        diff_content = parse_diff_content(diff_parts[i + 1], reject_invalid=True)
        i += 2

        # Special treatment if we are adding a new file
        if diff_dict["file"] == "ADDFILE":
            if len(diff_parts) != i or not line_info or line_info[:3] != (0, 0, 0) or \
                    line_info[3] != len(diff_content[1].strip("\n").split("\n")) or diff_content[0]:
                return DiffState(0b110)
            else:
                return DiffState(0b000)

        if not line_info or len(line_info) != 4:
            bad_line_number = True
        if diff_content is None:
            bad_diff_hunk = True

        # Skip the diff matching checks if bad format already occurred
        if bad_diff_hunk or bad_line_number:
            continue

        # Try to see if there is a match in the file context. Must match complete lines or till EOF.
        match_idx = diff_dict["file"].find(diff_content[0])
        if match_idx == -1 or (match_idx + len(diff_content[0]) != len(diff_dict["file"]) and
                               diff_dict["file"][match_idx + len(diff_content[0])] != "\n"):
            context_mismatch = True

        if line_info[0] <= 0:
            # -0,0 only happens when we create a new file (in which case the context is <BEF> ADDFILE\n...).
            if line_info[1] != 0 or diff_dict["file"] != "ADDFILE":
                line_number_mismatch = True
        else:
            # Check the line numbers regardless of whether the context matches.
            pre_diff_line_number = len(diff_content[0].strip("\n").split("\n"))
            post_diff_line_number = len(diff_content[1].strip("\n").split("\n"))
            if (pre_diff_line_number, post_diff_line_number) != (line_info[1], line_info[3]):
                line_number_mismatch = True
            else:
                line_offset += len(diff_content[1]) - line_info[1]

    if bad_diff_hunk or bad_line_number:
        return DiffState(bad_diff_hunk * 0b001 + bad_line_number * 0b010 + 0b100)
    else:
        return DiffState(context_mismatch * 0b001 + line_number_mismatch * 0b010)
