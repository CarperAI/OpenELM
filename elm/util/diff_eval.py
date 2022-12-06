import re


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
    pattern = r"<NME> (?P<name>.*?)\n<BEF> (?P<file>(.|\n)*?)\n<MSG> (?P<message>(.|\n)*?)\n<DFF> (?P<diff>(.|\n)*)"
    return re.match(pattern, content).groupdict()


def parse_line_info(content: str) -> tuple:
    """
    Parse @@ -x,y +a,b @@
    Args:
        the @@ ... @@ line
    Returns:
        (x, y, a, b) as integers
    """
    pattern = r"@@ -(\d*?),(\d*?) \+(\d*?),(\d*?) @@"
    match = re.match(pattern, content).groups()
    if len(match) >= 4:
        # shouldn't be more than 4, but in case of anything weird, we take the first 4 matching elements.
        return tuple([int(num) for num in match][:4])
    else:
        # incorrect format => return nothing
        return ()


def parse_diff_content(hunk: str) -> tuple:
    """
    Parse a diff content to turn it into (before_diff, after_diff) based on '+', '-' at the beginning of each line.
    Args:
        the diff content (without "@@ ... @@").
    Returns:
        (before_diff, after_diff)
    """
    hunk = hunk.split('\n')
    before_diff, after_diff = [], []
    for line in hunk:
        # We will be lenient and skip invalid lines whose leading characters are not '+', '-' or ' '.
        if not line:
            continue
        if line[0] == '-' or line[0] == ' ':
            before_diff.append(line[1:])
        if line[0] == '+' or line[0] == ' ':
            after_diff.append(line[1:])
    return '\n'.join(before_diff), '\n'.join(after_diff)


def replace_text(text: str, before: str, after: str) -> str:
    """
    Try to match `before` within `text` and replace the content into `after`.
    If not found, return the original text.
    Args:
        text: the original text.
        before: the text to be matched.
        after: the text to be replaced into.
    Returns:
        the text after the match-and-replace.
    """
    idx = text.find(before)
    if idx == -1:
        return text
    else:
        if idx + len(before) >= len(text):
            return text[:idx] + after
        else:
            return text[:idx] + after + text[idx + len(before):]


def apply_diff(file: str, diff: str, use_line_number=False) -> str:
    """
    Apply the diff to the file content. We try to be lenient here and keep applying the patch naively until we cannot.
    Args:
        file: the file content.
        diff: the diff hunk (containing "@@ -x,y +a,b @@").
        use_line_number: (Optional) use the line numbers in "@@ ... @@" faithfully.
    Return:
        the maximally patched file content.
    """
    diff = re.split('(@@ .*? @@).*\n', diff.lstrip())

    if use_line_number:
        # If we use the line numbers, we match-and-replace in a line-by-line fashion.
        file_by_line = file.split('\n')

    i = 0
    while i < len(diff) - 1:  # Need at least a pair of '@@ ... @@' and diff hunk to continue
        if not diff[i]:  # Skip empty match (should only happen at the beginning)
            i += 1
            continue

        # Expect a string with '@@ ... @@' followed by a diff hunk
        line_info = parse_line_info(diff[i])
        diff_content = diff[i + 1]
        i += 2

        # Generate the diff text before and after the patch based on the first character being '+' or '-'
        parsed_diff = parse_diff_content(diff_content)

        if use_line_number:
            # If line numbers cannot be parsed, skip.
            if not line_info:
                continue
            # Validate that the corresponding lines match (not the most efficient implementation though)
            diffed_lines = "\n".join(file_by_line[line_info[0] - 1 : line_info[0] + line_info[1] - 1])
            if diffed_lines != parsed_diff[0].rstrip("\n"):
                continue

        # Directly (and naively) apply patch by match-and-replace.
        file = replace_text(file, *parsed_diff)  # not the most efficient if use_line_number

    return file
