import argparse
import sys
import urllib.request
from pprint import pprint

from unidiff import PatchSet


def process_ind_patch(patch_diff) -> dict:
    """Process patch to get diff data."""
    patch_parsed_diff: dict = {
        "src_file": [],
        "tgt_file": [],
        "hunks": [],
        "addition_count": [],
        "deletion_count": [],
    }

    patch_parsed_diff["addition_count"] = patch_diff.added
    patch_parsed_diff["src_file"] = patch_diff.source_file
    patch_parsed_diff["tgt_file"] = patch_diff.target_file
    patch_parsed_diff["patch_info"] = patch_diff.patch_info
    patch_parsed_diff["deletion_count"] = patch_diff.removed
    patch_diff_list = list(patch_diff)
    for patch_diff_ind in patch_diff_list:
        patch_diff_ind = str(patch_diff_ind)
        patch_diff_split = patch_diff_ind.split("@@")
        patch_diff_line = patch_diff_split[2].split("\n")
        patch_diff_line_numbers = [
            list(map(int, hunk.strip("-+").split(",")))
            for hunk in patch_diff_split[1].strip().split(" ")
        ]
        patch_parsed_diff["hunks"].append(
            patch_diff_line_numbers + patch_diff_line[:-1]
        )
    return patch_parsed_diff


def patch_parse(patch_url: str) -> list:
    """Parse a patch to get diff data."""
    diff_list: list = []
    if ".diff" not in patch_url:
        patch_url = patch_url + ".diff"
    diff = urllib.request.urlopen(patch_url)
    encoding = diff.headers.get_charsets()[0]
    patch = PatchSet(diff, encoding=encoding)
    for patch_ind in patch:
        # Skip if the file is not a python file.
        if not patch_ind.target_file.endswith(".py"):
            continue
        patch_parsed = process_ind_patch(patch_ind)
        diff_list.append(patch_parsed)
    return diff_list


def apply_reverse_patch(
    diff_list: list, repo_data: tuple, length_threshold: int = 4096
) -> list:
    """Apply reverse patch to get before files. Returns list of modified files."""
    files_list: list = []
    for diff in diff_list:
        # Get raw after file.
        file_raw_url = (
            f"https://raw.githubusercontent.com/{repo_data[0]}/"
            f"{repo_data[1]}/{repo_data[2]}/{diff['tgt_file'][2:]}"
        )
        raw_file = urllib.request.urlopen(file_raw_url)
        raw_file_encoding = raw_file.headers.get_charsets()[0]
        raw_file = [line.decode(raw_file_encoding) for line in raw_file.readlines()]
        # file_length = sum(1 for _ in raw_file)
        # if file_length < length_threshold:
        files_list.append(raw_file)
        # Iterate over hunks for this file and apply the reverse patch.
        for hunk in diff_list[0]["hunks"]:
            hunk_list = []
            for line in hunk[3:]:
                if line.startswith("-") or line.startswith(" "):
                    hunk_list.append(line[1:] + "\n")
            files_list[0][hunk[0][0] : hunk[0][0] + hunk[0][1]] = hunk_list

    return files_list


def process_commit(commit_url: str, commit_message: str) -> dict:
    """Process a commit url to get the before files and diff dict."""
    # Get dict containing diff data.
    diff_list = patch_parse(commit_url)
    patch_url_split = commit_url.split("/")
    # author, repo name, commit hash
    repo_data = (patch_url_split[3], patch_url_split[4], patch_url_split[6])
    # Get list of files, each of which is a list of strings, one for each line.
    files_list = apply_reverse_patch(diff_list, repo_data)
    data_dict = {
        "before_files": files_list,
        "commit_message": commit_message,
        "diff": diff_list,
    }
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument(
        "--commit_url",
        type=str,
        default="https://github.com/lucidrains/denoising-diffusion-pytorch/commit/6b504c4ae9bffa6c36cbfb6a23ee5aba11c4e508",
    )
    args = parser.parse_args()
    data_dict = process_commit(args.commit_url, "test")
    print(data_dict["before_files"][-1])
