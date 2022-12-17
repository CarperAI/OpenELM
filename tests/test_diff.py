import itertools

from elm.util.diff_eval import apply_diff, parse_line_info, replace_text, split_diff, verify_diff, DiffState


def test_diff():
    test = "<NME> test.py\n<BEF> asdafdsf\nasdlfjalskj\n  asldfajlk\n<MSG> asldkjf\n<DFF> asdkjjfajjsdfj\najföj"
    result = split_diff(test)
    assert result["name"] == "test.py"
    assert result["file"] == "asdafdsf\nasdlfjalskj\n  asldfajlk"
    assert result["message"] == "asldkjf"
    assert result["diff"] == "asdkjjfajjsdfj\najföj"

    test = "@@ -11,22 +33,45678 @@"
    result = parse_line_info(test)
    assert result == (11, 22, 33, 45678)

    # More serious test on an actual diff
    test = (
        "<NME> conf_template.lua\n<BEF> local _M = {}\n\nfunction _M:get_ngx_conf_template()\n    return [[\n# "
        "user www www;\npid tmp/{{LOR_ENV}}-nginx.pid;\n\n# This number should be at maxium the number of CPU on "
        "the server\nworker_processes 4;\n\nevents {\n    # Number of connections per worker\n    "
        "worker_connections 4096;\n}\n\nhttp {\n    sendfile on;\n    include ./mime.types;\n\n    {{"
        "LUA_PACKAGE_PATH}}\n    lua_code_cache on;\n\n    server {\n        # List port\n        listen {{"
        "PORT}};\n\n        # Access log\n        access_log logs/{{LOR_ENV}}-access.log;\n\n        # Error log\n "
        "       error_log logs/{{LOR_ENV}}-error.log;\n\n        # this variable is for view "
        "render（lua-resty-template)\n        set $template_root '';\n\n        location /static {\n            "
        "alias {{STATIC_FILE_DIRECTORY}}; #app/static;\n        }\n\n        # lor runtime\n        {{"
        "CONTENT_BY_LUA_FILE}}\n    }\n}\n]]\nend\n\nreturn _M\n\n<MSG> Merge pull request #100 from "
        "hanxi/patch-2\n\nfix a typo\n<DFF> @@ -30,7 +30,7 @@ http {\n         # Error log\n         error_log "
        "logs/{{LOR_ENV}}-error.log;\n \n-        # this variable is for view render（lua-resty-template)\n+        "
        "# this variable is for view render(lua-resty-template)\n         set $template_root '';\n \n         "
        "location /static {\n"
    )
    result = split_diff(test)
    # Because it is a valid diff, whichever method we apply the patch should result in the same.
    assert apply_diff(
        result["file"], result["diff"], use_line_number=False
    ) == apply_diff(result["file"], result["diff"], use_line_number=True)
    # Reject if pre-diff lines don't fully match
    assert apply_diff(result["file"], result["diff"] + " ", use_line_number=True) == result["file"]

    # Test ADDFILE
    test = (
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -1,0 +0,0 @@\n"
        "+aaaaaaa\n+bbbbbbb"
    )
    result = split_diff(test)
    assert apply_diff(result["file"], result["diff"], use_line_number=False) == "aaaaaaa\nbbbbbbb"
    assert apply_diff(result["file"], result["diff"], use_line_number=True) == ""

    # Test the loose patching (still performs the patching until invalid diff lines)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\ninvalid\n"
    )
    result = split_diff(test)
    assert apply_diff(result["file"], result["diff"], use_line_number=False) == "aaaaaaa\nbbbbbbb\ncccccc"
    # the first part -1,1 is consistent, so perform the patching
    assert apply_diff(result["file"], result["diff"], use_line_number=True) == "aaaaaaa\nbbbbbbb\ncccccc"

    # Test multiple @@ ... @@
    test = (
        "<NME> test.py\n<BEF> cccccc\ndddddd\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n@@ -2,1 +4,1 @@\n-dddddd\n+eeeeee\n"
    )
    result = split_diff(test)
    assert apply_diff(result["file"], result["diff"], use_line_number=False) == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"
    assert apply_diff(result["file"], result["diff"], use_line_number=True) == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"


def test_replace_text():
    test = "abcdawasdfwgvwavwefadwwasd"
    before = "wasd"
    after = "1234"
    assert replace_text(test, before, after, 0) == ("abcda1234fwgvwavwefadwwasd", 9)

    test = "asdsdsadafwasd"
    assert replace_text(test, before, after, 0) == ("asdsdsadaf1234", 14)


def test_verify_diff():
    """
    Go over as many bad diff as possible to test the status code of `verify_diff`
    """

    # Valid diff
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n"
    )
    assert verify_diff(test) == DiffState(0)
    # Multiple valid
    test = (
        "<NME> test.py\n<BEF> cccccc\ndddddd\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n@@ -2,1 +4,1 @@\n-dddddd\n+eeeeee\n"
    )
    assert verify_diff(test) == DiffState(0)

    # Test ADDFILE
    test = (
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n"
    )
    assert verify_diff(test) == DiffState(6)  # Wrong line numbers
    test = (
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -0,0 +0,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n"
    )
    assert verify_diff(test) == DiffState(6)  # Contains lines not starting with "+"
    test = (
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -0,0 +0,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n+cccccc\n"
    )
    assert verify_diff(test) == DiffState(0)

    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n ccccc\n"
    )
    assert verify_diff(test) == DiffState(1)
    for a, b, c, d in itertools.product([0, 2], repeat=4):
        for e in ["c", ""]:
            test = (
                f"<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -{a},{b} +{c},{d} @@\n"
                f"+aaaaaaa\n+bbbbbbb\n ccccc{e}\n"
            )
            assert verify_diff(test) == DiffState(2) if e == "c" else DiffState(3)
    test = (
        "<NME> test.py\n<BFE> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,4 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n"
    )
    assert verify_diff(test) == DiffState(4)  # Invalid format (BFE instead of BEF)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n*cccccc\n"
    )
    assert verify_diff(test) == DiffState(5)  # Invalid hunk (start with other char)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ +1,1 -1,4 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n"
    )
    assert verify_diff(test) == DiffState(6)  # Invalid format (wrong format in @@ ... @@)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ +1,1 -1,4 @@\n"
        "+aaaaaaa\n+bbbbbbb\n*cccccc\n"
    )
    assert verify_diff(test) == DiffState(7)  # Both
