from elm.util.diff_eval import apply_diff, parse_line_info, replace_text, split_diff


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
        "location /static {\n "
    )
    result = split_diff(test)
    # Because it is a valid diff, whichever method we apply the patch should result in the same.
    assert apply_diff(
        result["file"], result["diff"], use_line_number=False
    ) == apply_diff(result["file"], result["diff"], use_line_number=True)


def test_replace_text():
    test = "abcdawasdfwgvwavwefadwwasd"
    before = "wasd"
    after = "1234"
    assert replace_text(test, before, after) == "abcda1234fwgvwavwefadwwasd"

    test = "asdsdsadafwasd"
    assert replace_text(test, before, after) == "asdsdsadaf1234"
