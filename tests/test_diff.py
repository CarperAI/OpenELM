import itertools

from openelm.utils.diff_eval import (
    DiffState,
    apply_diff,
    parse_line_info,
    replace_text,
    split_diff,
    verify_diff,
)


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
    assert (
        apply_diff(result["file"], result["diff"] + " ", use_line_number=True)
        == result["file"]
    )

    # Test ADDFILE
    test = (
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -1,0 +0,0 @@\n"
        "+aaaaaaa\n+bbbbbbb"
    )
    result = split_diff(test)
    assert (
        apply_diff(result["file"], result["diff"], use_line_number=False)
        == "aaaaaaa\nbbbbbbb"
    )
    assert apply_diff(result["file"], result["diff"], use_line_number=True) == ""

    # Test the loose patching (still performs the patching until invalid diff lines)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\ninvalid\n"
    )
    result = split_diff(test)
    assert (
        apply_diff(result["file"], result["diff"], use_line_number=False)
        == "aaaaaaa\nbbbbbbb\ncccccc"
    )
    # the first part -1,1 is consistent, so perform the patching
    assert (
        apply_diff(result["file"], result["diff"], use_line_number=True)
        == "aaaaaaa\nbbbbbbb\ncccccc"
    )

    # Test multiple @@ ... @@
    test = (
        "<NME> test.py\n<BEF> cccccc\ndddddd\n<MSG> asldkjf\n<DFF> @@ -1,1 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n cccccc\n@@ -2,1 +4,1 @@\n-dddddd\n+eeeeee\n"
    )
    result = split_diff(test)
    assert (
        apply_diff(result["file"], result["diff"], use_line_number=False)
        == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"
    )
    assert (
        apply_diff(result["file"], result["diff"], use_line_number=True)
        == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"
    )
    # Test adding new line on the top, plus the default behavior of line range
    test = (
        "<NME> test.py\n<BEF> cccccc\ndddddd\n<MSG> asldkjf\n<DFF> @@ -0,0 +1,3 @@\n"
        "+aaaaaaa\n+bbbbbbb\n@@ -2 +4 @@\n-dddddd\n+eeeeee\n"
    )
    result = split_diff(test)
    assert apply_diff(result["file"], result["diff"], use_line_number=False) == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"
    assert apply_diff(result["file"], result["diff"], use_line_number=True) == "aaaaaaa\nbbbbbbb\ncccccc\neeeeee"


def test_replace_text():
    test = "abcdawasd\nfwgvwavwefadwwasd"
    before = "wasd"
    after = "1234"
    assert replace_text(test, before, after, 0) == ("abcda1234\nfwgvwavwefadwwasd", 9)

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
    # One can ignore the line range (default to 1)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ -1 +1,3 @@\n"
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
        "<NME> test.py\n<BEF> ADDFILE\n<MSG> asldkjf\n<DFF> @@ -0,0 +1,3 @@\n"
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
    assert verify_diff(test) == DiffState(
        6
    )  # Invalid format (wrong format in @@ ... @@)
    test = (
        "<NME> test.py\n<BEF> cccccc\n<MSG> asldkjf\n<DFF> @@ +1,1 -1,4 @@\n"
        "+aaaaaaa\n+bbbbbbb\n*cccccc\n"
    )
    assert verify_diff(test) == DiffState(7)  # Both


def selected_real_world_diffs():
    diff1 = '<NME> Num.hs\n<BEF> module Data.Propagator.Num where\n\nimport Control.Monad\nimport ' \
            'Control.Monad\nimport Control.Monad.ST\nimport Data.Propagator.Cell\nimport ' \
            'Data.Propagator.Class\nimport Data.Propagator.Supported\nimport Numeric.Natural\n  plus :: Cell s a -> ' \
            'Cell s a -> Cell s a -> ST s ()\n  plus x y z = do\n    lift2 (+) x y z\n    lift2 (-) z x y \n    lift2 ' \
            '(-) z y z\n\n  times :: Cell s a -> Cell s a -> Cell s a -> ST s ()\n  times = lift2 (*)\n\n  cabs :: Eq ' \
            'a => Cell s a -> Cell s a -> ST s ()\n  cabs x y = do\n    lift1 abs x y\n    watch y $ \\b -> when (b ' \
            '== 0) $ write x 0\n  cabs :: Cell s a -> Cell s a -> ST s ()\n  default cabs :: (Num a, Eq a) => Cell s ' \
            'a -> Cell s a -> ST s ()\n  cabs x y = do\n    lift1 abs x y\n    watch y $ \\b -> when (b == 0) $ write ' \
            'x 0\n\n  csignum :: Cell s a -> Cell s a -> ST s ()\n  default csignum :: (Num a, Eq a) => Cell s a -> ' \
            'Cell s a -> ST s ()\n  csignum x y = do\n    lift1 signum x y\n    watch y $ \\b -> when (b == 0) $ ' \
            'write x 0\n\ninstance PropagatedNum Integer where\n      else watch y $ \\ b -> when (b /= 0) $ write x ' \
            '0\n  cabs = unify\n\ninstance PropagatedNum Int \ninstance PropagatedNum Word where\n  cabs = unify\n\n  ' \
            '    if c == 0 then do\n        watch x $ \\ a -> when (a /= 0) $ write y 0\n        watch y $ \\ b -> ' \
            'when (b /= 0) $ write x 0\n      else do\n        watch x $ \\ a -> write y (c `div` a)\n        watch y ' \
            '$ \\ b -> write x (c `div` b)\n  cabs = unify\n\ninstance PropagatedNum (Supported Natural) where\n  ' \
            'ctimes x y z = do\n    lift2 (*) x y z\n    watch z $ \\c ->\n      when (c == 0) $ do\n        watch x ' \
            '$ \\ a -> when (a /= 0) $ write y 0\n        watch y $ \\ b -> when (b /= 0) $ write x 0\n  cabs = ' \
            'unify\n\ninstance PropagatedNum Int\ninstance PropagatedNum (Supported Int)\n\ninstance PropagatedNum ' \
            'Word where\n  cabs = unify\n\ninstance PropagatedNum (Supported Word) where\n  cabs = ' \
            'unify\n\nctimesFractional :: (Eq a, Fractional a) => Cell s a -> Cell s a -> Cell s a -> ST s (' \
            ')\nctimesFractional x y z = do\n  watch x $ \\a ->\n    if a == 0\n    then write z 0\n    else do\n     ' \
            ' with y $ \\b -> write z (a*b)\n      with z $ \\c -> write y (c/a) -- a /= 0 determined above\n  watch ' \
            'y $ \\b ->\n    if b == 0\n    then write z 0\n    else do\n      with x $ \\a -> write z (a*b)\n      ' \
            'with z $ \\c -> write x (c/b) -- b /= 0 determined above\n  watch z $ \\c -> do\n    with x $ \\a -> ' \
            'when (a /= 0) $ write y (c/a)\n    with y $ \\b -> when (b /= 0) $ write x (c/b)\n\ninstance ' \
            'PropagatedNum Rational where\n  ctimes = ctimesFractional\n\ninstance PropagatedNum (Supported Rational) ' \
            'where\n  ctimes = ctimesFractional\n\ninstance PropagatedNum Double where\n  ctimes = ' \
            'ctimesFractional\n\ninstance PropagatedNum (Supported Double) where\n  ctimes = ' \
            'ctimesFractional\n\ninstance PropagatedNum Float where\n  ctimes = ctimesFractional\n\ninstance ' \
            'PropagatedNum (Supported Float) where\n  ctimes = ctimesFractional\n\nclass PropagatedNum a => ' \
            'PropagatedFloating a where\n  cexp :: Cell s a -> Cell s a -> ST s ()\n  default cexp :: Floating a => ' \
            'Cell s a -> Cell s a -> ST s ()\n  cexp x y = do\n    lift1 exp x y\n    lift1 log y x\n\n  csqrt :: ' \
            'Cell s a -> Cell s a -> ST s ()\n  default csqrt :: Floating a => Cell s a -> Cell s a -> ST s ()\n  ' \
            'csqrt x y = do\n    lift1 sqrt x y\n    lift1 (\\a -> a*a) y x\n\n  csin :: Cell s a -> Cell s a -> ST s ' \
            '()\n  default csin :: (Floating a, Ord a) => Cell s a -> Cell s a -> ST s ()\n  csin x y = do\n    lift1 ' \
            'sin x y\n    watch y $ \\b -> do\n       unless (abs b <= 1) $ fail "output of sin not between -1 and ' \
            '1"\n       write x (asin b)\n\n  ccos :: Cell s a -> Cell s a -> ST s ()\n  default ccos :: (Floating a, ' \
            'Ord a) => Cell s a -> Cell s a -> ST s ()\n  ccos x y = do\n    lift1 cos x y\n    watch y $ \\b -> do\n ' \
            '      unless (abs b <= 1) $ fail "output of cos not between -1 and 1"\n       write x (acos b)\n\n  ctan ' \
            ':: Cell s a -> Cell s a -> ST s ()\n  default ctan :: (Floating a, Ord a) => Cell s a -> Cell s a -> ST ' \
            's ()\n  ctan x y = do\n    lift1 tan x y\n    watch y $ \\b -> do\n      unless (abs b <= pi/2) $ fail ' \
            '"output of tan not between -pi/2 and pi/2"\n      write x (atan b)\n\n  csinh :: Cell s a -> Cell s a -> ' \
            'ST s ()\n  default csinh :: (Floating a, Ord a) => Cell s a -> Cell s a -> ST s ()\n  csinh x y = do\n   ' \
            ' lift1 sinh x y\n    lift1 asinh y x\n\n  ccosh :: Cell s a -> Cell s a -> ST s ()\n  default ccosh :: (' \
            'Floating a, Ord a) => Cell s a -> Cell s a -> ST s ()\n  ccosh x y = do\n    lift1 cosh x y\n    watch y ' \
            '$ \\b -> do\n      unless (b >= 1) $ fail "output of cosh not >= 1"\n      lift1 acosh y x\n\n  ctanh :: ' \
            'Cell s a -> Cell s a -> ST s ()\n  default ctanh :: (Floating a, Ord a) => Cell s a -> Cell s a -> ST s ' \
            '()\n  ctanh x y = do\n    lift1 tanh x y\n    watch y $ \\b -> do\n      unless (abs b <= 1) $ fail ' \
            '"output of tanh not between -1 and 1"\n      write x (tanh b)\n\ninstance PropagatedFloating ' \
            'Float\ninstance PropagatedFloating (Supported Float)\ninstance PropagatedFloating Double\ninstance ' \
            'PropagatedFloating (Supported Double)\n\n-- Interval arithmetic\n\nclass (Floating a, Ord a) => ' \
            'PropagatedInterval a where\n  infinity :: a\n\n\ninstance PropagatedInterval Double where\n  infinity = ' \
            '1/0\n\ninstance PropagatedInterval (Supported Double) where\n  infinity = 1/0\n\ninstance ' \
            'PropagatedInterval Float where\n  infinity = 1/0\n\ninstance PropagatedInterval (Supported Float) ' \
            'where\n  infinity = 1/0\n\ninstance PropagatedInterval a => PropagatedNum (Interval a) where\n  ctimes = ' \
            'ctimesFractional\n\n  cabs x y = do\n    write y (0...infinity)\n    lift1 abs x y\n    -- todo: use ' \
            'symmetric_positive\n    watch y $ \\case\n      I _ b -> write x (-b...b)\n      Empty -> write x ' \
            'Empty\n\n  csignum x y = do\n    write y (-1...1)\n    lift1 signum x y\n    watch y $ \\case\n      I a ' \
            'b | a < 1 && b > -1 -> write x $ I (if a <= -1 then -infinity else 0) (if b >= 1 then infinity else 0)\n ' \
            '     _ -> write x Empty\n\nsymmetric_positive :: (Num a, Ord a) => (Interval a -> Interval a) -> Cell s ' \
            '(Interval a) -> Cell s (Interval a) -> ST s ()\nsymmetric_positive f x y = do\n  watch y $ \\case\n    ' \
            'Empty -> write x Empty -- if the result is empty then the input is empty\n    I a a\' -> do\n      when ' \
            '(a\' <= 0) $ with x $ \\c -> write y (- f c)\n      when (a >= 0)  $ with x $ \\c -> write y (f c)\n  ' \
            'lift1 (\\c -> let d = f c in hull (-d) d) x y\n\n-- x = f y + p*n\n-- n = (x - f y)/p, ' \
            'n is an integer\nperiodic :: RealFrac a => Interval a -> (Interval a -> Interval a) -> Cell s (Interval ' \
            'a) -> Cell s (Interval a) -> ST s ()\nperiodic p f x y = do\n  watch2 x y $ \\a b -> let c = f b in case ' \
            '(a - c) / p of\n    Empty -> write x Empty\n    I l h -> write x (c + p*(fromIntegral (ceiling l :: ' \
            'Integer)...fromIntegral (floor h :: Integer)))\n\ninstance (PropagatedInterval a, RealFloat a) => ' \
            'PropagatedFloating (Interval a) where\n  cexp x y = do\n    write y (0...infinity)\n    lift1 exp x y\n  ' \
            '  lift1 log y x\n\n  csqrt x y = do\n    write x (0...infinity)\n    lift1 (\\b -> b*b) y x\n    ' \
            'symmetric_positive sqrt x y\n\n  csin x y = do\n    write y (-1...1)\n    lift1 sin x y\n    periodic (' \
            '2*pi) asin y x\n\n  ccos x y = do\n    write y (-1...1)\n    lift1 cos x y\n    periodic (2*pi) acos y ' \
            'x\n\n  ctan x y = do\n    write y (-pi/2...pi/2)\n    lift1 tan x y\n    periodic pi atan y x\n\n  csinh ' \
            'x y = do\n    lift1 sinh x y\n    lift1 asinh y x\n\n  ccosh x y = do\n    write y (1...infinity)\n    ' \
            'lift1 cosh x y\n    symmetric_positive acosh x y\n\n  ctanh x y = do\n    write y (-1...1)\n    lift1 ' \
            'tanh x y\n    lift1 atanh y x\n\n<MSG> PropagatedNum rational\n\n<DFF> @@ -1,3 +1,6 @@\n+{-# LANGUAGE ' \
            'DefaultSignatures #-}\n+{-# LANGUAGE FlexibleInstances #-}\n+{-# LANGUAGE TypeSynonymInstances #-}\n ' \
            'module Data.Propagator.Num where\n \n import Control.Monad\n@@ -10,13 +13,14 @@ class (Propagated a, ' \
            'Num a) => PropagatedNum a where\n   plus :: Cell s a -> Cell s a -> Cell s a -> ST s ()\n   plus x y z = ' \
            'do\n     lift2 (+) x y z\n-    lift2 (-) z x y \n+    lift2 (-) z x y\n     lift2 (-) z y z\n \n   times ' \
            ':: Cell s a -> Cell s a -> Cell s a -> ST s ()\n   times = lift2 (*)\n \n-  cabs :: Eq a => Cell s a -> ' \
            'Cell s a -> ST s ()\n+  cabs :: Cell s a -> Cell s a -> ST s ()\n+  default cabs :: Eq a => Cell s a -> ' \
            'Cell s a -> ST s ()\n   cabs x y = do\n     lift1 abs x y\n     watch y $ \\b -> when (b == 0) $ write x ' \
            '0\n@@ -36,7 +40,25 @@ instance PropagatedNum Natural where\n       else watch y $ \\ b -> when (b /= 0) ' \
            '$ write x 0\n   cabs = unify\n \n-instance PropagatedNum Int \n+instance PropagatedNum Int\n+\n instance ' \
            'PropagatedNum Word where\n   cabs = unify\n \n+instance PropagatedNum Rational where\n+  times x y z = ' \
            'do\n+    watch x $ \\a ->\n+      if a == 0\n+      then write z 0\n+      else do\n+        with y $ ' \
            '\\b -> write z (a*b)\n+        with z $ \\c -> write y (c/a) -- a /= 0 determined above\n+    watch y $ ' \
            '\\b ->\n+      if b == 0\n+      then write z 0\n+      else do\n+        with x $ \\a -> write z (' \
            'a*b)\n+        with z $ \\c -> write x (c/b) -- b /= 0 determined above\n+    watch z $ \\c -> do\n+     ' \
            ' with x $ \\a -> when (a /= 0) $ write y (c/a)\n+      with y $ \\b -> when (b /= 0) $ write x (c/b)\n '

    assert verify_diff(diff1).value == 0
