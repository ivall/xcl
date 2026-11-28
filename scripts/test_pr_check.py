from pathlib import Path
import unittest

from pr_check import check, smoke_blocks


HEAD = "1234567890abcdef"
BODY = "**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\nNone\n"


def changed(path, binary=False):
    return (path, "modified", binary, lambda: 10)


class TestOnlyApprovalTests(unittest.TestCase):
    def test_runtime_path_requires_smoke_until_approved(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        problems = check(BODY, HEAD, files, lambda _: False)
        self.assertEqual(2, len(problems))
        self.assertTrue(all("Regression run" in problem for problem in problems))
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False, test_only=True))

    def test_approval_keeps_description_and_asset_checks(self):
        files = [
            changed("crates/xcl/src/hle/d3d12x/vram.rs"),
            changed("assets/content.rpf"),
            changed("notes/attachment.txt", binary=True),
        ]
        body = "**Tested**\nSynthetic fixtures\n**AI use**\nQwen\n"
        problems = check(body, HEAD, files, lambda _: False, test_only=True)
        self.assertEqual(4, len(problems))
        for reason in ("Not tested", "AI use", "content.rpf", "attachment.txt"):
            self.assertTrue(any(reason in problem for problem in problems), problems)

    def test_runtime_build_inputs_require_smoke_until_approved(self):
        for path in ("crates/xcl/data/stubdb.toml", "crates/xcl/build.rs", "crates/xcl/Cargo.toml",
                     "Cargo.toml", "Cargo.lock"):
            with self.subTest(path=path):
                problems = check(BODY, HEAD, [changed(path)], lambda _: False)
                self.assertEqual(2, len(problems))
                self.assertTrue(all("Regression run" in problem for problem in problems))
                self.assertEqual([], check(BODY, HEAD, [changed(path)], lambda _: False, test_only=True))
        for path in ("tests/testtitle/Cargo.toml", "tests/testgamedll/build.rs", "crates/xcl/tests/data/x.toml"):
            with self.subTest(path=path):
                self.assertEqual([], check(BODY, HEAD, [changed(path)], lambda _: False))

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))


class BannedModelTests(unittest.TestCase):
    def ai_use(self, line):
        body = BODY.replace("**AI use**\nNone\n", "**AI use**\n" + line + "\n")
        return check(body, HEAD, [changed("README.md")], lambda _: True)

    def test_family_names_of_small_and_open_weight_models_are_refused(self):
        for model in ("CodeLlama 34B", "TinyLlama", "Codestral 22B", "Devstral Small", "QwQ-32B",
                      "Phi 4", "Magistral", "Ministral 8B", "CodeQwen1.5"):
            with self.subTest(model=model):
                problems = self.ai_use("- Cline with " + model + ", wrote the code")
                self.assertEqual(1, len(problems), problems)
                self.assertIn("does not accept", problems[0])

    def test_frontier_models_and_ordinary_words_pass(self):
        for line in ("- Claude Code with Claude Opus 5.5", "- Claude Code with Claude Sonnet 5.5",
                     "- GPT-6", "- Gemini 4", "- Codex", "- Phil 2 checked the alphabet and graph 3"):
            with self.subTest(line=line):
                self.assertEqual([], self.ai_use(line))


class SmokeBlockTests(unittest.TestCase):
    def test_two_blocks_in_one_fence_are_two_blocks(self):
        run = "title: G\nseconds: 60\nframes shown: 100\ndevice removals: 0\ncrashes: 0\n"
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa\n" + run + "\n"
                "xcl-smoke v1\ncommit: 1234567890\n" + run + "```\n")
        blocks = smoke_blocks(body)
        self.assertEqual(["1111111aaaa", "1234567890"], [b["commit"] for b in blocks])
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        self.assertEqual([], check(BODY + body, HEAD, files, lambda sha: sha == "1111111aaaa"))

    def test_blocks_in_separate_fences_stay_separate(self):
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa\ntitle: G\n```\ntext\n"
                "```\nxcl-smoke v1\ncommit: 1234567890\ntitle: G\n```\n")
        self.assertEqual(["1111111aaaa", "1234567890"], [b["commit"] for b in smoke_blocks(body)])

    def smoke_problems(self, main_run, pr_run):
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa\n" + main_run + "```\n"
                "```\nxcl-smoke v1\ncommit: 1234567890\n" + pr_run + "```\n")
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        return check(BODY + body, HEAD, files, lambda sha: sha == "1111111aaaa")

    def test_blocks_with_only_commit_lines_fail(self):
        problems = self.smoke_problems("", "")
        for field in ("title", "seconds", "frames shown"):
            for run in ("the PR's", "the main"):
                self.assertTrue(any(p.startswith(run) and f"no `{field}` line" in p for p in problems), problems)

    def test_non_numeric_frames_shown_fails(self):
        run = "title: G\nseconds: 60\ndevice removals: 0\ncrashes: 0\n"
        problems = self.smoke_problems(run + "frames shown: 100\n", run + "frames shown: many\n")
        self.assertIn("the PR's regression run (`xcl-smoke` block of commit 1234567890) "
                      "has a non-numeric `frames shown`: many", problems)


class TemplatePlaceholderTests(unittest.TestCase):
    def test_untouched_pr_template_fails_all_required_sections(self):
        template = Path(__file__).resolve().parent.parent / ".github" / "pull_request_template.md"
        body = template.read_text(encoding="utf-8")
        problems = check(body, HEAD, [changed("README.md")], lambda _: True)
        self.assertEqual(
            [
                "the **Tested** section is missing or empty",
                "the **Not tested** section is missing or empty",
                "the **AI use** section is missing or empty",
            ],
            problems,
        )

    def test_filled_template_with_comments_and_real_lines_passes(self):
        body = (
            "**Tested**\n"
            "<!-- comment only does not count, real line below does -->\n"
            "- hardware (GPU + driver, CPU), Rust version, debug / release: RTX 3070 Ti, Rust 1.88.0\n"
            "- `cargo test --workspace -- --test-threads=4`: N passed, N failed, N ignored (SKIPPED GPU tests: …)\n"
            "**Not tested**\n"
            "- AMD GPUs\n"
            "**AI use**\n"
            "- none\n"
        )
        self.assertEqual([], check(body, HEAD, [changed("README.md")], lambda _: True))

    def test_bare_bullets_and_html_comments_only_fail(self):
        body = "**Tested**\n<!-- todo -->\n-\n**Not tested**\n*\n**AI use**\n  -  \n"
        problems = check(body, HEAD, [changed("README.md")], lambda _: True)
        self.assertEqual(3, len(problems))


class AssetAndModelPolicyTests(unittest.TestCase):
    def test_banned_models_checked_only_under_ai_use(self):
        for model in ("qwen", "DeepSeek", "Llama", "mistral", "Mixtral", "Gemma", "phi-3", "PHI4"):
            with self.subTest(model=model):
                body = f"**Tested**\nUnit tests\n**Not tested**\nGPU\n**AI use**\n{model}\n"
                problems = check(body, HEAD, [changed("README.md")], lambda _: True)
                self.assertEqual(1, len(problems))
                self.assertIn("**AI use**", problems[0])

        prose_mention = (
            "Fixes #1 (notes mention qwen and llama)\n"
            "**Tested**\nUnit tests\n**Not tested**\nGPU\n**AI use**\nClaude Opus 4\n"
        )
        self.assertEqual([], check(prose_mention, HEAD, [changed("README.md")], lambda _: True))

    def test_game_files_and_test_data_size_limits(self):
        self.assertEqual(
            [],
            check(
                BODY,
                HEAD,
                [
                    changed("tests/data/testgame/MicrosoftGame.config"),
                    ("tests/data/small.bin", "added", True, lambda: 1 << 20),
                    ("assets/removed.rpf", "removed", True, lambda: 1 << 30),
                ],
                lambda _: True,
            ),
        )
        problems = check(
            BODY,
            HEAD,
            [
                changed("samples/MicrosoftGame.config"),
                changed("samples/appxmanifest.xml"),
                ("tests/data/too_big.bin", "added", True, lambda: (1 << 20) + 1),
            ],
            lambda _: True,
        )
        self.assertEqual(3, len(problems))


class RegressionPolicyTests(unittest.TestCase):
    RUNTIME = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]

    @staticmethod
    def smoke_body(
        *,
        main_commit="1111111aaaa",
        pr_commit=HEAD,
        title="G",
        main_seconds=60,
        pr_seconds=60,
        frames_main=1800,
        frames_pr=1800,
        commit_main=1000,
        commit_pr=1000,
        device_removals=0,
        crashes=0,
        exit_line="clean",
    ):
        return (
            f"{BODY}"
            f"```\n"
            f"xcl-smoke v1\ncommit: {main_commit}\ntitle: {title}\nseconds: {main_seconds}\n"
            f"frames shown: {frames_main}\npeak commit: {commit_main} MB\n"
            f"device removals: 0\ncrashes: 0\nexit: clean\n"
            f"```\n"
            f"```\n"
            f"xcl-smoke v1\ncommit: {pr_commit}\ntitle: {title}\nseconds: {pr_seconds}\n"
            f"frames shown: {frames_pr}\npeak commit: {commit_pr} MB\n"
            f"device removals: {device_removals}\ncrashes: {crashes}\nexit: {exit_line}\n"
            f"```\n"
        )

    def test_thresholds_and_clean_run_boundaries(self):
        ok_body = self.smoke_body(frames_main=1000, frames_pr=800, commit_main=1000, commit_pr=1250)
        self.assertEqual([], check(ok_body, HEAD, self.RUNTIME, lambda sha: sha == "1111111aaaa"))

        too_slow = self.smoke_body(frames_main=1000, frames_pr=799)
        self.assertIn("fewer frames", check(too_slow, HEAD, self.RUNTIME, lambda sha: sha == "1111111aaaa")[0])

        too_much_mem = self.smoke_body(commit_main=1000, commit_pr=1251)
        self.assertIn("more memory", check(too_much_mem, HEAD, self.RUNTIME, lambda sha: sha == "1111111aaaa")[0])

    def test_failures_uncommitted_and_mismatched_runs_fail(self):
        for kwargs, needle in [
            ({"device_removals": 1}, "device removals"),
            ({"crashes": 1}, "crashes"),
            ({"exit_line": "killed (timeout)"}, "did not end by itself"),
            ({"pr_commit": f"{HEAD} uncommitted"}, "no uncommitted changes"),
            ({"pr_seconds": 30}, "not of the same title for the same time"),
        ]:
            with self.subTest(needle=needle):
                body = self.smoke_body(**kwargs)
                problems = check(body, HEAD, self.RUNTIME, lambda sha: sha == "1111111aaaa")
                self.assertTrue(any(needle in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
