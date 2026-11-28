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
        run = "title: G\ngpu: GPU A\nseconds: 60\nframes shown: 100\ndevice removals: 0\ncrashes: 0\n"
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
        for field in ("title", "seconds", "frames shown", "gpu"):
            for run in ("the PR's", "the main"):
                self.assertTrue(any(p.startswith(run) and f"no `{field}` line" in p for p in problems), problems)

    RUN = "title: G\ngpu: {gpu}\nseconds: 60\ndevice removals: 0\ncrashes: 0\npeak commit: {} MB\nframes shown: {}\n"

    def block(self, commit, peak, frames, gpu="GPU A"):
        return f"```\nxcl-smoke v1\ncommit: {commit}\n" + self.RUN.format(peak, frames, gpu=gpu) + "```\n"

    def test_runs_on_different_gpus_are_not_compared(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        on_main = lambda sha: sha in ("1111111aaaa", "2222222bbbb")
        # main only on a slower GPU: refused, though the PR's numbers would pass against it
        body = self.block("1111111aaaa", 4000, 50, "GPU B") + self.block("1234567890", 4000, 100)
        self.assertEqual(["the main and PR regression runs are not on the same GPU (GPU A against GPU B): "
                          "run both on one machine"], check(BODY + body, HEAD, files, on_main))
        # a main run on the PR's GPU next to one on another: compared with the one on the same GPU
        body = (self.block("1111111aaaa", 1000, 300, "GPU B") + self.block("2222222bbbb", 4000, 100)
                + self.block("1234567890", 4000, 100))
        self.assertEqual([], check(BODY + body, HEAD, files, on_main))
        # the same GPU written with other case or spacing is the same GPU
        body = self.block("1111111aaaa", 4000, 100, "gpu  a") + self.block("1234567890", 4000, 100)
        self.assertEqual([], check(BODY + body, HEAD, files, on_main))
        # a block without a gpu line is incomplete
        body = self.block("1111111aaaa", 4000, 100).replace("gpu: GPU A\n", "") + self.block("1234567890", 4000, 100)
        self.assertIn("the main regression run (`xcl-smoke` block of commit 1111111aaaa) has no `gpu` line",
                      check(BODY + body, HEAD, files, on_main))

    def test_a_main_run_with_uncommitted_changes_is_not_a_baseline(self):
        body = ("```\nxcl-smoke v1\ncommit: 1111111aaaa (uncommitted changes)\n" + self.RUN.format(100, 100, gpu="GPU A") + "```\n"
                "```\nxcl-smoke v1\ncommit: 1234567890\n" + self.RUN.format(100, 100, gpu="GPU A") + "```\n")
        problems = check(BODY + body, HEAD, [changed("crates/xcl/src/hle/d3d12x/vram.rs")], lambda sha: sha == "1111111aaaa")
        self.assertEqual(["the main regression run (`xcl-smoke` block of commit 1111111aaaa) has uncommitted "
                          "changes: rerun scripts/smoke.ps1 on a clean checkout of main"], problems)

    def test_the_newest_main_block_is_the_baseline_whatever_its_place(self):
        # an old main run with a low peak pasted first, the current main run after it
        old = "```\nxcl-smoke v1\ncommit: 1111111aaaa\n" + self.RUN.format(1000, 100, gpu="GPU A") + "```\n"
        new = "```\nxcl-smoke v1\ncommit: 2222222bbbb\n" + self.RUN.format(4000, 100, gpu="GPU A") + "```\n"
        pr = "```\nxcl-smoke v1\ncommit: 1234567890\n" + self.RUN.format(4000, 100, gpu="GPU A") + "```\n"
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        on_main = lambda sha: sha in ("1111111aaaa", "2222222bbbb")
        behind = {"1111111aaaa": 40, "2222222bbbb": 2}.get
        for order in (old + new, new + old):
            self.assertEqual([], check(BODY + order + pr, HEAD, files, on_main, main_behind=behind))
        # without the distance the first block is taken, as before
        self.assertIn("the PR's run commits much more memory than main's (4000 MB against 1000 MB)",
                      check(BODY + old + new + pr, HEAD, files, on_main))
        # an unknown distance (a failed API call) does not reorder: the first block, not the older one
        flaky = {"1111111aaaa": 40, "2222222bbbb": None}.get
        self.assertEqual([], check(BODY + new + old + pr, HEAD, files, on_main, main_behind=flaky))
        # a failing comparison with an old baseline says how old it is
        self.assertIn("the PR's run commits much more memory than main's (4000 MB against 1000 MB); "
                      "the main block is 40 commit(s) behind main: rerun it before comparing",
                      check(BODY + old + pr, HEAD, files, on_main, main_behind=behind))

    def test_non_numeric_frames_shown_fails(self):
        run = "title: G\ngpu: GPU A\nseconds: 60\ndevice removals: 0\ncrashes: 0\n"
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


if __name__ == "__main__":
    unittest.main()
