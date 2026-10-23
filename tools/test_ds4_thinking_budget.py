"""CPU-only regression tests of DS4's actual GEN parser and serve decode loop.

The small C++ harness substitutes a deterministic model for GPU execution. The parser,
request type, decode round and emission loop are compiled straight from ds4_main.cpp;
only the T-line printf is replaced to collect the tokens for assertions.

    python -m unittest tools.test_ds4_thinking_budget -v
"""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def top_level(source, signature):
    start = source.index(signature)
    end = re.search(r"^};?\s*$", source[start:], re.MULTILINE)
    if end is None:
        raise AssertionError(f"unterminated source block: {signature}")
    return source[start:start + end.end()] + "\n"


@unittest.skipUnless(shutil.which("c++"), "C++17 compiler required")
class ThinkingBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="maya-ds4-thinking-")
        cls.addClassCleanup(cls.tmp.cleanup)
        directory = Path(cls.tmp.name)
        source = (ROOT / "src/ds4/ds4_main.cpp").read_text()
        blocks = [top_level(source, signature) for signature in (
            "std::vector<int> parse_ids(", "float dspark_cum_default(",
            "bool round(", "struct Req {", "bool parse_gen(",
        )]
        (directory / "ds4_budget_protocol.inc").write_text("\n".join(blocks))
        start = source.index("        for (;;) {\n            if (in.stop.load())")
        end = source.index("        if (dump_f != nullptr) std::fflush(dump_f);", start)
        loop = source[start:end]
        emit = 'std::printf("T %d\\n", tok);'
        if loop.count(emit) != 1:
            raise AssertionError("serve loop's token emission changed; update the CPU harness")
        loop = loop.replace(emit, "emitted.push_back(tok);")
        (directory / "ds4_budget_loop.inc").write_text(loop)
        cls.exe = directory / "thinking-budget-test"
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-I", str(directory), "-I", str(ROOT / "src/ds4"),
                        str(ROOT / "tools/ds4_thinking_budget_test.cpp"), "-o", str(cls.exe)],
                       check=True, timeout=120)

    def run_case(self, name):
        result = subprocess.run([str(self.exe), name], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_protocol_parsing_and_request_reset(self):
        self.run_case("parse")

    def test_invalid_disabled_and_large_budgets(self):
        self.run_case("edges")

    def test_forces_next_token_with_and_without_drafting(self):
        self.run_case("force")

    def test_natural_close_and_budget_equal_to_limit(self):
        self.run_case("natural")

    def test_no_budget_matches_original_decode(self):
        self.run_case("disabled")

    def test_stop_and_cancel_still_take_precedence(self):
        self.run_case("stop")


if __name__ == "__main__":
    unittest.main()
