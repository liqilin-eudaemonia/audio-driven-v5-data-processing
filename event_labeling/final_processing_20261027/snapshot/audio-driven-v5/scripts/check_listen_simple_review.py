#!/usr/bin/env python3
"""Real-browser checks of the prompt comparison and synchronized evidence playback."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_previews/listen_benchmark_review/native_simple"
URL = "http://127.0.0.1:8876/listen_benchmark_review/native_simple/"


def main():
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width":1560,"height":1050}, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(URL)
        page.wait_for_function("window.promptReview && window.promptState")
        page.evaluate("document.fonts.ready")
        assert page.locator("#samples button").count() == 50
        assert page.locator("#stats tr").count() == 4
        assert page.evaluate("promptReview.status.completed_requests") == 50
        assert "51 个英文词" in page.locator("#analysis").inner_text()
        checks.append("50 samples, completed 50-request comparison and qualified analysis")
        page.wait_for_function("document.querySelector('#video').readyState>=2 && document.querySelector('#audio').readyState>=2")
        page.locator('[data-model="native_simple_v2"][data-event="0"]').click()
        page.wait_for_timeout(1800)
        state = page.evaluate("({v:video.currentTime,a:audio.currentTime,paused:video.paused,muted:video.muted,loop:promptState().loop})")
        assert not state["paused"] and state["muted"] and state["loop"] == [0,1], state
        assert 0 <= state["v"] <= 1.1 and abs(state["v"]-state["a"])<.25, state
        page.locator("#stopLoop").click()
        assert page.evaluate("promptState().loop") is None
        page.evaluate("video.pause()")
        checks.append("independent WAV, muted video, event seek and repeat playback")
        page.locator('[data-case="S0699_I00000500"]').click()
        assert "每次“okay”" in page.locator("#feedback").inner_text()
        assert page.locator('[data-model="native_simple_v2"]').count()==2
        assert page.locator('[data-model="native_simple_v2"][disabled]').count()==2
        assert "Nodding head" in page.locator("#columns").inner_text()
        checks.append("failed temporal validation preserves raw nod and okay descriptions with disabled zero-length playback")
        page.locator("#reference").check()
        assert page.locator("#columns article").count() == 3
        page.locator("#feedbackOnly").check()
        assert page.locator("#samples button").count() == 4
        page.locator("#choice").select_option("uncertain")
        page.locator("#note").fill("AUTOMATED BROWSER CHECK — not human feedback")
        page.locator("#save").click()
        with page.expect_download() as dl:
            page.locator("#export").click()
        exported = json.loads(Path(dl.value.path()).read_text())
        assert exported["human_ground_truth"] is False
        assert exported["records"][0]["sample_id"].endswith("3491a3549642")
        page.reload()
        page.locator('[data-case="S0699_I00000500"]').click()
        assert page.locator("#note").input_value().startswith("AUTOMATED BROWSER CHECK")
        checks.append("optional previous-long-prompt reference, feedback filter, browser persistence and JSON export")
        page.locator('[data-case="S3370_I00000221"]').click()
        page.wait_for_function("document.querySelector('#video').readyState>=2")
        page.screenshot(path=str(OUT / "comparison_desktop.png"), full_page=True)
        data = page.evaluate("promptReview")
        for sample in data["samples"]:
            for key in ("audio", "video"):
                response = context.request.get(URL+sample[key], headers={"Range":"bytes=0-127"})
                assert response.status == 206, (sample["id"], key, response.status)
        checks.append("100 original media endpoints support byte-range playback")
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(OUT / "comparison_mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth+1")
        checks.append("390px layout without page overflow")
        assert not errors, errors
        browser.close()
    report = dict(passed=True, checks=checks, page_errors=errors, synthetic_review_only=True)
    (OUT / "browser_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
