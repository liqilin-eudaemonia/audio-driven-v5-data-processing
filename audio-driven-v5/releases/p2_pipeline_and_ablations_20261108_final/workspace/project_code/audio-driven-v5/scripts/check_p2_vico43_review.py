"""Verify source identity, all43 browser cases, playback and mobile layout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT/"data_previews/listen_benchmark_review/p2_vico43_combined8000_20261106"
OUT = ROOT/"experiments/p2_vico_motion8000_20261106"
BASE = ROOT/"experiments/p2_vico256_combined8000_20261106"
URL = "http://127.0.0.1:8876/listen_benchmark_review/p2_vico43_combined8000_20261106/"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d = json.loads((PAGE/"data.json").read_text());assert len(d["cases"]) == 43
    assert len({c["id"] for c in d["cases"]}) == 43
    assert sha(OUT/"RESULTS.json") == d["source_results_sha256"]
    result = json.loads((OUT/"RESULTS.json").read_text())
    per_clip = {r["id"]: r["metrics_raw"] for r in result["per_clip"]}
    assert d["summary"] == result["metrics_table_scale"]
    assert d["coverage"] == result["coverage_clips"]
    checks = []
    for c in d["cases"]:
        uid = c["id"];assert c["metrics"] == per_clip[uid]
        native = BASE/"generated/base_event_feature_whisper_8000/vico"/uid
        meta = json.loads((native/"metadata.json").read_text())
        assert meta["step"] == 8000 and meta["seed"] == 137
        assert meta["listener_future_input"] is False and meta["target_description_access"] is False
        assert (PAGE/c["native_video"]).resolve() == (native/"generated.mp4").resolve()
        assert (PAGE/c["scoring_video"]).resolve() == (BASE/"scoring/base_event_feature_whisper_8000"/(uid+".listener.mp4")).resolve()
        assert (PAGE/c["first_frame"]).resolve() == (BASE/"conditions/vico"/uid/"first_frame.png").resolve()
        assert sha(PAGE/c["generated_wav"]) == c["audio"]["generated_wav_sha256"]
        assert sha(PAGE/c["speaker_wav"]) == c["audio"]["speaker_wav_sha256"]
        assert sha(PAGE/c["scoring_video"]) == c["source_sha256"]["prediction"]
        assert sha(PAGE/c["raw_gt"]) == c["source_sha256"]["gt"]
        for role in ["speaker", "gt"]:
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,nb_frames,r_frame_rate", "-of", "json", str(PAGE/c[role+"_video"])]))["streams"][0]
            assert (info["codec_name"], info["width"], info["height"], info["r_frame_rate"], int(info["nb_frames"])) == ("h264",256,256,"30/1",c["frames"])
        checks.append(dict(id=uid, identity=True, preview_frames=c["frames"]))
    os.environ["LD_LIBRARY_PATH"] = "/tmp/native-fps-browser-libs/extracted/usr/lib/x86_64-linux-gnu:"+os.environ.get("LD_LIBRARY_PATH", "")
    from playwright.sync_api import sync_playwright
    errors, browser_checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/root/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell", headless=True, args=["--no-sandbox", "--disable-gpu"])
        page = browser.new_page(viewport={"width":1500,"height":1050}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        assert page.goto(URL).status == 200
        page.wait_for_function("window.reviewData && reviewDebug().loaded", timeout=60000)
        assert page.locator("#list button").count() == 43
        page.locator("#overview summary").click();assert page.locator("#aggregate tbody tr").count() == 7
        page.click("#play");page.wait_for_timeout(1000)
        playback = page.evaluate("reviewDebug()")
        assert playback["context"] == "running" and playback["sources"] == 2
        assert min(playback["times"].values()) > .3
        assert max(playback["times"].values())-min(playback["times"].values()) < .15
        for mode, count in [("speaker",1), ("generated",1), ("silent",0), ("both",2)]:
            page.select_option("#audioMode",mode);assert page.evaluate("reviewDebug().sources") == count
        page.select_option("#speed", "0.5")
        assert page.evaluate("sources.every(s=>s.playbackRate.value===.5)")
        page.locator("#seek").evaluate("el=>{el.value=1.25;el.dispatchEvent(new Event('input'))}")
        page.wait_for_timeout(150);assert page.evaluate("reviewDebug().times.gt") > 1.2
        page.click("#pause");assert page.evaluate("reviewDebug().sources") == 0
        page.select_option("#speed", "1")
        for c in d["cases"]:
            page.evaluate("id=>show(reviewData.cases.findIndex(c=>c.id===id)).catch(fail)", c["id"])
            page.wait_for_function("id=>reviewDebug().sample===id&&reviewDebug().loaded", arg=c["id"], timeout=60000)
            page.select_option("#resolution", "native")
            page.wait_for_function("reviewDebug().loaded && reviewDebug().sizes.prediction[0]===512", timeout=60000)
            assert page.evaluate("Object.values(videos).every(v=>v.readyState>=2 && !v.error)")
            page.select_option("#resolution", "scoring")
            page.wait_for_function("reviewDebug().loaded && reviewDebug().sizes.prediction[0]===256", timeout=60000)
            assert page.evaluate("Object.values(videos).every(v=>v.videoWidth===256&&v.videoHeight===256)")
            assert page.locator("#description").inner_text() == c["plan"]["description"]
            if c["duration"] == 1:assert page.locator("#caseMetrics").inner_text().count("长度不足") == 2
            if c["duration"] == 3:assert page.locator("#caseMetrics").inner_text().count("长度不足") == 1
            browser_checks.append(dict(id=c["id"], native512=True, scored256=True, gt_and_speaker256=True))
            print("browser checked", len(browser_checks), c["id"], flush=True)
        # Real playback in the scored view as well as the native view.
        page.evaluate("show(0).catch(fail)");page.wait_for_function("reviewDebug().loaded")
        page.click("#play");page.wait_for_timeout(600);assert page.evaluate("reviewDebug().times.prediction") > .2
        page.click("#pause")
        page.select_option("#resolution", "native");page.wait_for_function("reviewDebug().loaded&&reviewDebug().sizes.prediction[0]===512")
        page.click("#reset");page.screenshot(path=str(OUT/"visualization_desktop.png"), full_page=True)
        page.set_viewport_size({"width":390,"height":844});page.wait_for_timeout(150)
        assert not page.evaluate("document.documentElement.scrollWidth>innerWidth"), "Mobile overflow"
        page.screenshot(path=str(OUT/"visualization_mobile.png"), full_page=True)
        page.reload();page.wait_for_function("reviewDebug().loaded", timeout=60000)
        assert page.evaluate("reviewDebug().sample") == d["cases"][0]["id"]
        assert not errors, errors
        browser.close()
    req = urllib.request.Request(URL+d["cases"][0]["native_video"], headers={"Range":"bytes=0-31"})
    with urllib.request.urlopen(req,timeout=10) as response:assert response.status == 206 and len(response.read()) == 32
    report = dict(passed=True, url=URL, cases=43, source_and_preview_checks=checks,
        browser_cases=browser_checks, actual_synchronized_playback=playback, both_resolution_views=True,
        seek_speed_audio_switch=True, audio_source_sha_verified_in_browser=True, mobile_no_overflow=True,
        byte_range=True, page_errors=errors, data_sha256=sha(PAGE/"data.json"), html_sha256=sha(PAGE/"index.html"))
    (OUT/"VISUALIZATION_BROWSER_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("Browser audit passed: all43 cases, both resolutions, source media, audio and playback controls")


if __name__ == "__main__":main()
