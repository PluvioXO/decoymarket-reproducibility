"""Capture public plan pages and pre-payment quote steps from a request manifest."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from scripts.collect_live import write_json


async def capture(browser, request):
    context = await browser.new_context(locale="en-US", timezone_id="America/Chicago")
    page = await context.new_page()
    page.set_default_timeout(6000)
    result = {"request": request, "time_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = await page.goto(request["url"], wait_until="domcontentloaded", timeout=16000)
        result["http_status"] = response.status if response else None
        for step in request.get("steps", []):
            locator = (page.get_by_text(step["name"], exact=True) if step["role"] == "text"
                       else page.get_by_role(step["role"], name=step["name"], exact=True))
            target = locator.nth(step.get("index", 0))
            if "zip" in step:
                if not re.fullmatch(r"\d{5}", step["zip"]) or step["role"] != "textbox":
                    raise ValueError("Only public ZIP inputs are supported")
                await target.fill(step["zip"])
            else:
                await target.click()
        await page.wait_for_timeout(request.get("settle_ms", 1600))
    except Exception as error:
        result["error"] = str(error).splitlines()[0]
    try:
        result.update(await page.evaluate("""() => {
          const visible = node => node.getClientRects().length > 0;
          return {url: location.href, title: document.title, body: document.body.innerText,
            links: [...document.querySelectorAll('a[href]')].filter(visible)
              .map(node => [node.innerText.trim(), node.href]),
            buttons: [...document.querySelectorAll('button, [role=button]')].filter(visible)
              .map(node => (node.getAttribute('aria-label') || node.innerText).trim()),
            fields: [...document.querySelectorAll('input')].filter(visible)
              .map(node => [node.getAttribute('aria-label'), node.placeholder, node.name, node.type])};
        }"""))
    except Exception as error:
        result["capture_error"] = str(error).splitlines()[0]
    finally:
        await context.close()
    return result


async def run(requests, output, chrome):
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=chrome, headless=True)
        limit = asyncio.Semaphore(4)

        async def one(request):
            async with limit:
                result = await capture(browser, request)
                write_json(output / f"{request['id']}.json", result)
                print(request["id"], result.get("http_status"), len(result.get("body", "")), flush=True)

        try:
            await asyncio.gather(*(one(request) for request in requests))
        finally:
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chrome", required=True)
    args = parser.parse_args()
    requests = json.loads(args.manifest.read_text())
    ids = [request["id"] for request in requests]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z0-9-]+", item) for item in ids):
        parser.error("Request IDs must be unique and contain lowercase letters, digits, and hyphens")
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(requests, args.output, args.chrome))


if __name__ == "__main__":
    main()
