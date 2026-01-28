import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from colorsys import rgb_to_hsv

from pathlib import Path
import requests
import pyucalgarysrs

class TrexKeogramDownloader:
    def __init__(
        self,
        save_dir="./keograms",
        stream=2,
        dataset="TREX_RGB_HOURLY_KEOGRAM",
        base_http="https://data.phys.ucalgary.ca/sort_by_project/TREx/RGB",
        headers=None,
        timeout=60,
    ):


        self.save_dir = None if save_dir is None else Path(save_dir)
        self.stream = int(stream)
        self.dataset = dataset
        self.base_http = base_http.rstrip("/")
        self.timeout = int(timeout)

        default_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                "Gecko/20100101 Firefox/128.0"
            ),
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://data.phys.ucalgary.ca/",
            "Connection": "keep-alive",
        }
        if headers:
            default_headers.update(headers)

        self.session = requests.Session()
        self.session.headers.update(default_headers)

        self.srs = pyucalgarysrs.PyUCalgarySRS()

    def _date_parts(self, date_str):
        y, m, d = date_str.split("-")
        return y, m, d, f"{y}{m}{d}"

    def _http_fallback_url(self, date_str, hour, device):
        y, m, d, ymd = self._date_parts(date_str)
        hh = f"{hour:02d}"
        filename = f"{ymd}_{hh}_{device}_full-keogram.jpg"
        return f"{self.base_http}/stream{self.stream}/{y}/{m}/{d}/{device}/ut{hh}/{filename}"

    def _api_pick_url(self, date_str, hour, site_uid, device):
        import datetime as dt

        y, m, d, _ = self._date_parts(date_str)
        hh = f"{hour:02d}"
        start = dt.datetime(int(y), int(m), int(d), int(hour), 0, 0)
        end = start + dt.timedelta(hours=1)

        listing = self.srs.data.get_urls(self.dataset, start, end, site_uid=site_uid)
        urls = list(getattr(listing, "urls", []))

        target_suffix = f"_{hh}_{device}_full-keogram.jpg"
        picked = [u for u in urls if u.endswith(target_suffix)]
        if not picked:
            picked = [u for u in urls if (device in u and u.endswith("_full-keogram.jpg"))]

        return picked[0] if picked else None, urls

    def fetch_one(self, date_str, hour, human_name, site_uid, device, save=True):
        """
        Returns a dict with:
          - ok (bool)
          - source ("API" or "HTTP-fallback")
          - url
          - local_path (or None)
          - error (or None)
          - content_type (or None)
          - bytes (int or None)
        """
        from pathlib import Path

        # 1) try API
        api_url = None
        api_urls_count = None
        api_error = None

        try:
            api_url, api_urls = self._api_pick_url(date_str, hour, site_uid, device)
            api_urls_count = len(api_urls)
        except Exception as e:
            api_error = repr(e)

        # 2) decide URL
        if api_url:
            url = api_url
            source = "API"
        else:
            url = self._http_fallback_url(date_str, hour, device)
            source = "HTTP-fallback"

        # 3) download
        try:
            r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            if r.status_code == 404:
                return {
                    "ok": False,
                    "human_name": human_name,
                    "site_uid": site_uid,
                    "device": device,
                    "date": date_str,
                    "hour": hour,
                    "source": source,
                    "url": url,
                    "local_path": None,
                    "error": f"404 not found via {source}",
                    "content_type": (r.headers.get("Content-Type") or "").lower(),
                    "bytes": 0,
                    "api_urls_count": api_urls_count,
                    "api_error": api_error,
                }
            r.raise_for_status()

            ctype = (r.headers.get("Content-Type") or "").lower()
            if "image" not in ctype:
                snippet = r.content[:160]
                return {
                    "ok": False,
                    "human_name": human_name,
                    "site_uid": site_uid,
                    "device": device,
                    "date": date_str,
                    "hour": hour,
                    "source": source,
                    "url": url,
                    "local_path": None,
                    "error": f"unexpected content-type {ctype!r} via {source} (first160={snippet!r})",
                    "content_type": ctype,
                    "bytes": len(r.content),
                    "api_urls_count": api_urls_count,
                    "api_error": api_error,
                }

            local_path = None
            if save and self.save_dir is not None:
                hh = f"{hour:02d}"
                local_path = self.save_dir / date_str / f"ut{hh}" / Path(url).name
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(r.content)

            return {
                "ok": True,
                "human_name": human_name,
                "site_uid": site_uid,
                "device": device,
                "date": date_str,
                "hour": hour,
                "source": source,
                "url": url,
                "local_path": str(local_path) if local_path else None,
                "error": None,
                "content_type": ctype,
                "bytes": len(r.content),
                "api_urls_count": api_urls_count,
                "api_error": api_error,
                "content": r.content,  # keep if caller wants to display without re-reading
            }

        except Exception as e:
            return {
                "ok": False,
                "human_name": human_name,
                "site_uid": site_uid,
                "device": device,
                "date": date_str,
                "hour": hour,
                "source": source,
                "url": url,
                "local_path": None,
                "error": repr(e),
                "content_type": None,
                "bytes": None,
                "api_urls_count": api_urls_count,
                "api_error": api_error,
            }

    def fetch_many(self, date_str, hour, sites, save=True, display=True):
        """
        sites: list of tuples (human_name, site_uid, device)
        Returns list of result dicts.
        """
        from io import BytesIO
        from PIL import Image
        import matplotlib.pyplot as plt

        results = []
        images = []

        for human_name, site_uid, device in sites:
            res = self.fetch_one(date_str, hour, human_name, site_uid, device, save=save)
            results.append(res)

            if res["ok"]:
                try:
                    img = Image.open(BytesIO(res["content"])).convert("RGB")
                    images.append((human_name, device, img))
                except Exception:
                    pass

        if display and images:
            fig, axes = plt.subplots(len(images), 1, figsize=(12, 3.5 * len(images)))
            if len(images) == 1:
                axes = [axes]
            for ax, (human_name, device, img) in zip(axes, images):
                ax.imshow(img)
                ax.set_title(f"{human_name} — {device} — {date_str} UT{hour:02d}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()

        return results


@dataclass
class KeogramThresholds:
    green_s_min: float = 0.20
    green_v_min: float = 0.15
    green_h_min: float = 80.0
    green_h_max: float = 170.0

    red_s_min: float = 0.15
    red_v_min: float = 0.10
    red_h_min: float = 330.0
    red_h_max: float = 30.0

    white_v_min: float = 0.75
    white_s_max: float = 0.20

    black_v_max: float = 0.08
    black_s_max: float = 0.10


@dataclass
class KeogramCrop:
    top: float = 0.12
    bottom: float = 0.12
    left: float = 0.02
    right: float = 0.02


class KeogramScorer:
    def __init__(self, crop: Optional[KeogramCrop] = None, thresholds: Optional[KeogramThresholds] = None):
        self.crop = crop or KeogramCrop()
        self.th = thresholds or KeogramThresholds()

    def _crop_data_band(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        x0 = int(w * self.crop.left)
        x1 = int(w * (1 - self.crop.right))
        y0 = int(h * self.crop.top)
        y1 = int(h * (1 - self.crop.bottom))
        return img.crop((x0, y0, x1, y1))

    def _to_hsv(self, img: Image.Image):
        arr = np.asarray(img).astype(np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        hsv = np.array([rgb_to_hsv(*px) for px in flat], dtype=np.float32)
        h = hsv[:, 0].reshape(arr.shape[:2]) * 360.0  # degrees
        s = hsv[:, 1].reshape(arr.shape[:2])
        v = hsv[:, 2].reshape(arr.shape[:2])
        return h, s, v

    def score_image(self, img: Union[Image.Image, Path, str], crop: bool = True) -> Dict[str, float]:
        if isinstance(img, (Path, str)):
            img = Image.open(img).convert('RGB')
        if crop:
            img = self._crop_data_band(img)

        h, s, v = self._to_hsv(img)

        # green mask
        green_mask = (s > self.th.green_s_min) & (v > self.th.green_v_min) & (h >= self.th.green_h_min) & (h <= self.th.green_h_max)
        green_cov = float(green_mask.mean())
        green_int = float((v[green_mask] * s[green_mask]).mean()) if green_cov > 0 else 0.0
        green_score = green_cov * green_int

        # red mask (wrap hue)
        red_mask = (s > self.th.red_s_min) & (v > self.th.red_v_min) & ((h >= self.th.red_h_min) | (h <= self.th.red_h_max))
        red_cov = float(red_mask.mean())
        red_int = float((v[red_mask] * s[red_mask]).mean()) if red_cov > 0 else 0.0
        red_score = red_cov * red_int

        # white mask (moon/cloud)
        white_mask = (v > self.th.white_v_min) & (s < self.th.white_s_max)
        white_cov = float(white_mask.mean())
        white_brightness = float(v[white_mask].mean()) if white_cov > 0 else 0.0
        white_score = white_cov * white_brightness

        # black mask (night / no signal)
        black_mask = (v < self.th.black_v_max) & (s < self.th.black_s_max)
        black_cov = float(black_mask.mean())
        black_darkness = float((1.0 - v[black_mask]).mean()) if black_cov > 0 else 0.0
        black_score = black_cov * black_darkness

        aurora_score = max(green_score, red_score) * (1.0 - min(white_cov, 1.0))

        return {
            'green_score': green_score,
            'green_coverage': green_cov,
            'green_intensity': green_int,
            'red_score': red_score,
            'red_coverage': red_cov,
            'red_intensity': red_int,
            'white_score': white_score,
            'white_coverage': white_cov,
            'white_brightness': white_brightness,
            'black_score': black_score,
            'black_coverage': black_cov,
            'black_darkness': black_darkness,
            'aurora_score': aurora_score,
        }

    def score_folder(self, root: Path, save_json: bool = False, json_dir: Optional[Path] = None) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        root = Path(root)
        for p in sorted(root.rglob('*.jpg')):
            try:
                scores = self.score_image(p, crop=True)
                scores['path'] = str(p)
                rows.append(scores)

                if save_json:
                    out_dir = json_dir or p.parent
                    out_dir = Path(out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / (p.stem + '.json')
                    out_path.write_text(json.dumps(scores, indent=2))
            except Exception as e:
                print('failed', p, e)
        return rows
