"""Download remaining ChestXray14 tar.gz files from NIH Box."""
import urllib.request
import os
from pathlib import Path

# All 12 download links (indexed 1-12)
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',  # 001
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',  # 002
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',  # 003
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',  # 004
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',  # 005
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',  # 006
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',  # 007
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',  # 008
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',  # 009
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',  # 010
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',  # 011
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz',  # 012
]

dest_dir = Path("data/raw/chestxray14/CXR8/images")

# Download images_005 through images_012 (indices 4-11)
for idx in range(4, 12):
    fn = f"images_{idx+1:03d}.tar.gz"
    filepath = dest_dir / fn
    
    if filepath.exists() and filepath.stat().st_size > 2_500_000_000:
        # Already exists and seems complete (>2.5 GB)
        print(f"SKIP {fn} (already exists, {filepath.stat().st_size / 1e9:.2f} GB)")
        continue
    
    print(f"Downloading {fn}...")
    try:
        urllib.request.urlretrieve(links[idx], str(filepath))
        size_gb = filepath.stat().st_size / 1e9
        print(f"  -> {size_gb:.2f} GB")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nAll downloads complete!")
