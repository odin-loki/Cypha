# enwik8 benchmark slice

`enwik8.8mb` is the first **8,388,608** bytes of the [enwik8](https://mattmahoney.net/dc/textdata.html) corpus.

Download (not committed — ~8 MB):

```bash
curl -fsSL -o /tmp/enwik8.zip https://data.deepai.org/enwik8.zip
unzip -p /tmp/enwik8.zip enwik8 | head -c 8388608 > bench/data/enwik8/enwik8.8mb
sha256sum bench/data/enwik8/enwik8.8mb
# 09f6dd7241a8ae21edfd6762f3c6712a1fd02f7f322c5e77cab8bb88f292ee8e
```

Used by `scripts/measure_enwik_gate24.sh`, `native/tools/cyphalm_hp_sku_measure`, and BPC gap reports.
