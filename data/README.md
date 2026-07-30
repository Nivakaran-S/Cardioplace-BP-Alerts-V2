# HEMOBP corpus

Not committed. `vip.csv` alone is 251 MB and there is no `.gitattributes` in this repo, so
committing it would put a quarter-gigabyte blob into every clone permanently.

Download the three CSVs from Figshare into this directory:

| File | Size | Figshare id | Contents |
|---|---|---|---|
| `vip.csv` | 251 MB | 15142157 | 4,366,298 intradialytic vital-sign readings |
| `d1.csv`  | 11 MB  | 15142151 | 165,986 dialysis session records |
| `idp.csv` | 35 KB  | 15142154 | 1,072 patient demographics |

```bash
curl -L -o data/vip.csv https://ndownloader.figshare.com/files/15142157
curl -L -o data/d1.csv  https://ndownloader.figshare.com/files/15142151
curl -L -o data/idp.csv https://ndownloader.figshare.com/files/15142154
```

The ingest reconciles what it derives against the published counts and fails the run if the
ratio leaves its band, so a truncated or partial download is caught rather than silently
trained on. See `src/components/data_ingestion.py`.
