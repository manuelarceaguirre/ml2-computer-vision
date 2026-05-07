# Colab GPU run

Use this to run the experiment grid on a Colab GPU while keeping code in GitHub and the assignment data in Google Drive.

## 1. Colab setup cell

```bash
!nvidia-smi
!git clone https://github.com/manuelarceaguirre/ml2-computer-vision.git
%cd ml2-computer-vision
!python -m pip install -q -r requirements-colab.txt
```

## 2. Get the data from Drive

Recommended: upload `data.zip` to Google Drive, then either mount Drive or download by file ID.

### Option A: mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then copy your uploaded zip into the repo:

```bash
!cp "/content/drive/MyDrive/data.zip" ./data.zip
!unzip -q -o data.zip
```

### Option B: download by Drive file ID

```bash
FILE_ID="PASTE_DRIVE_FILE_ID_HERE"
python - <<'PY'
import os
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

auth.authenticate_user()
file_id = os.environ['FILE_ID']
service = build('drive', 'v3')
request = service.files().get_media(fileId=file_id)
with open('data.zip', 'wb') as f:
    downloader = MediaIoBaseDownload(f, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"download {int(status.progress() * 100)}%" if status else "download starting")
PY
unzip -q -o data.zip
```

## 3. Run the full grid on CUDA

```bash
chmod +x gptgrid
GPTGRID_BATCH_SIZE=128 GPTGRID_TEACHER_BATCH_SIZE=32 ./gptgrid
```

If Colab runs out of memory, resume with smaller batches:

```bash
GPTGRID_BATCH_SIZE=64 GPTGRID_TEACHER_BATCH_SIZE=16 ./gptgrid
```

Outputs:

- `gptgrid_results.json`
- `gptgrid_results.jsonl`
- `log.txt`
- current best submitted student: `model.pt`
