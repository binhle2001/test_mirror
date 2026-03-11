# Hướng dẫn chạy thực nghiệm & tái lập kết quả (MIRROR / EMNLP 2025)

Tài liệu này hướng dẫn cách chạy phần code thực nghiệm trong repo này cho paper:
`c:\Users\binhltl1\Desktop\mirror\2025.emnlp-main.751v2.pdf` (ACL Anthology: https://aclanthology.org/2025.emnlp-main.751/).

Repo có 3 khối chính:
- `mirror/`: pipeline tổng hợp dữ liệu MIRROR (tạo hội thoại + sinh ảnh + lọc).
- `llm_therapist/`: baseline “text-only therapist” (LLM counselor + LLM virtual client).
- `mirror-llava/`: fine-tuning & inference framework (dựa trên LLaVA v1.5) + sinh ảnh trong lúc inference (PhotoMaker) + lọc chất lượng (API).

## 0) Khuyến nghị môi trường

Các script trong repo dùng `bash`, đường dẫn kiểu Linux (`/home/model/...`), và các dependency GPU (PyTorch/Deepspeed/vLLM/Diffusers). Vì vậy nên chạy trên:
- Linux (khuyến nghị), hoặc
- Windows + WSL2 (Ubuntu) + CUDA passthrough.

Python tối thiểu: 3.10+ (repo hiện chạy tốt với 3.11).

## 1) Chuẩn bị dữ liệu & thư mục

### 1.1 Mirror dataset (text/metadata)

Mirror dataset được cung cấp qua HuggingFace:
https://huggingface.co/datasets/multimodal-reframing/mirror

Vì ràng buộc giấy phép CelebA, dataset KHÔNG kèm ảnh. Bạn có 2 lựa chọn:
- (A) Tải text/metadata từ HF và tự tái tạo ảnh theo pipeline `mirror/step3` + `mirror/step4`.
- (B) Nếu paper/nhóm tác giả cung cấp bộ ảnh nội bộ, bạn cần tự đặt đúng cấu trúc thư mục mà script yêu cầu.

### 1.2 Các dataset nguồn (cần tự tải)

Tùy theo bạn muốn tái lập thí nghiệm nào, sẽ cần một hoặc nhiều nguồn:
- CelebA (ảnh gốc) để tái tạo ảnh biểu cảm.
- Cactus dataset (cho screenplay/dialogue seed).
- AnnoMI (nếu chạy demo `mirror_annomi_demo.sh`).

### 1.3 Cấu trúc thư mục mẫu (tham khảo)

Bạn có thể đặt dữ liệu ngoài repo, miễn là tham số `--data_path/--image_dir` trỏ đúng.
Ví dụ (Linux):

```text
/data/
  celeba/
    img_align_celeba/...
    celeba.csv
  cactus/
    cactus_data.csv
  mirror/
    mirror_data.csv
    images/...
```

## 2) Cài đặt dependency

### 2.1 PhotoMaker (cho sinh ảnh trong inference)

Tại root repo:

```bash
pip install -r photomaker_requirements.txt
```

Ngoài ra bạn cần các dependency nền tảng (tùy môi trường):
- PyTorch + CUDA
- transformers, accelerate
- deepspeed (cho training)
- vllm (cho API mô tả biểu cảm khuôn mặt)
- flask, requests

Repo này không kèm file `requirements.txt` tổng hợp, nên phần cài đặt cụ thể sẽ phụ thuộc vào stack bạn đang dùng (CUDA version, driver, GPU).

## 3) Pipeline tạo dữ liệu MIRROR (mirror/)

Chi tiết các bước đã có trong [mirror/README.md](file:///c:/Users/binhltl1/Desktop/mirror/mirror/README.md). Dưới đây là “bản chạy nhanh” theo đúng các entrypoints trong repo.

### Step 1: Facial annotation (DeepFace)

```bash
python -m mirror.step1.face_annot \
  --data_path ../data/celeba.csv \
  --img_data_dir /data/celeba/img_align_celeba \
  --save_path ../data/proc_celeba.csv \
  --drop_duplicated
```

### Step 2: Counseling screenplay generation (GPT)

```bash
python -m mirror.step2.run_step2 \
  --model gpt-4o-mini \
  --prompt_ver session_v3 \
  --data_path ../data/cactus_data.csv \
  --save_dir ../data/prompts/
```

Sau đó postprocess:

```bash
python -m mirror.step2.postprocess \
  --batch_input_dir ../data/prompts/ \
  --batch_output_dir ../data/batch_outputs/ \
  --save_path ../data/mirror_data.csv
```

### Step 3: Facial expression synthesis

Tạo prompt cho LLM:

```bash
python -m mirror.step3.preprocess_for_llm \
  --data_path ../data/mirror_data.csv \
  --model_name Meta-Llama-3-8B-Instruct \
  --save_path ../data/llama3_8b_prompt.jsonl
```

Chạy LLM để sinh mô tả biểu cảm:

```bash
python -m mirror.step3.annotate_llm \
  --prompt_path ../data/llama3_8b_prompt.jsonl \
  --model_name_or_path /model/Meta-Llama-3-8B-Instruct \
  --save_path ../data/llama3_8b_result.jsonl
```

Tạo prompt cho PhotoMaker:

```bash
python -m mirror.step3.preprocess_for_photomaker \
  --data_path ../data/mirror_data.csv \
  --llm_result_path ../data/llama3_8b_result.jsonl \
  --celeba_path ../data/proc_celeba.csv \
  --save_path photomaker_prompts/prompt.jsonl
```

Sinh ảnh:

```bash
python -m mirror.step3.run_step3 \
  --prompt_path photomaker_prompts/prompt.jsonl \
  --save_dir ../data/images/
```

### Step 4: Filtering (quality & safety)

```bash
python -m mirror.step4.safety --canary_dir ./step4/data/models/canary --data_path ../data/mirror_data.csv
python -m mirror.step4.clip --data_path ../data/mirror_data.csv --image_dir /data/images/
python -m mirror.step4.attr --image_dir /data/images/
python -m mirror.step4.nsfw --image_dir /data/images/
python -m mirror.step4.identity --data_path ../data/mirror_data_w_annot.csv --image_dir /data/images/
```

## 4) Baseline text-only therapist (llm_therapist/)

Các script chạy sẵn nằm ở [llm_therapist/run_scripts](file:///c:/Users/binhltl1/Desktop/mirror/llm_therapist/run_scripts/).

Ví dụ chạy GPT counselor:

```bash
cd llm_therapist
export GEMINI_API_KEY="..."
python -m src.run \
  --client_model_name gemini-3-flash-preview \
  --counselor_model_path gemini-3-flash-preview \
  --input_data ../../data/processed/test.csv
```

Output sẽ được ghi vào `--output_dir` (mặc định: `results_v3/`) theo format JSONL:
- `idx`
- `history` (list message)
- `total_tokens`

Lưu ý: [llm_therapist/src/run.py](file:///c:/Users/binhltl1/Desktop/mirror/llm_therapist/src/run.py) có `assert len(data_df) == 800`, nên file CSV input cần đúng 800 dòng theo format mà script mong đợi.

## 5) MIRROR-LLaVA training & inference (mirror-llava/)

Hướng dẫn gốc nằm ở [mirror-llava/README.md](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/README.md). Phần này bổ sung “đúng đường đi nước bước” theo các script trong `mirror-llava/scripts/`.

### 5.1 Training (LoRA, Deepspeed)

Script chính: [finetune_mirror_lora.sh](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/scripts/v1_5/finetune_mirror_lora.sh)

Bạn cần chuẩn bị data JSON theo format LLaVA ở:
- `mirror-llava/playground/data/train/mirror_base.json`
- `mirror-llava/playground/data/train/mirror_planning.json`
- `mirror-llava/playground/data/train/mirror_ec_planning.json`

Repo chỉ kèm file `*-sample.json` để minh họa format.

Chạy training (ví dụ 5 epoch, 7B, planning):

```bash
cd mirror-llava
bash scripts/v1_5/finetune_mirror_lora.sh 5 7b planning
```

Checkpoint output theo convention (script sẽ tự build path):
- `mirror-llava/checkpoints/llava-v1.5-7b-mirror_planning-task-lora-epoch5`

### 5.2 Chạy 2 API phục vụ inference (bắt buộc nếu bạn bật sinh ảnh)

Inference của MIRROR-LLaVA gọi 2 API:
- API mô tả biểu cảm (LLM): `api/description`
- API kiểm định ảnh (identity/quality): `api/verification`

#### (A) API mô tả biểu cảm (port 6000)

```bash
cd api/description
python app.py
```

Mặc định script dùng model path trong [api/description/app.py](file:///c:/Users/binhltl1/Desktop/mirror/api/description/app.py) (ví dụ `/home/model/Meta-Llama-3-8B-Instruct`) và cần GPU + vLLM.

Endpoint:
- `POST /describe` (form-data: `history`, `client_utt`) → JSON `{ "response": "..." }`

#### (B) API kiểm định ảnh (port 5000)

```bash
cd api/verification
python app.py --temp ./temp
```

Endpoint:
- `POST /verify` (multipart: `image`, `base_image`, form `statement`) → HTTP 200 nếu pass.

### 5.3 Inference / counseling simulation

Các script eval nằm ở [mirror-llava/scripts/v1_5/eval](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/scripts/v1_5/eval/).

Trước khi chạy, export env vars (không có dấu cách quanh dấu `=`):

```bash
export GEMINI_API_KEY="..."
export IMG_VERIFICATION_URL="http://127.0.0.1:5000"
export IMG_DESCRIPTION_URL="http://127.0.0.1:6000"
```

Chạy baseline LLaVA counseling:

```bash
cd mirror-llava
bash scripts/v1_5/eval/llava_counseling.sh 7b
```

Chạy MIRROR-LLaVA counseling (ví dụ epoch=5, 7b, planning):

```bash
cd mirror-llava
bash scripts/v1_5/eval/mirror_counseling.sh 5 7b planning
```

Lưu ý quan trọng:
- Script eval mặc định trỏ tới `./playground/data/eval/test.csv`, nhưng thư mục `eval/` không đi kèm trong repo. Bạn cần tự tạo `test.csv` theo format mà [llava/chat/model_chat.py](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/llava/chat/model_chat.py) đọc (các cột như `idx`, `personal_info`, `personality`, `distorted_thought`, `reason_for_seeking_counseling`, `img_path`, `dominant_gender`, ...).
- Kết quả inference được ghi JSONL vào `--output_dir` (script set mặc định `./playground/data/eval/answers`).
- Ảnh sinh ra được ghi vào `--image_save_dir` (script set theo model/config).

## 6) “Chứng minh” tái lập kết quả thực nghiệm (cách đối chiếu)

Paper báo cáo kết quả dựa trên:
- hội thoại được sinh ra bởi counselor model (text-only hoặc VLM),
- và (thường) một quy trình chấm điểm (human expert hoặc LLM-as-a-judge) cho các tiêu chí như counseling skills / therapeutic alliance trong bối cảnh resistance.

Trong repo này, phần “raw outputs” để đối chiếu thường là các file JSONL sinh ra bởi:
- `llm_therapist/src/run.py` (text-only baseline)
- `mirror-llava/llava/chat/model_chat.py` (VLM + sinh ảnh)

### 6.1 Artifact tối thiểu để chứng minh bạn chạy đúng pipeline

Sau khi chạy xong, bạn nên lưu lại các artifact sau (để đối chiếu và báo cáo):
- File JSONL output (mỗi dòng là 1 sample, có `idx` và `history`)
- Thư mục ảnh được sinh ra (`images_.../`) kèm prompt/negative_prompt đã lưu trong history
- Log console (tham số chạy: epoch/model-size/cot-type, seed, temperature)

### 6.2 Sanity check nhanh trên output JSONL

Các check sau giúp xác nhận bạn đang chạy đúng đường ống trước khi so sánh điểm số:
- Số dòng JSONL = số dòng trong `test.csv` (trừ các `idx` đã cache).
- `history` xen kẽ `Client`/`Counselor`, và mỗi turn của client có `image_path`.
- `total_tokens` tăng theo số lượt hội thoại (nếu dùng OpenAI client).
- Ảnh sinh ra có kích thước 336×336 (theo [image_gen.py](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/llava/chat/image_gen.py)).

### 6.3 Đối chiếu số liệu trong paper

Để đối chiếu đúng với các bảng trong paper:
1) Chạy đúng model/config (baseline LLaVA vs MIRROR-LLaVA base/planning/ec_planning; đúng epoch/model-size).
2) Dùng đúng tập test mà paper dùng (cùng sampling/seed nếu có).
3) Dùng cùng phương pháp chấm điểm (human expert rubric hoặc prompt đánh giá nếu paper dùng LLM).
4) Tổng hợp điểm theo đúng cách paper mô tả (mean, win-rate, v.v.).

Nếu bạn có sẵn script chấm điểm (không nằm trong repo), bạn chỉ cần trỏ vào JSONL output như ở mục 6.1.

## 7) Điểm vào code (entrypoints) để bạn debug nhanh

- Dataset pipeline: [mirror/README.md](file:///c:/Users/binhltl1/Desktop/mirror/mirror/README.md)
- Text-only baseline: [llm_therapist/src/run.py](file:///c:/Users/binhltl1/Desktop/mirror/llm_therapist/src/run.py)
- VLM counseling runner: [llava/chat/model_chat.py](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/llava/chat/model_chat.py)
- Counseling loop (virtual client + image gen): [llava/chat/therapy.py](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/llava/chat/therapy.py)
- Image generator (PhotoMaker + APIs): [llava/chat/image_gen.py](file:///c:/Users/binhltl1/Desktop/mirror/mirror-llava/llava/chat/image_gen.py)
- Description API: [api/description/app.py](file:///c:/Users/binhltl1/Desktop/mirror/api/description/app.py)
- Verification API: [api/verification/app.py](file:///c:/Users/binhltl1/Desktop/mirror/api/verification/app.py)
