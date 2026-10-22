# 操作手冊

這份是實際操作用的指令參考。概念說明看 `README.md`。

## 每次開工

```bash
conda activate InCoP
cd /nfs/home/charlie/InCoP
```

所有指令都要從 repo 根目錄執行 —— config 裡的資料路徑是相對路徑（`dataset/...`）。

### 先確認載入到正確的程式碼

```bash
python -c "import opencood; print(opencood.__file__)"
```

要印出 `/nfs/home/charlie/InCoP/opencood/__init__.py`。如果印出別的路徑（例如
`InCoP_supplement`），表示 `setup.py develop` 註冊的絕對路徑過期了，重跑一次：

```bash
python setup.py develop
```

### 確認在哪個 branch

```bash
git branch --show-current
```

`enable-pose-noise` 是解鎖 pose noise 的分支（見下方「Pose noise 實驗」）。
`main` 是作者原版。

## 確認 GPU 有沒有人在用

這是共用伺服器，**動手前一定要看**：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

找 `memory.used` 只有幾百 MiB、而且 `utilization` 是 0% 的。兩個條件都要符合——
有時記憶體佔著但 util 是 0，表示別人的 job 正在等待而不是結束了。

選定後用 `CUDA_VISIBLE_DEVICES=<編號>` 指定。下面範例都用 `N` 代表。

## 推論

拿現成權重跑評估，幾分鐘就有結果。

```bash
CUDA_VISIBLE_DEVICES=N python opencood/tools/inference_isaac.py \
  --model_dir opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42 \
  --fusion_method intermediate \
  --eval_split test \
  --num_workers 6
```

| 參數 | 說明 |
| --- | --- |
| `--model_dir` | 一個 run 目錄，要含 `config.yaml` 和 `.pth` |
| `--fusion_method` | `intermediate` / `late` / `no`（單機器人） |
| `--eval_split` | `test` 或 `val` |
| `--num_workers` | 資料載入執行緒，4–8 合理 |
| `--eval_dataset_dir` | 指定別的資料集目錄（例如 highoverlap） |

結果會印在 stdout，同時寫成 `<model_dir>/eval_*.yaml`。

> **注意**：如果檔名跟作者既有的評估檔相同，**會直接覆寫**。184 個原始檔都有
> `.orig` 備份，覆寫了可以 `cp <檔案>.orig <檔案>` 還原。

## 訓練

```bash
CUDA_VISIBLE_DEVICES=N python opencood/tools/train_isaac.py \
  --hypes_yaml opencood/hypes_yaml/isaacsim/BEVFusion_intermediate_ours/ours_pretrained_hospital.yaml \
  --num_workers 8
```

25 epochs 在 RTX 3090 上約需**一天**。結果寫到
`opencood/logs/<config名>_<時間戳>/`。

繼續未完成的訓練：加 `--model_dir <之前的 run 目錄>`。

## 背景執行：一定要用 setsid

直接用 `&` 丟背景，終端或 session 一關進程就會被殺掉。訓練跑了幾小時會全白費。

```bash
mkdir -p runlogs

setsid nohup env CUDA_VISIBLE_DEVICES=N python opencood/tools/train_isaac.py \
  --hypes_yaml <config路徑> --num_workers 8 \
  > runlogs/train.log 2>&1 < /dev/null &
disown
```

`setsid` 讓進程脫離 session 的 process group。驗證方式是看 PPID 是不是 1：

```bash
ps -u $USER -o pid,ppid,cmd | grep train_isaac | grep -v grep
```

PPID 為 1 就表示已經脫離，session 關掉也不受影響。

**不要在指令裡用 `grep` 過濾輸出**。出錯時錯誤訊息會被一起濾掉，變成什麼都看不到。
完整寫進 log，要看再 grep。

## 看進度

```bash
# 訓練：每 2 epoch 一次驗證
grep "At epoch" runlogs/train.log

# 推論結果
grep "Isaac multi-class" runlogs/*.log

# 看有沒有爆掉
grep -E "Traceback|Error|out of memory" runlogs/train.log
```

## 停掉

別人要用 GPU 時：

```bash
pkill -u $USER -f "train_isaac.py|inference_isaac.py"
```

然後確認清乾淨（這一步別省，dataloader worker 有時會留著）：

```bash
ps -u $USER -o pid,cmd | grep -E "train_isaac|inference_isaac" | grep -v grep
nvidia-smi
```

## Pose noise 實驗

需要在 `enable-pose-noise` branch 上（`main` 會因為防護而報
`ValueError: OPV2V is clean-pose only`）。

```bash
CUDA_VISIBLE_DEVICES=N python opencood/tools/inference_isaac.py \
  --model_dir <run目錄> --fusion_method intermediate --eval_split test \
  --pose_noise_pos_std 0.2 --pose_noise_rot_std 2.0
```

`pos_std` 單位公尺，`rot_std` 單位度。作者的慣例是 `rot = pos × 10`。

掃描多個噪聲水平的腳本在 `runlogs/sweep.sh`，直接改裡面的數列即可。

## 產生視覺化

不吃 GPU，從已錄好的資料渲染。

```bash
cd IsaacSimGenScript

# 單張圖
python visualize.py \
  --root Dataset/hospital_v1/dual_case_shadow/case_0/Jackal_R1 \
  --frame 0001166 \
  --panels rgb,bbox_3d,lidar_bev,lidar_projection,pose \
  --output_folder visualization_outputs

# 整段影片（指到 case 目錄會自動處理兩台機器人）
python visualize.py \
  --root Dataset/hospital_v1/dual_case_shadow/case_0 \
  --panels rgb,bbox_3d,lidar_bev,lidar_projection,pose \
  --output_folder visualization_outputs
```

可用的 panel 只有 `rgb`、`bbox_3d`、`lidar_bev`、`lidar_projection`、`pose`。
釋出的資料沒有 depth 和 2D 標註，那幾個 panel 會失敗。

## 資料格式轉換

資料已經轉好了，這節只在需要重建時用。

Hugging Face 上的 `InCoP_Dataset.tar.zst` 是**未轉換的 Isaac Sim 原始格式**，
README 沒提這一步。解壓到 `IsaacSimGenScript/Dataset/` 之後：

```bash
cd IsaacSimGenScript/opv2v_converter

# 一般的 train/validate/test（distance + shadow 條件）
python convert_isaac_to_opv2v_held_out_scene.py \
  --source /nfs/home/charlie/InCoP/IsaacSimGenScript/Dataset \
  --output-parent /nfs/home/charlie/InCoP/dataset \
  --case-set complementary --workers 16

# 高重疊評估集
python convert_isaac_to_opv2v_held_out_scene.py \
  --source /nfs/home/charlie/InCoP/IsaacSimGenScript/Dataset \
  --output-parent /nfs/home/charlie/InCoP/dataset \
  --case-set highoverlap --workers 16
```

一次產生三個 fold。`complementary` 約 75 GB、`highoverlap` 約 28 GB，各需數小時。
轉換器的預設參數就是作者當初的設定，不需要調整。

`--limit-frames 2` 可以做微型試轉（幾分鐘），用來確認目錄結構正確。

## 查作者的基準數字

```bash
# 列出所有 run
ls opencood/logs/ | grep -v '\.orig'

# 讀某個評估檔的 mAP
python -c "import yaml; print(yaml.safe_load(open('<評估檔路徑>'))['mAP'])"

# 找出用某個資料集的 run（run 名稱不可靠，要看 config 內容）
grep -l "dataset/real_world" opencood/logs/*/config.yaml
```

評估檔名的後綴代表實驗情境：

| 後綴 | 情境 |
| --- | --- |
| （無） | 標準 test split |
| `_val_` | 驗證集，不是 test |
| `_selected_` | 作者手選的子集，樣本少、數字不可比 |
| `_data_*_highoverlap_` | 高重疊評估集 |
| `_drop_partner_bev_` | 移除 partner 的 BEV |
| `_noise_t0p2_r2_` | pose 噪聲 0.2m / 2° |
| `_random_pair_` | 隨機配對 |

比較數字時記得排除 `_val_` 和 `_selected_`，它們跟標準 test 不可比。

## 常見問題

**`PermissionError: /nfs/home/uuugaga/.cache`**
作者的 run config 含硬編碼路徑。55 個已經修好，如果遇到新的：

```bash
sed -i 's|cache_dir: /nfs/home/uuugaga/.cache/huggingface/hub|cache_dir: null|' <config.yaml>
```

**`ValueError: OPV2V is clean-pose only`**
在 `main` branch 上跑 pose noise。切到 `enable-pose-noise`。

**`import opencood` 載到錯的目錄**
重跑 `python setup.py develop`。

**訓練莫名中斷**
沒用 `setsid`。見上方「背景執行」。

**mAP 是 0**
如果是訓練初期（epoch 10 以前）通常正常，模型還沒學到東西。如果到後期仍是
精確的 0，才需要懷疑座標系或 score threshold 的問題。

## 重要路徑

| 用途 | 路徑 |
| --- | --- |
| 作者的 CGRF hospital run | `opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42` |
| 從零訓練的 run | `opencood/logs/isaacsim_ours_pretrained_hospital_2026_09_23_11_22_53` |
| 主方法 config | `opencood/hypes_yaml/isaacsim/BEVFusion_intermediate_ours/ours_pretrained_hospital.yaml` |
| 互補配對的定義 | `IsaacSimGenScript/map/planning/dual.py` |
| 資料載入邏輯 | `opencood/data_utils/datasets/basedataset/isaacsim_basedataset.py` |
| 預訓練權重 | `checkpoints/single_agent/{hospital,office,warehouse}` |
