# 0. 目錄

# 3. 安裝與入門指南
## 3.1. 安裝 Conda
首先，安裝 Conda 環境管理器。推薦使用 Miniconda，因為它比 Anaconda 更輕量。可以從以下連結下載安裝：
[Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

[回到目錄](#0-目錄)


## 3.2. 建立 conda 環境
接著，使用以下命令建立一個新的 conda 環境並啟動他。此處以`XuanMultimodal`做為環境名稱，並安裝了 Python 3.10.9 版本。
```bash
conda create -n XuanMultimodal python=3.10.9
conda activate XuanMultimodal
```

[回到目錄](#0-目錄)


## 3.3. 安裝 git 與 pytorch
透過以下命令在環境中安裝 Git 和 PyTorch。這裡安裝的是 PyTorch 2.0.1 版本，並確保相容於 CUDA 11.8。
P.S. 如果你需要安裝最新版本的 PyTorch，可以使用註解掉的命令行。
```bash
conda install -c anaconda git
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

[回到目錄](#0-目錄)


## 3.4. 下載`XuanMultimodal`專案，並安裝 requirements 中的套件
下載以下連結的專案，並置於根目錄底下：
[XuanMultimodal 專案](https://github.com/shiuan89910/XuanMultimodal/archive/refs/heads/main.zip)
>根目錄的位置
>Windows: C:\Users\使用者名稱
>Ubuntu: /home/使用者名稱

再透過以下命令進入專案目錄，此處為`XuanMultimodal`，並安裝所有依賴。
```bash
cd XuanMultimodal
pip install -r requirements.txt
```

[回到目錄](#0-目錄)


## 3.5. 下載 Speech-to-Text (STT) 與 Text-to-Speech (TTS) 模型
關於開源 STT 與 TTS 模型的權重下載連結。您可以透過 Hugging Face 平台獲取這些資源，進行研究或開發工作。

[回到目錄](#0-目錄)

### 3.5.1. 權重下載
開源 STT 與 TTS 模型的權重可以透過 [Hugging Face](https://huggingface.co/models) 進行下載。Hugging Face 提供了廣泛的預訓練模型，支持各種自然語言處理任務。

[回到目錄](#0-目錄)

### 3.5.2. 授權條款注意事項
在使用本項目提供的開源 Speech-to-Text (STT) 與 Text-to-Speech (TTS) 模型或任何其他資源時，**強烈建議**用戶仔細查看每個模型或資源的授權條款。不同的模型和資源可能會有不同的授權要求，這可能會影響您使用這些資源的方式。
請前往相應的平台或資源頁面，如 [Hugging Face 模型庫](https://huggingface.co/models)，以獲取詳細的授權信息。確保您的使用方式符合這些授權條款的規定，以避免侵犯著作權或其他法律問題。
使用這些資源時，如果有任何疑問，建議咨詢法律專業人士或直接與模型/資源的提供者聯繫以獲取進一步的指導。

[回到目錄](#0-目錄)

### P.S. 下載的模型請置於`XuanMultimodal`目錄底下

[回到目錄](#0-目錄)


## 3.6. 啟動 .py 檔
### 3.6.1. STT，執行以下命令
```bash
# 將手機錄製的 M4A 檔，命名為 test.m4a
# 並置於 XuanMultimodal 目錄底下

python speech_to_text.py
```

[回到目錄](#0-目錄)

### 3.6.2. TTS，執行以下命令
```bash
# 將文本以 TXT 檔儲存，並命名為 test.txt
# 並置於 XuanMultimodal 目錄底下

python txet_to_speech.py
```

[回到目錄](#0-目錄)



# 5. 致謝
本專案的參考來源，特此致謝
[openai](https://github.com/openai/whisper)
[Hugging Face 的 openai/whisper-medium]([https://huggingface.co/coqui/XTTS-v2](https://huggingface.co/openai/whisper-medium))
[coqui-ai](https://github.com/coqui-ai/TTS)
[Hugging Face 的 coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
[haotian-liu](https://github.com/haotian-liu/LLaVA)

[回到目錄](#0-目錄)
