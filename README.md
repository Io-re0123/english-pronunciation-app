# english-pronunciation-app
英語の発音を評価するブラウザアプリを実装しました。

## 内容
Pythonスクリプトは、**「AI英語発音コーチ」**というウェブアプリです。
**できること:**
1.  **お手本表示:** AIが英語の練習用例文を出してくれます。
2.  **録音:** あなたがその例文を読み上げると、声を録音します。
3.  **発音チェック:** AIがあなたの発音を聞いて、どれくらい上手に言えたか点数をつけ、どこを直せば良くなるかアドバイスをくれます。
**仕組みのポイント:**
*   **例文作り & アドバイス:** Googleの賢いAI (Gemini) が担当。
*   **聞き取り:** Whisperという音声認識AIが、あなたの英語を聞き取ります。
これらを使って、手軽に英語の発音練習ができるようになっています。

## 主なライブラリ
**Webアプリケーション**
・[Streamlit](https://streamlit.io/ "Streamlit")
・[audio-recorder-streamlit](https://pypi.org/project/audio-recorder-streamlit/ "audio_recorder_streamlit")

**ML**
・[PyTorch](https://pytorch.org/ "PyTorch")
・[Transformers](https://github.com/huggingface/transformers "transformers")

**音声処理**
・[librosa](https://librosa.org/ "librosa")
・[SoundFile](https://pypi.org/project/SoundFile/ "soundfile")

**データ操作**
・[NumPy](https://numpy.org/ "NumPy")
・[math](https://docs.python.org/3/library/math.html "math")

**評価指標**
・[jiwer](https://pypi.org/project/jiwer/ "jiwer")

**AIサービス**
・[google-generativeai](https://pypi.org/project/google-generativeai/ "google-generativeai")

**システムユーティリティ**
・[os](https://docs.python.org/3/library/os.html "os")
・[io](https://docs.python.org/3/library/io.html "io")