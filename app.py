import streamlit as st
from audio_recorder_streamlit import audio_recorder
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import numpy as np
import math
from jiwer import wer
import google.generativeai as genai
import os
import io
import soundfile as sf

# Gemini APIキー
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    st.error("環境変数 GOOGLE_API_KEY が設定されていません。アプリケーションを実行する前に設定してください。")
    st.stop()

MODEL_CHECKPOINT = "openai/whisper-base.en"
WHISPER_MAX_LENGTH = 448
SIGMOID_A = 0.5
SIGMOID_B = 6
DEFAULT_SAMPLE_RATE = 16000

_model_scorer = None
_processor_scorer = None
_device_scorer = None

def _initialize_scoring_model_and_processor():
    global _model_scorer, _processor_scorer, _device_scorer
    if _model_scorer is None or _processor_scorer is None:
        _device_scorer = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            _processor_scorer = WhisperProcessor.from_pretrained(MODEL_CHECKPOINT)
            _model_scorer = WhisperForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT).to(_device_scorer)
            _model_scorer.eval()
            return True
        except Exception as e:
            st.error(f"Whisperモデル/プロセッサのロード中にエラーが発生しました: {e}")
            return False
    return True

def preprocess_audio_for_scoring(audio_path_or_data, sampling_rate_in=None):
    if not _initialize_scoring_model_and_processor():
        raise Exception("Scoring model could not be initialized.")
    target_sr = DEFAULT_SAMPLE_RATE
    if isinstance(audio_path_or_data, str):
        try:
            audio, sr_orig = librosa.load(audio_path_or_data, sr=None, mono=True)
        except Exception as e:
            st.error(f"音声ファイル {audio_path_or_data} のロード中にエラー: {e}")
            raise
    elif isinstance(audio_path_or_data, np.ndarray):
        if sampling_rate_in is None:
            raise ValueError("音声データがnumpy配列の場合、sampling_rate_inの指定が必要です。")
        audio = audio_path_or_data
        sr_orig = sampling_rate_in
        if audio.ndim > 1 and audio.shape[1] > 1:
             audio = np.mean(audio, axis=1)
        elif audio.ndim > 1 and audio.shape[1] == 1:
            audio = audio.squeeze()
        elif audio.ndim == 1:
            pass
        else:
            st.error(f"予期しないNumpy配列の形状です: {audio.shape}")
            raise ValueError(f"予期しないNumpy配列の形状: {audio.shape}")
    else:
        raise ValueError("audio_path_or_data はファイルパス(str)またはnumpy配列とsampling_rate_inである必要があります。")
    if sr_orig != target_sr:
        try:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=target_sr)
        except Exception as e:
            st.error(f"音声のリサンプリング中にエラー: {e}")
            raise
    try:
        input_features = _processor_scorer(audio, sampling_rate=target_sr, return_tensors="pt").input_features
        return input_features.to(_device_scorer)
    except Exception as e:
        st.error(f"Whisperプロセッサでの音声処理中にエラー: {e}")
        raise

def calculate_entropy(probabilities):
    probabilities = torch.where(probabilities == 0, torch.tensor(1e-12, device=probabilities.device), probabilities)
    return -torch.sum(probabilities * torch.log(probabilities), dim=-1)

def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

def calculate_fluency_score(perplexity, a=SIGMOID_A, b=SIGMOID_B):
    if math.isnan(perplexity) or math.isinf(perplexity):
        return 0.0
    return 100 / (1 + math.exp(a * (perplexity - b)))

def evaluate_speech(audio_path_or_data, ground_truth_text, sampling_rate_in=None):
    if not _initialize_scoring_model_and_processor():
        return {"total_entropy": float('nan'), "total_perplexity": float('nan'),"total_fluency_score": 0.0, "decoded_text": "エラー: 評価モデルの準備に失敗しました。","wer": 1.0, "words_with_metrics": []}
    try:
        input_features = preprocess_audio_for_scoring(audio_path_or_data, sampling_rate_in)
    except Exception as e:
        return {"total_entropy": float('nan'), "total_perplexity": float('nan'),"total_fluency_score": 0.0, "decoded_text": "エラー: 音声処理に失敗しました。","wer": 1.0, "words_with_metrics": []}
    with torch.no_grad():
        try:
            outputs = _model_scorer.generate(input_features=input_features,return_dict_in_generate=True,output_scores=True,max_length=WHISPER_MAX_LENGTH,num_beams=1,do_sample=False)
        except Exception as e:
            st.error(f"モデルによる生成処理中にエラー: {e}")
            return {"total_entropy": float('nan'), "total_perplexity": float('nan'),"total_fluency_score": 0.0, "decoded_text": "エラー: モデルによる音声認識に失敗しました。","wer": 1.0, "words_with_metrics": []}
        predicted_ids_full = outputs.sequences
        stacked_scores = torch.stack(outputs.scores, dim=1)
        probabilities = torch.softmax(stacked_scores, dim=-1)
    squeezed_probabilities = probabilities.squeeze(0)
    if squeezed_probabilities.ndim == 1 and squeezed_probabilities.shape[0] == _model_scorer.config.vocab_size:
        squeezed_probabilities = squeezed_probabilities.unsqueeze(0)
    if squeezed_probabilities.numel() == 0 or squeezed_probabilities.ndim < 2:
        token_entropies = torch.tensor([], device=_device_scorer)
    else:
        token_entropies = calculate_entropy(squeezed_probabilities)
    token_perplexities = torch.exp(token_entropies) if token_entropies.numel() > 0 else torch.tensor([], device=_device_scorer)
    decoded_preds_list = _processor_scorer.batch_decode(predicted_ids_full, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_preds = decoded_preds_list[0].strip() if decoded_preds_list else ""
    token_ids = predicted_ids_full[0].tolist()
    words_with_metrics = []
    for i_token_loop in range(len(token_perplexities)):
        if i_token_loop + 1 < len(token_ids):
            current_token_id_in_sequence = token_ids[i_token_loop+1]
            word = _processor_scorer.decode([current_token_id_in_sequence], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if word.strip():
                word_perplexity_val = token_perplexities[i_token_loop].item()
                word_fluency_score_val = calculate_fluency_score(word_perplexity_val)
                words_with_metrics.append({"word": word.strip(),"entropy": token_entropies[i_token_loop].item(),"perplexity": word_perplexity_val,"fluency_score": word_fluency_score_val})
    if len(token_entropies) > 0:
        total_entropy_val = token_entropies.mean().item()
        total_perplexity_val = torch.exp(torch.tensor(total_entropy_val, device=_device_scorer)).item()
        total_fluency_score_val = calculate_fluency_score(total_perplexity_val)
    else:
        total_entropy_val = float('nan')
        total_perplexity_val = float('nan')
        total_fluency_score_val = 0.0
    wer_val = calculate_wer(ground_truth_text.lower().strip(), decoded_preds.lower().strip())
    return {"total_entropy": total_entropy_val,"total_perplexity": total_perplexity_val,"total_fluency_score": total_fluency_score_val,"decoded_text": decoded_preds,"wer": wer_val,"words_with_metrics": words_with_metrics}


# Gemini API
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model_feedback = genai.GenerativeModel('gemini-2.0-flash-lite')

def generate_sentence_for_pronunciation_practice(topic="日常会話", difficulty="初級から中級", word_count=(7, 15)):
    prompt = f"""日本人英語学習者のための、発音練習に適した自然な英語の文章を1つ生成してください。文章は以下の条件を満たすようにしてください：- トピック: 「{topic}」に関連するもの - 難易度: {difficulty}レベルの学習者向け - 単語数: {word_count[0]}語から{word_count[1]}語程度 - 一般的で平易な語彙を使用し、複雑すぎる文法や特殊なスラングは避ける - 完全な一つの文章であること 生成するのは英語の文章のみとし、「Here is a sentence:」のような前置きは不要です。例: I enjoy listening to music in my free time."""
    try:
        response = gemini_model_feedback.generate_content(prompt)
        sentence = response.text.strip()
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if not sentence: return "I would like to try that new cafe."
        return sentence
    except Exception as e:
        st.warning(f"Geminiによる教師文生成エラー: {e}")
        return "The park is a beautiful place to relax."

# フィードバック生成プロンプト
def generate_simplified_feedback(teacher_sentence, whisper_results):
    prompt = f"""
    あなたは英語学習者の発音を評価するAIアシスタントです。
    以下の情報を基に、学習者へのフィードバックを**厳密に指定された形式で**日本語で生成してください。

    ## 入力情報
    お手本: "{teacher_sentence}"
    学習者の発音 (AI認識): "{whisper_results['decoded_text']}"
    単語誤り率 (WER): {whisper_results['wer']:.2%} (0%に近いほど良い)
    総合流暢さスコア (AIによる推定): {whisper_results['total_fluency_score']:.0f}/100 (100に近いほど良い)

    ## 出力形式 (この形式を厳守してください)
    総合スコア: [0-100の整数値]/100 点

    ワンポイントアドバイス:
    [ここに1〜2点の簡潔なアドバイス]

    ## 指示
    - 総合スコアは、単語誤り率と流暢さスコアを総合的に判断して、0から100の整数で採点してください。
    - ワンポイントアドバイスは、学習者がやる気を損なわないように良いところを誉めつつ、次に何を意識すれば良いか、明確な行動指針を提示してください。
    - 余計な挨拶や前置き、例示の繰り返しは一切含めないでください。
    """
    try:
        response = gemini_model_feedback.generate_content(prompt)
        # レスポンス内容をデバッグ用に表示してみる
        # st.write("--- Gemini Raw Response ---")
        # st.text(response.text)
        # st.write("--- End Gemini Raw Response ---")
        return response.text.strip()
    except Exception as e:
        st.warning(f"Geminiによるフィードバック生成エラー: {e}")
        return "総合スコア: 判定不能/100 点\n\nワンポイントアドバイス:\nフィードバックの生成に失敗しました。もう一度お試しください。"

# Streamlit アプリのUI
st.set_page_config(page_title="英語発音練習アプリ", layout="centered") # 中央揃えに変更
st.title("AI英語発音コーチ 🎤")

# モデルのロード状態をセッション状態で管理
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    with st.spinner("AIモデルを準備中..."):
        if _initialize_scoring_model_and_processor():
            st.session_state.model_loaded = True
        else:
            st.error("モデルの準備に失敗しました。アプリをリロードしてください。")
            st.stop()

# セッションステートの初期化 (現在のステップなど)
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1 # 1: 文生成, 2: 録音, 3: 結果表示
if 'teacher_sentence' not in st.session_state:
    st.session_state.teacher_sentence = ""
if 'feedback' not in st.session_state: # 簡略化されたフィードバック用
    st.session_state.feedback = ""
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'audio_bytes_data' not in st.session_state:
    st.session_state.audio_bytes_data = None

# ステップごとの画面定義

def step1_generate_sentence():
    st.header("ステップ1: 練習する英文を確認")
    st.write("下のボタンを押して、練習する英文を生成しましょう。")

    if st.button("教師文を生成する", key="generate_sentence_btn", type="primary", use_container_width=True):
        with st.spinner("練習文を生成中..."):
            st.session_state.teacher_sentence = generate_sentence_for_pronunciation_practice(
                topic="簡単な日常会話",
                difficulty="初級"
            )
            # リセット処理
            st.session_state.feedback = ""
            st.session_state.analysis_results = None
            st.session_state.audio_bytes_data = None
            st.session_state.current_step = 2 # 次のステップへ
            st.rerun() # 画面を再描画してステップ2を表示

    if st.session_state.teacher_sentence and st.session_state.current_step != 1 : # 生成済みだがまだステップ1にいる場合(ほぼないはず)
        st.info(f"**お題:**\n\n## \"{st.session_state.teacher_sentence}\"")
        if st.button("このお題で練習開始 →", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()

def step2_record_audio():
    st.header("ステップ2: お手本を発音して録音")
    if not st.session_state.teacher_sentence:
        st.warning("まずステップ1でお題を生成してください。")
        if st.button("← ステップ1に戻る", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
        return

    st.info(f"**お題:**\n\n## \"{st.session_state.teacher_sentence}\"")
    st.markdown("---")
    st.write("下のマイクアイコンをクリックして、上の英文を発音してください。")

    recorded_audio_bytes = audio_recorder(
        text="▶️ クリックして録音開始",
        recording_color="#e87070",
        neutral_color="#6aa36f",
        icon_size="3x", # 少し大きく
        pause_threshold=2.0, # 無音停止までの秒数 (短めに)
        sample_rate=DEFAULT_SAMPLE_RATE
    )

    if recorded_audio_bytes:
        st.session_state.audio_bytes_data = recorded_audio_bytes
        st.audio(st.session_state.audio_bytes_data, format='audio/wav')
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("← 録り直す", use_container_width=True):
                st.session_state.audio_bytes_data = None # 録音データをクリア
                st.rerun()
        with col_nav2:
            if st.button("これで評価する →", type="primary", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
    
    st.markdown("---")
    if st.button("別の練習文にする (ステップ1へ)", use_container_width=True):
        st.session_state.current_step = 1
        st.session_state.teacher_sentence = "" # 練習文もクリア
        st.rerun()


def step3_show_results():
    st.header("ステップ3: AIコーチからのフィードバック")
    if not st.session_state.audio_bytes_data or not st.session_state.teacher_sentence:
        st.warning("録音データまたは練習文がありません。ステップ1からやり直してください。")
        if st.button("ステップ1に戻る", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
        return

    # 評価処理（一度だけ実行されるようにする）
    if st.session_state.analysis_results is None:
        with st.spinner("あなたの発音を評価中です...しばらくお待ちください。"):
            try:
                audio_data_io = io.BytesIO(st.session_state.audio_bytes_data)
                audio_np, sr = sf.read(audio_data_io, dtype='float32')
                
                analysis = evaluate_speech(
                    audio_path_or_data=audio_np,
                    sampling_rate_in=sr,
                    ground_truth_text=st.session_state.teacher_sentence
                )
                st.session_state.analysis_results = analysis

                # フィードバックを生成
                feedback_text = generate_simplified_feedback(
                    st.session_state.teacher_sentence,
                    analysis
                )
                st.session_state.feedback = feedback_text
                st.success("評価が完了しました！")
            except Exception as e:
                st.error(f"評価処理中にエラーが発生しました: {e}")
                st.error("録音データが短すぎるか、形式に問題がある可能性があります。")
                if st.button("やり直す (ステップ2へ)", use_container_width=True):
                    st.session_state.current_step = 2
                    st.session_state.analysis_results = None # 結果をクリア
                    st.session_state.feedback = ""
                    st.rerun()
                return # エラー時は以降の表示をスキップ
    
    # 結果表示
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.info(f"**練習したお題:** \"{st.session_state.teacher_sentence}\"")
        
        with st.container(border=True):
            st.markdown("#### AIによる認識結果")
            st.write(f"`{results['decoded_text']}`")
        
        st.markdown("---")

        if st.session_state.feedback:
            st.markdown("#### AIコーチから")
            # フィードバック文字列を "総合スコア:" と "ワンポイントアドバイス:" で分割して表示
            feedback_parts = st.session_state.feedback.split("ワンポイントアドバイス:")
            if len(feedback_parts) >= 1:
                score_part = feedback_parts[0].replace("総合スコア:", "").strip()
                try:
                    # "X/100 点" からスコアを抽出
                    actual_score_str = score_part.split("/")[0].strip()
                    actual_score = int(actual_score_str)
                    st.metric(label="総合スコア", value=f"{actual_score} / 100 点")
                    # プログレスバーで視覚化
                    st.progress(actual_score / 100.0)
                except: # スコア形式が予期しない場合
                    st.markdown(f"**総合スコア:** {score_part}")


            if len(feedback_parts) >= 2:
                advice_part = feedback_parts[1].strip()
                st.markdown("**ワンポイントアドバイス:**")
                st.success(f"{advice_part}") # success を使って目立たせる
            elif len(feedback_parts) == 1 and "総合スコア:" not in feedback_parts[0]: # スコア部分がなくアドバイスのみの場合
                st.markdown("**ワンポイントアドバイス:**")
                st.success(f"{feedback_parts[0].strip()}")


        # 詳細な分析結果はオプションで表示
        with st.expander("より詳しい分析結果を見る (上級者向け)"):
            st.markdown(f"**単語誤り率 (WER):** {results['wer']:.2%} (低いほど良い)")
            st.markdown(f"**AIによる総合的な流暢さ:** {results['total_fluency_score']:.0f}/100 (高いほど良い)")
            if results['words_with_metrics']:
                st.markdown("**単語ごとの詳細:**")
                word_data = []
                for item in results['words_with_metrics']:
                    word_data.append({
                        "単語/トークン": item['word'],
                        "パープレキシティ": f"{item['perplexity']:.2f}",
                        "流暢さスコア": f"{item['fluency_score']:.0f}/100"
                    })
                st.dataframe(word_data, use_container_width=True, hide_index=True)
            else:
                st.write("単語ごとの詳細な分析データはありません。")
    
    st.markdown("---")
    col_nav_end1, col_nav_end2 = st.columns(2)
    with col_nav_end1:
        if st.button("もう一度同じお題で練習 (ステップ2へ)", use_container_width=True):
            st.session_state.current_step = 2
            st.session_state.analysis_results = None # 結果をクリアして再評価できるようにする
            st.session_state.feedback = ""
            st.session_state.audio_bytes_data = None # 録音もクリア
            st.rerun()
    with col_nav_end2:
        if st.button("新しいお題で練習 (ステップ1へ)", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.session_state.teacher_sentence = "" # 全てリセット
            st.session_state.analysis_results = None
            st.session_state.feedback = ""
            st.session_state.audio_bytes_data = None
            st.rerun()


# メインのルーティングロジック
if st.session_state.model_loaded: # モデルがロードされていれば各ステップを表示
    if st.session_state.current_step == 1:
        step1_generate_sentence()
    elif st.session_state.current_step == 2:
        step2_record_audio()
    elif st.session_state.current_step == 3:
        step3_show_results()
else:
    st.info("モデルの準備をお待ちください...")


st.markdown("---")
st.caption("© 2024 AI英語発音コーチ")