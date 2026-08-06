import streamlit as st
import pandas as pd
import pulp
import io

# 页面基础配置
st.set_page_config(page_title="文化祭シフト最適化システム", layout="wide")

st.title("🎉 高校文化祭 スマートシフト最適化システム")
st.markdown("生徒の参加可能時間と希望順位（第1〜第3希望）を基に、制約を厳密に満たしながら満足度を最大化するスケジュールを自動生成します。")

# --- モジュール 1: サイドバー（ファイルアップロード & パラメータ設定） ---
st.sidebar.header("📁 データ入力 & 設定")

uploaded_time_file = st.sidebar.file_uploader("1. 参加可能時間 Excel", type=["xlsx"])
uploaded_grade_file = st.sidebar.file_uploader("2. 学年 Excel", type=["xlsx"])
uploaded_pref_file = st.sidebar.file_uploader("3. 希望順位 Excel", type=["xlsx"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ シフト制約パラメータ")
min_shifts = st.sidebar.number_input("1人あたりの最小担当回数", min_value=1, max_value=10, value=3)
max_shifts = st.sidebar.number_input("1人あたりの最大担当回数", min_value=1, max_value=15, value=6)
default_demand = st.sidebar.number_input("各部門・時間帯のデフォルト必要人数", min_value=1, max_value=30, value=10)

# --- モジュール 2: 最適化の実行 ---
if uploaded_time_file and uploaded_grade_file and uploaded_pref_file:
    if st.sidebar.button("🚀 最適化を実行する", type="primary"):
        with st.spinner("数理最適化モデルを構築中... しばらくお待ちください。"):
            try:
                # Excelファイルの読み込み
                df_time = pd.read_excel(uploaded_time_file)
                df_grade = pd.read_excel(uploaded_grade_file)
                df_pref = pd.read_excel(uploaded_pref_file)

                students = df_time['生徒'].tolist()
                time_slots = [col for col in df_time.columns if col != '生徒']
                departments = sorted(df_pref['第1希望'].dropna().unique().tolist())

                # 配置可能性辞書
                available = {}
                for _, row in df_time.iterrows():
                    s = row['生徒']
                    for t in time_slots:
                        available[s, t] = int(row[t])

                # 学年辞書
                student_grade = dict(zip(df_grade['生徒'], df_grade['学年']))

                # 希望順位スコア辞書 (第1=3点, 第2=2点, 第3=1点)
                preference_score = {}
                for s in students:
                    for r in departments:
                        preference_score[s, r] = 0

                for _, row in df_pref.iterrows():
                    s = row['生徒']
                    if pd.notna(row.get('第1希望')):
                        preference_score[s, row['第1希望']] = 3
                    if pd.notna(row.get('第2希望')):
                        preference_score[s, row['第2希望']] = 2
                    if pd.notna(row.get('第3希望')):
                        preference_score[s, row['第3希望']] = 1

                # 部門必要人数
                department_demands = {t: {r: default_demand for r in departments} for t in time_slots}

                # PuLP モデル構築
                model = pulp.LpProblem("School_Festival_Real_Optimization", pulp.LpMaximize)

                active_keys = [
                    (s, t, r)
                    for s in students
                    for t in time_slots
                    for r in departments
                    if available.get((s, t), 0) == 1
                ]
                x = {
                    (s, t, r): pulp.LpVariable(f"x_{s}_{t}_{r}", cat=pulp.LpBinary)
                    for s, t, r in active_keys
                }

                def get_x(s, t, r):
                    return x.get((s, t, r), 0)

                # 目的関数
                model += pulp.lpSum(
                    preference_score.get((s, r), 0) * get_x(s, t, r)
                    for s, t, r in active_keys
                ), "Maximize_Preference_Score"

                # 制約1：部門需要充足
                for t in time_slots:
                    for r in departments:
                        required_num = department_demands.get(t, {}).get(r, 5)
                        model += (
                            pulp.lpSum(get_x(s, t, r) for s in students) == required_num,
                            f"Demand_{t}_{r}"
                        )

                # 制約2：時間重複禁止
                for s in students:
                    for t in time_slots:
                        model += (
                            pulp.lpSum(get_x(s, t, r) for r in departments) <= 1,
                            f"Time_Conflict_{s}_{t}"
                        )

                # 制約3：公平性（工時上下限）
                for s in students:
                    total_shifts = pulp.lpSum(
                        get_x(s, t, r) for t in time_slots for r in departments
                    )
                    model += total_shifts >= min_shifts, f"Fairness_Min_{s}"
                    model += total_shifts <= max_shifts, f"Fairness_Max_{s}"

                # 求解
                model.solve(pulp.PULP_CBC_CMD(msg=False))
                status = pulp.LpStatus[model.status]

                if status == 'Optimal':
                    results = []
                    for (s, t, r), var in x.items():
                        if pulp.value(var) is not None and pulp.value(var) > 0.5:
                            score = preference_score.get((s, r), 0)
                            rank_str = {3: "第1希望", 2: "第2希望", 1: "第3希望"}.get(score, "希望外")
                            results.append({
                                "生徒": s,
                                "学年": student_grade.get(s, "不明"),
                                "時間帯": t,
                                "部門": r,
                                "希望順位": rank_str,
                                "得分": score
                            })
                    df_result = pd.DataFrame(results)
                    st.session_state['df_result'] = df_result
                    st.session_state['success'] = True
                    st.success("✨ 最適化が成功しました！下記より結果を確認・ダウンロードできます。")
                else:
                    st.error(f"⚠️ 求解ステータス: {status}。制約条件（必要人数や担当回数）が厳しすぎます。設定を調整してください。")
                    st.session_state['success'] = False

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                st.session_state['success'] = False

else:
    st.info("👈 まず左側のサイドバーから 3 つの Excel ファイルをアップロードしてください。")

# --- モジュール 3 & 4: 結果の表示とダウンロード ---
if 'success' in st.session_state and st.session_state['success']:
    df_result = st.session_state['df_result']

    st.markdown("---")
    st.header("📊 最適化結果ダッシュボード")

    # サマリー指標
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総配置延べ人数", f"{len(df_result)} 件")
    with col2:
        first_count = (df_result['希望順位'] == "第1希望").sum()
        st.metric("第1希望達成数", f"{first_count} 件 ({first_count/len(df_result)*100:.1f}%)")
    with col3:
        second_count = (df_result['希望順位'] == "第2希望").sum()
        st.metric("第2希望達成数", f"{second_count} 件")
    with col4:
        outside_count = (df_result['希望順位'] == "希望外").sum()
        st.metric("希望外配置数", f"{outside_count} 件")

    tab1, tab2 = st.tabs(["📋 生徒別集計テーブル", "📅 時間帯・部門別シフト表"])

    with tab1:
        st.subheader("生徒ごとの担当回数と希望内訳")
        student_counts = df_result.groupby("生徒", as_index=False).agg(
            担当回数=("部門", "size"),
            第1希望数=("希望順位", lambda x: (x == "第1希望").sum()),
            第2希望数=("希望順位", lambda x: (x == "第2希望").sum()),
            第3希望数=("希望順位", lambda x: (x == "第3希望").sum()),
            希望外数=("希望順位", lambda x: (x == "希望外").sum()),
        )
        st.dataframe(student_counts, use_container_width=True)

    with tab2:
        st.subheader("時間帯・部門別の配置メンバー一覧")
        # 簡易的なピボットや一覧表示
        st.dataframe(df_result, use_container_width=True)

    # 📥 ダウンロード機能
    st.markdown("---")
    st.subheader("💾 結果のエクスポート")
    
    # 转换为 Excel 字节流供下载
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, sheet_name='詳細シフト', index=False)
        student_counts.to_excel(writer, sheet_name='生徒別集計', index=False)
    processed_data = output.getvalue()

    st.download_button(
        label="📥 すべての結果を Excel でダウンロード",
        data=processed_data,
        file_name="文化祭シフト最適化結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
